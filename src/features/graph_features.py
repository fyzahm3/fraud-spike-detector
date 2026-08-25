"""Causal graph-derived transaction features (Phase 2).

Entity graph: nodes are card1 values, P_emaildomain values, device clusters
(DeviceType|DeviceInfo from the identity table) and addr clusters
(addr1|addr2). Every transaction instantaneously creates edges card–email,
card–device, card–addr.

Features are computed in ONE pass over time-ordered transactions, and every
feature of row i uses only rows processed strictly before i. This mirrors a
streaming deployment: when a transaction arrives, only history is available.
Concretely:

- trailing-window counts/degrees over half-open window (t-W, t]:
    * g_card_cnt_24h, g_card_amt_sum_24h        — card velocity
    * g_card_n_email_7d / n_dev_7d / n_addr_7d  — card node degree by type
    * g_email_n_card_7d, g_dev_n_card_7d        — email/device node degree
- exponentially-decayed fraud rates per node (tau = 48h), w = exp(-dt/tau):
    * g_{card,email,dev,addr}_fr_decay          — S/W accumulated fraud mass
    * g_nbhr_fr_mean                            — mean neighbor-node rate
      (the "attention-like" aggregation: risk of the company this card keeps)

Cold start (no prior history for an entity): counts = 0, rates = 0.0.
Equal timestamps: processed in (TransactionDT, TransactionID) order; a row's
features include same-second rows preceding it in that order — exactly what
a queueing scorer would see.

tests/test_graph_features.py verifies these properties against a brute-force
reference implementation on synthetic data, plus future-corruption invariance.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import pandas as pd

WINDOW_24H = 86_400
WINDOW_7D = 604_800
DECAY_TAU_SECONDS = 48 * 3_600  # ~2 days: recent history dominates

FEATURE_NAMES = [
    "g_card_cnt_24h",
    "g_card_amt_sum_24h",
    "g_card_n_email_7d",
    "g_card_n_dev_7d",
    "g_card_n_addr_7d",
    "g_email_n_card_7d",
    "g_dev_n_card_7d",
    "g_card_fr_decay",
    "g_email_fr_decay",
    "g_dev_fr_decay",
    "g_addr_fr_decay",
    "g_nbhr_fr_mean",
]

_CARD, _EMAIL, _DEV, _ADDR = "c", "e", "d", "a"
_RATE_FEATURE_BY_TYPE = {
    _CARD: "g_card_fr_decay",
    _EMAIL: "g_email_fr_decay",
    _DEV: "g_dev_fr_decay",
    _ADDR: "g_addr_fr_decay",
}

# Distinct-degree pairs we track, as (node_type, counterparty_type).
_TRACKED_PAIRS = {
    (_CARD, _EMAIL), (_CARD, _DEV), (_CARD, _ADDR),
    (_EMAIL, _CARD), (_DEV, _CARD),
}

_PAIR_TRACKED_BOTH_WAYS = {(a, b) for a, b in _TRACKED_PAIRS} | {
    (b, a) for a, b in _TRACKED_PAIRS}


class GraphFeatureBuilder:
    """Stateful incremental builder over chronologically ordered chunks.

    Usage patterns (both identical in result):
      b = GraphFeatureBuilder(); feats = b.transform(full_df)        # one shot
      b = GraphFeatureBuilder()
      f_train = b.process(train_df); f_val = b.process(val_df); ...  # streaming

    State persists across process() calls, so later chunks see earlier
    chunks' history and nothing else — matching a deployed scorer. Chunks
    must arrive in (TransactionDT, TransactionID) order; our time-based
    splits are strictly disjoint, so feeding them in sequence equals the
    one-shot pass exactly (asserted by tests).
    """

    def __init__(self) -> None:
        # Per-node state:
        #   cnt24[node] -> deque[(t,)]
        #   amt24[node] -> [deque[(t, amt)], running_sum]
        #   deg7[node]  -> [deque[(t, cp_key)], {cp_key: active_count}]
        #   decay[node] -> [S, W, last_t]
        self._cnt24: dict[str, deque] = {}
        self._amt24: dict[str, list] = {}
        self._deg7: dict[str, list] = {}
        self._decay: dict[str, list] = {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-shot convenience wrapper (fresh state is guaranteed because
        each call site constructs its own builder)."""
        return self.process(df)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        out = {name: np.zeros(n, dtype=np.float32) for name in FEATURE_NAMES}
        if n == 0:
            return pd.DataFrame(out, index=df.index)

        dt = df["TransactionDT"].to_numpy(dtype=np.int64)
        tid = df["TransactionID"].to_numpy(dtype=np.int64)
        y = (df["isFraud"].to_numpy(dtype=np.float64)
             if "isFraud" in df.columns else None)
        amt = df["TransactionAmt"].to_numpy(dtype=np.float64)
        cards = df["card1"].to_numpy() if "card1" in df else np.array([np.nan] * n)
        emails = (df["P_emaildomain"].to_numpy()
                  if "P_emaildomain" in df else np.array([None] * n, dtype=object))
        addrs = self._addr_keys(df)
        devs = self._device_keys(df)

        cnt24, amt24, deg7, decay = (
            self._cnt24, self._amt24, self._deg7, self._decay)

        def prune(node: str, t: int) -> None:
            q = cnt24.get(node)
            if q:
                while q and q[0][0] <= t - WINDOW_24H:
                    q.popleft()
            a = amt24.get(node)
            if a:
                dq, s = a
                while dq and dq[0][0] <= t - WINDOW_24H:
                    s -= dq.popleft()[1]
                a[1] = s
            d = deg7.get(node)
            if d:
                dq, ctr = d
                while dq and dq[0][0] <= t - WINDOW_7D:
                    _, cp = dq.popleft()
                    ctr[cp] -= 1
                    if not ctr[cp]:
                        del ctr[cp]

        def decayed_rate(node: str, t: int) -> float:
            st = decay.get(node)
            if st is None:
                return 0.0
            S, W, last = st
            factor = math.exp(-(t - last) / DECAY_TAU_SECONDS)
            S *= factor
            W *= factor
            decay[node] = [S, W, t]
            return S / W if W > 0.0 else 0.0

        order = np.lexsort((tid, dt))  # primary time, secondary id
        for i in order:
            t = int(dt[i])

            nodes: list[tuple[str, str]] = []
            c = cards[i]
            if not pd.isna(c):
                nodes.append((f"{_CARD}:{int(c)}", _CARD))
            e = emails[i]
            if not pd.isna(e):
                nodes.append((f"{_EMAIL}:{e}", _EMAIL))
            d = devs[i]
            if d is not None:
                nodes.append((f"{_DEV}:{d}", _DEV))
            a = addrs[i]
            if a is not None:
                nodes.append((f"{_ADDR}:{a}", _ADDR))

            # ---------------- READ phase: prior state only ----------------
            for key, ntype in nodes:
                prune(key, t)
                if ntype == _CARD:
                    ctr = deg7[key][1] if key in deg7 else {}
                    n_by_type = {_EMAIL: 0, _DEV: 0, _ADDR: 0}
                    # distinct counterparties: one per active dict KEY
                    for cp in ctr:
                        prefix = cp.split(":", 1)[0]
                        if prefix in n_by_type:
                            n_by_type[prefix] += 1
                    out["g_card_cnt_24h"][i] = len(cnt24.get(key, ()))
                    out["g_card_amt_sum_24h"][i] = (
                        amt24[key][1] if key in amt24 else 0.0)
                    out["g_card_n_email_7d"][i] = n_by_type[_EMAIL]
                    out["g_card_n_dev_7d"][i] = n_by_type[_DEV]
                    out["g_card_n_addr_7d"][i] = n_by_type[_ADDR]
                elif ntype in (_EMAIL, _DEV):
                    name = ("g_email_n_card_7d" if ntype == _EMAIL
                            else "g_dev_n_card_7d")
                    out[name][i] = len(deg7[key][1]) if key in deg7 else 0

            rates = {}
            for key, ntype in nodes:
                r = decayed_rate(key, t)
                rates[ntype] = r
                out[_RATE_FEATURE_BY_TYPE[ntype]][i] = r
            nbhr = [rates[ty] for ty in (_EMAIL, _DEV, _ADDR) if ty in rates]
            out["g_nbhr_fr_mean"][i] = float(np.mean(nbhr)) if nbhr else 0.0

            # ------------- WRITE phase: row joins the history -------------
            yv = y[i] if y is not None else 0.0
            for key, ntype in nodes:
                if ntype == _CARD:
                    cnt24.setdefault(key, deque()).append((t,))
                    entry = amt24.get(key)
                    if entry is None:
                        amt24[key] = [deque([(t, amt[i])]), amt[i]]
                    else:
                        entry[0].append((t, amt[i]))
                        entry[1] += amt[i]
                for k2, ty2 in nodes:
                    if k2 != key and (ntype, ty2) in _PAIR_TRACKED_BOTH_WAYS:
                        entry = deg7.get(key)
                        if entry is None:
                            entry = deg7[key] = [deque(), {}]
                        entry[0].append((t, k2))
                        entry[1][k2] = entry[1].get(k2, 0) + 1
                st = decay.get(key)
                if st is None:
                    decay[key] = [yv, 1.0, t]
                else:
                    # Rescale to current time BEFORE adding so every stored
                    # contribution carries its correct exp(-dt/tau) weight.
                    f = math.exp(-(t - st[2]) / DECAY_TAU_SECONDS)
                    st[0] = st[0] * f + yv
                    st[1] = st[1] * f + 1.0
                    st[2] = t

        return pd.DataFrame(out, index=df.index)

    @staticmethod
    def _addr_keys(df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        a1 = df["addr1"].to_numpy() if "addr1" in df else np.full(n, np.nan)
        a2 = df["addr2"].to_numpy() if "addr2" in df else np.full(n, np.nan)
        keys = np.empty(n, dtype=object)
        for j in range(n):
            x, z = a1[j], a2[j]
            keys[j] = (None if _is_nan(x) and _is_nan(z)
                       else f"{'-' if _is_nan(x) else int(x)}|{'-' if _is_nan(z) else int(z)}")
        return keys

    @staticmethod
    def _device_keys(df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        info = df["DeviceInfo"].to_numpy() if "DeviceInfo" in df else np.full(n, np.nan)
        typ = df["DeviceType"].to_numpy() if "DeviceType" in df else np.full(n, np.nan)
        keys = np.empty(n, dtype=object)
        for j in range(n):
            di, ty = info[j], typ[j]
            keys[j] = (None if _is_nan(di) and _is_nan(ty)
                       else f"{'-' if _is_nan(ty) else ty}|{'-' if _is_nan(di) else di}")
        return keys


def _is_nan(v) -> bool:
    try:
        return bool(math.isnan(v))
    except TypeError:
        return False
