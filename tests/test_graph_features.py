"""Phase-2 acceptance tests: graph features must be strictly causal.

The single most important test in the project lives here:
test_features_match_brute_force_past_only recomputes every feature for every
row from ONLY the rows that precede it in (TransactionDT, TransactionID)
order, and demands equality with the fast single-pass builder.

Also covered: future-corruption invariance, window boundary semantics
(half-open), cold-start behavior, and determinism.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.features.graph_features import (
    DECAY_TAU_SECONDS,
    FEATURE_NAMES,
    WINDOW_24H,
    WINDOW_7D,
    GraphFeatureBuilder,
)

# ------------------------------------------------------------- helpers --

def make_graph_fixture(n: int = 120, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_dt = np.sort(rng.choice(np.arange(0, 30 * 86_400), size=n - 4,
                                 replace=False)).astype(np.int64)
    dt = list(base_dt) + [base_dt[50], base_dt[50], base_dt[50] + 1,
                          base_dt[80]]  # deliberate timestamp ties
    tid = list(range(1000, 1000 + n))
    df = pd.DataFrame({
        "TransactionID": tid,
        "TransactionDT": dt,
        "isFraud": rng.integers(0, 2, n),
        "TransactionAmt": np.round(rng.uniform(5, 500, n), 2),
        "card1": rng.choice([10, 11, 12, 13], n),
        "P_emaildomain": rng.choice(["a.com", "b.com", None], n),
        "addr1": rng.choice([100.0, 200.0, np.nan], n),
        "addr2": rng.choice([87.0, np.nan], n),
        # identity-table fields appear here because callers pass joined frames
        "DeviceType": rng.choice(["desktop", "mobile", None], n),
        "DeviceInfo": rng.choice(["Windows", "iOS", None], n),
    })
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _node_of(row) -> dict[str, str]:
    nodes = {}
    if not pd.isna(row["card1"]):
        nodes["card"] = f"c:{int(row['card1'])}"
    if not pd.isna(row["P_emaildomain"]):
        nodes["email"] = f"e:{row['P_emaildomain']}"
    dev = None if (pd.isna(row.get("DeviceType")) and pd.isna(row.get("DeviceInfo"))) \
        else f"{'-' if pd.isna(row.get('DeviceType')) else row['DeviceType']}|" \
             f"{'-' if pd.isna(row.get('DeviceInfo')) else row['DeviceInfo']}"
    if dev is not None:
        nodes["dev"] = f"d:{dev}"
    # mirrors builder: each side independently becomes '-' when missing
    a1, a2 = row["addr1"], row["addr2"]
    addr = None if (pd.isna(a1) and pd.isna(a2)) \
        else f"{'-' if pd.isna(a1) else int(a1)}|{'-' if pd.isna(a2) else int(a2)}"
    if addr is not None:
        nodes["addr"] = f"a:{addr}"
    return nodes


def brute_force_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deliberately naive O(n^2): recompute each feature from prior rows."""
    order = df.sort_values(["TransactionDT", "TransactionID"],
                           kind="mergesort").index.to_numpy()
    rows = df.loc[order].to_dict("records")
    out = {name: [] for name in FEATURE_NAMES}

    def decayed(prior_rows, key, t_i):
        S = W = 0.0
        for j in prior_rows:
            if key in _node_of(j).values():
                w = math.exp(-(t_i - j["TransactionDT"]) / DECAY_TAU_SECONDS)
                S += w * j["isFraud"]
                W += w
        return S / W if W > 0 else 0.0

    for pos, row in enumerate(rows):
        prior = rows[:pos]
        t_i = row["TransactionDT"]
        nodes = _node_of(row)
        card = nodes.get("card")

        same_card_24h = [j for j in prior if card and card in _node_of(j).values()
                         and t_i - j["TransactionDT"] < WINDOW_24H]
        out["g_card_cnt_24h"].append(len(same_card_24h))
        out["g_card_amt_sum_24h"].append(
            sum(j["TransactionAmt"] for j in same_card_24h))

        # Pair events exist on every prior row carrying BOTH of this row's
        # same two entities; window is half-open (t-W, t], so same-timestamp
        # prior rows count (Delta t == 0 ok).
        def distinct_pair(cp_type, anchor_type):
            if anchor_type not in nodes:
                return 0
            anchor_key = nodes[anchor_type]
            seen = set()
            for j in prior:
                nj = _node_of(j)
                if t_i - j["TransactionDT"] < WINDOW_7D \
                        and nj.get(anchor_type) == anchor_key \
                        and cp_type in nj:
                    seen.add(nj[cp_type])
            return len(seen)

        out["g_card_n_email_7d"].append(distinct_pair("email", "card") if card else 0)
        out["g_card_n_dev_7d"].append(distinct_pair("dev", "card") if card else 0)
        out["g_card_n_addr_7d"].append(distinct_pair("addr", "card") if card else 0)
        out["g_email_n_card_7d"].append(
            distinct_pair("card", "email") if "email" in nodes else 0)
        out["g_dev_n_card_7d"].append(
            distinct_pair("card", "dev") if "dev" in nodes else 0)

        out["g_card_fr_decay"].append(decayed(prior, card, t_i) if card else 0.0)
        out["g_email_fr_decay"].append(
            decayed(prior, nodes.get("email"), t_i) if "email" in nodes else 0.0)
        out["g_dev_fr_decay"].append(
            decayed(prior, nodes.get("dev"), t_i) if "dev" in nodes else 0.0)
        out["g_addr_fr_decay"].append(
            decayed(prior, nodes.get("addr"), t_i) if "addr" in nodes else 0.0)

        nbhr_types = [ty for ty in ("email", "dev", "addr") if ty in nodes]
        nbhr = [decayed(prior, nodes[ty], t_i) for ty in nbhr_types]
        out["g_nbhr_fr_mean"].append(float(np.mean(nbhr)) if nbhr else 0.0)

    result = pd.DataFrame(out, index=order)
    return result.sort_index()


# ----------------------------------------------------------------- tests --

def test_features_match_brute_force_past_only():
    """THE leakage test: builder output == brute-force strictly-past values."""
    df = make_graph_fixture()
    got = GraphFeatureBuilder().transform(df).sort_index()
    want = brute_force_features(df)

    assert list(got.columns) == FEATURE_NAMES
    for name in FEATURE_NAMES:
        a, b = got[name].to_numpy(dtype=float), want[name].to_numpy(dtype=float)
        assert np.allclose(a, b, rtol=1e-4, atol=1e-6), \
            f"feature {name} differs from brute-force past-only reference"


def test_future_rows_cannot_change_past_features():
    # Work on a time-ordered frame so "prefix" is unambiguous: the first k
    # rows in (TransactionDT, TransactionID) order.
    ordered = make_graph_fixture().sort_values(
        ["TransactionDT", "TransactionID"]).reset_index(drop=True)
    k = len(ordered) // 2
    boundary_t = int(ordered["TransactionDT"].iloc[k - 1])
    assert (ordered["TransactionDT"].iloc[k:] >= boundary_t).all()

    corrupted = ordered.copy()
    corrupted.loc[k:, "isFraud"] = 1          # worst-case label flip
    corrupted.loc[k:, "TransactionAmt"] = 9_999.0
    corrupted.loc[k:, "card1"] = 42           # new entity injection

    f_clean = GraphFeatureBuilder().transform(ordered.iloc[:k])
    f_corrupt = GraphFeatureBuilder().transform(corrupted.iloc[:k])
    pd.testing.assert_frame_equal(f_clean, f_corrupt)


def test_window_boundary_is_half_open():
    # Prior event exactly WINDOW_24H old must be EXCLUDED; one second newer
    # must be INCLUDED. Two minimal two-row scenarios, no ordering traps.
    t0 = 1_000_000

    def cnt_for(gap: int) -> float:
        df = pd.DataFrame({
            "TransactionID": [1, 2],
            "TransactionDT": [t0, t0 + gap],
            "isFraud": [1, 0],
            "TransactionAmt": [10.0, 10.0],
            "card1": [7, 7],
            "P_emaildomain": ["x.com"] * 2,
            "addr1": [np.nan] * 2,
            "addr2": [np.nan] * 2,
        })
        f = GraphFeatureBuilder().transform(df)
        return float(f["g_card_cnt_24h"].iloc[1])

    assert cnt_for(WINDOW_24H) == 0        # exactly at boundary -> excluded
    assert cnt_for(WINDOW_24H - 1) == 1    # one second inside -> included
    assert cnt_for(0) == 1                 # same timestamp, earlier id -> in


def test_cold_start_entities_default_to_zero():
    df = pd.DataFrame({
        "TransactionID": [1],
        "TransactionDT": [86_400],
        "isFraud": [1],
        "TransactionAmt": [50.0],
        "card1": [999],
        "P_emaildomain": [None],
        "addr1": [np.nan],
        "addr2": [np.nan],
    })
    f = GraphFeatureBuilder().transform(df)
    for name in FEATURE_NAMES:
        assert f[name].iat[0] == pytest.approx(0.0), name


def test_builder_deterministic():
    df = make_graph_fixture(seed=11)
    a = GraphFeatureBuilder().transform(df)
    b = GraphFeatureBuilder().transform(df)
    pd.testing.assert_frame_equal(a, b)


def test_sequential_chunks_equal_one_shot():
    """Streaming through split boundaries must equal a single full pass."""
    ordered = make_graph_fixture(90, seed=8).sort_values(
        ["TransactionDT", "TransactionID"]).reset_index(drop=True)
    k1, k2 = 30, 60

    one_shot = GraphFeatureBuilder().transform(ordered)

    b = GraphFeatureBuilder()
    parts = [b.process(ordered.iloc[:k1]),
             b.process(ordered.iloc[k1:k2]),
             b.process(ordered.iloc[k2:])]
    streamed = pd.concat(parts).sort_index()

    pd.testing.assert_frame_equal(one_shot.sort_index(), streamed)


def test_decay_weights_favor_recent_history():
    # With a stable baseline of legit activity, an inserted fraud raises the
    # card's decayed rate far more when it is recent than when it is a week
    # old (its weight decays as exp(-dt/tau)).
    def scenario(fraud_gap_before_reader: int) -> float:
        rows = {"TransactionID": list(range(1, 24)),
                "TransactionDT": [], "isFraud": [],
                "TransactionAmt": [10.0] * 23, "card1": [5] * 23,
                "P_emaildomain": [None] * 23,
                "addr1": [np.nan] * 23, "addr2": [np.nan] * 23}
        t = 0
        reader_t = 21 * 43_200  # baseline: legit every 12h, then reader
        for k in range(21):
            rows["TransactionDT"].append(t)
            rows["isFraud"].append(0)
            t += 43_200
        rows["TransactionDT"].append(reader_t - fraud_gap_before_reader)
        rows["isFraud"].append(1)                       # the inserted fraud
        rows["TransactionDT"].append(reader_t)
        rows["isFraud"].append(0)                       # the reader row
        df = pd.DataFrame(rows).sort_values(
            ["TransactionDT", "TransactionID"]).reset_index(drop=True)
        f = GraphFeatureBuilder().transform(df)
        return float(f["g_card_fr_decay"].iloc[-1])

    assert scenario(3_600) > scenario(7 * 86_400)
