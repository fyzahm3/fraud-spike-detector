"""Causal rolling-window spike scoring and event extraction (Phase 3).

Individual transaction risk is distinct from a "spike."
This module aggregates per-transaction risk scores over trailing time windows
(1h, 24h) per entity (default: card1) and normalizes recent activity against
the entity's historical average baseline.

Defense-only reminder: Flagging and explaining only. No automated blocking or action.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import numpy as np
import pandas as pd

SPIKE_WINDOW_1H = 3_600
SPIKE_WINDOW_24H = 86_400

SPIKE_FEATURE_NAMES = [
    "spike_risk_cnt_1h",       # count of transactions above the risk threshold, trailing 1h
    "spike_risk_amt_1h",       # sum of TransactionAmt for those, trailing 1h
    "spike_risk_cnt_24h",      # count of transactions above the risk threshold, trailing 24h
    "spike_risk_amt_24h",      # sum of TransactionAmt for those, trailing 24h
    "spike_baseline_ratio_24h",  # current 24h risk count / entity's historical average risk count
]


class SpikeScorer:
    """Causal, streaming rolling-window aggregation of per-transaction risk
    scores, grouped by entity (default: card1). Same usage pattern as
    GraphFeatureBuilder: process() can be called once on a full frame or
    repeatedly on sequential chunks — state persists across calls.
    """

    def __init__(self, entity_col: str = "card1", risk_threshold: float = 0.5) -> None:
        self.entity_col = entity_col
        self.risk_threshold = float(risk_threshold)
        # Per-entity state:
        #   win1h[ent]  -> deque[(t, amt, score)]
        #   win24h[ent] -> deque[(t, amt, score)]
        #   history[ent]-> [first_t, total_risk_count]
        self._win1h: dict[str, deque] = {}
        self._win24h: dict[str, deque] = {}
        self._history: dict[str, list] = {}

    def transform(self, df: pd.DataFrame, risk_scores: np.ndarray) -> pd.DataFrame:
        """One-shot convenience wrapper."""
        return self.process(df, risk_scores)

    def process(self, df: pd.DataFrame, risk_scores: np.ndarray) -> pd.DataFrame:
        n = len(df)
        out = {name: np.zeros(n, dtype=np.float32) for name in SPIKE_FEATURE_NAMES}
        if n == 0:
            return pd.DataFrame(out, index=df.index)

        dt = df["TransactionDT"].to_numpy(dtype=np.int64)
        tid = df["TransactionID"].to_numpy(dtype=np.int64)
        amt = df["TransactionAmt"].to_numpy(dtype=np.float64)
        entities = (df[self.entity_col].to_numpy()
                    if self.entity_col in df else np.array([np.nan] * n, dtype=object))
        scores = np.asarray(risk_scores, dtype=np.float64)

        win1h = self._win1h
        win24h = self._win24h
        history = self._history

        def prune(ent: str, t: int) -> None:
            q1 = win1h.get(ent)
            if q1:
                while q1 and q1[0][0] <= t - SPIKE_WINDOW_1H:
                    q1.popleft()
            q24 = win24h.get(ent)
            if q24:
                while q24 and q24[0][0] <= t - SPIKE_WINDOW_24H:
                    q24.popleft()

        order = np.lexsort((tid, dt))
        for i in order:
            t = int(dt[i])
            ent_val = entities[i]
            if pd.isna(ent_val) or ent_val is None:
                continue
            ent_key = str(int(ent_val)) if isinstance(ent_val, (int, float, np.integer, np.floating)) and not pd.isna(ent_val) else str(ent_val)

            # ---------------- READ phase: prior state only ----------------
            prune(ent_key, t)
            q1 = win1h.get(ent_key)
            q24 = win24h.get(ent_key)

            cnt1 = len(q1) if q1 else 0
            amt1 = sum(item[1] for item in q1) if q1 else 0.0
            cnt24 = len(q24) if q24 else 0
            amt24 = sum(item[1] for item in q24) if q24 else 0.0

            hist = history.get(ent_key)
            if hist is None or hist[1] == 0:
                baseline_ratio = 0.0
            else:
                first_t, total_risk_cnt = hist
                # Active days span up to current time t
                days = max(1.0, (t - first_t) / 86_400.0)
                hist_avg_24h = total_risk_cnt / days
                baseline_ratio = (cnt24 / hist_avg_24h) if hist_avg_24h > 0 else 0.0

            out["spike_risk_cnt_1h"][i] = float(cnt1)
            out["spike_risk_amt_1h"][i] = float(amt1)
            out["spike_risk_cnt_24h"][i] = float(cnt24)
            out["spike_risk_amt_24h"][i] = float(amt24)
            out["spike_baseline_ratio_24h"][i] = float(baseline_ratio)

            # ------------- WRITE phase: row joins history if risky -------------
            s_i = float(scores[i])
            if s_i >= self.risk_threshold:
                win1h.setdefault(ent_key, deque()).append((t, amt[i], s_i))
                win24h.setdefault(ent_key, deque()).append((t, amt[i], s_i))
                if hist is None:
                    history[ent_key] = [t, 1]
                else:
                    hist[1] += 1

        return pd.DataFrame(out, index=df.index)


@dataclass
class SpikeEvent:
    entity_id: str
    window_start: int          # TransactionDT
    window_end: int
    transaction_ids: list[int]
    aggregate_risk_score: float
    baseline_ratio: float


def detect_spike_events(
    df: pd.DataFrame,
    risk_scores: np.ndarray,
    threshold: float,
    entity_col: str = "card1",
    max_gap_seconds: int = 86_400,
) -> list[SpikeEvent]:
    """Groups contiguous flagged rows per entity into single SpikeEvent objects.

    A spike is a PATTERN across a window, not one transaction — a single
    very-high-risk transaction with no surrounding cluster must NOT produce a
    SpikeEvent; it stays a transaction-level flag only.
    """
    scores = np.asarray(risk_scores, dtype=float)
    flagged_mask = scores >= threshold
    if not np.any(flagged_mask):
        return []

    sub_df = df[flagged_mask].copy()
    sub_df["_score"] = scores[flagged_mask]

    # Calculate spike features for ratio lookup if available
    scorer = SpikeScorer(entity_col=entity_col, risk_threshold=threshold)
    spike_feats = scorer.process(df, scores)
    sub_df["_ratio"] = spike_feats["spike_baseline_ratio_24h"][flagged_mask].to_numpy()

    events: list[SpikeEvent] = []

    for ent, group in sub_df.groupby(entity_col, observed=True):
        group_sorted = group.sort_values(by=["TransactionDT", "TransactionID"])
        current_cluster: list[pd.Series] = []
        last_dt = None

        for _, row in group_sorted.iterrows():
            curr_dt = int(row["TransactionDT"])
            if not current_cluster:
                current_cluster.append(row)
            else:
                if curr_dt - last_dt <= max_gap_seconds:
                    current_cluster.append(row)
                else:
                    if len(current_cluster) >= 2:
                        events.append(_create_spike_event(ent, current_cluster))
                    current_cluster = [row]
            last_dt = curr_dt

        if len(current_cluster) >= 2:
            events.append(_create_spike_event(ent, current_cluster))

    return events


def _create_spike_event(entity: str | float | int, rows: list[pd.Series]) -> SpikeEvent:
    tids = [int(r["TransactionID"]) for r in rows]
    dts = [int(r["TransactionDT"]) for r in rows]
    scores = [float(r["_score"]) for r in rows]
    ratios = [float(r["_ratio"]) for r in rows]

    ent_id = str(int(entity)) if isinstance(entity, (int, float, np.integer, np.floating)) and not pd.isna(entity) else str(entity)

    return SpikeEvent(
        entity_id=ent_id,
        window_start=min(dts),
        window_end=max(dts),
        transaction_ids=tids,
        aggregate_risk_score=float(np.mean(scores)),
        baseline_ratio=float(np.max(ratios)),
    )


def select_spike_threshold_on_validation(
    df: pd.DataFrame,
    risk_scores: np.ndarray,
    entity_col: str = "card1",
    grid_size: int = 50,
) -> float:
    """Select the risk threshold that maximizes F1 score of spike detection on validation data.

    Uses validation data ONLY.
    """
    lo, hi = float(np.min(risk_scores)), float(np.max(risk_scores))
    if lo == hi:
        return lo

    candidates = np.linspace(lo, hi, grid_size)
    best_t, best_f1 = hi, -1.0

    y_true = df["isFraud"].to_numpy(dtype=int) if "isFraud" in df.columns else np.zeros(len(df), dtype=int)

    # Compute actual fraud clusters (spikes) in ground truth
    fraud_df = df[y_true == 1]
    actual_spikes_count = 0
    for _, group in fraud_df.groupby(entity_col, observed=True):
        group_sorted = group.sort_values(by=["TransactionDT", "TransactionID"])
        cluster_len = 0
        last_dt = None
        for _, row in group_sorted.iterrows():
            curr_dt = int(row["TransactionDT"])
            if last_dt is None or curr_dt - last_dt <= SPIKE_WINDOW_24H:
                cluster_len += 1
            else:
                if cluster_len >= 2:
                    actual_spikes_count += 1
                cluster_len = 1
            last_dt = curr_dt
        if cluster_len >= 2:
            actual_spikes_count += 1

    if actual_spikes_count == 0:
        return float(np.median(risk_scores))

    for t in candidates:
        events = detect_spike_events(df, risk_scores, threshold=t, entity_col=entity_col)
        tp, fp = 0, 0
        for ev in events:
            # Check if this detected spike contains any actual fraud
            sub_fraud = df[df["TransactionID"].isin(ev.transaction_ids)]["isFraud"].sum() if "isFraud" in df.columns else 0
            if sub_fraud > 0:
                tp += 1
            else:
                fp += 1
        fn = max(0, actual_spikes_count - tp)
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    return best_t
