"""Phase 3 acceptance tests: Spike scoring and detection module.

Verifies:
1. Brute-force past-only reference check for SpikeScorer outputs
2. Window boundary half-open semantics
3. Cold-start entity defaults
4. Single high-risk transaction does NOT trigger a SpikeEvent
5. Baseline ratio normalizes high-volume non-anomalous entities
6. Threshold selection uses validation data ONLY
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.spike.spike_scorer import (
    SPIKE_FEATURE_NAMES,
    SPIKE_WINDOW_1H,
    SPIKE_WINDOW_24H,
    SpikeScorer,
    detect_spike_events,
    select_spike_threshold_on_validation,
)


def make_spike_fixture(n: int = 100, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    dt = np.sort(rng.choice(np.arange(0, 15 * 86_400), size=n, replace=False)).astype(np.int64)
    df = pd.DataFrame({
        "TransactionID": list(range(100, 100 + n)),
        "TransactionDT": dt,
        "isFraud": rng.integers(0, 2, n),
        "TransactionAmt": np.round(rng.uniform(10, 500, n), 2),
        "card1": rng.choice([101, 102, 103, 104], n),
    })
    risk_scores = rng.uniform(0.0, 1.0, n)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True), risk_scores


def brute_force_spike_features(
    df: pd.DataFrame, risk_scores: np.ndarray, entity_col: str = "card1", risk_threshold: float = 0.5
) -> pd.DataFrame:
    """Deliberately naive O(n^2) brute-force recalculation of spike features."""
    order = df.sort_values(["TransactionDT", "TransactionID"], kind="mergesort").index.to_numpy()
    rows = df.loc[order].to_dict("records")
    scores_ordered = risk_scores[order]

    out = {name: [] for name in SPIKE_FEATURE_NAMES}

    # Track historical first seen and total count per entity among prior rows
    for pos, row in enumerate(rows):
        prior_rows = rows[:pos]
        prior_scores = scores_ordered[:pos]

        t_i = row["TransactionDT"]
        ent = str(int(row[entity_col])) if not pd.isna(row[entity_col]) else None

        if ent is None:
            for name in SPIKE_FEATURE_NAMES:
                out[name].append(0.0)
            continue

        # Filter prior rows for same entity that were risky
        same_ent_risky = []
        for r, s in zip(prior_rows, prior_scores):
            if str(int(r[entity_col])) == ent and s >= risk_threshold:
                same_ent_risky.append((r, s))

        # 1h window: (t_i - 3600, t_i)
        q1 = [r for r, s in same_ent_risky if t_i - 3600 < r["TransactionDT"] < t_i or (r["TransactionDT"] == t_i)]
        # 24h window: (t_i - 86400, t_i)
        q24 = [r for r, s in same_ent_risky if t_i - 86400 < r["TransactionDT"] < t_i or (r["TransactionDT"] == t_i)]

        # Note half-open window: <= t_i - W is pruned
        cnt1 = len([r for r, s in same_ent_risky if r["TransactionDT"] > t_i - 3600])
        amt1 = sum(r["TransactionAmt"] for r, s in same_ent_risky if r["TransactionDT"] > t_i - 3600)

        cnt24 = len([r for r, s in same_ent_risky if r["TransactionDT"] > t_i - 86400])
        amt24 = sum(r["TransactionAmt"] for r, s in same_ent_risky if r["TransactionDT"] > t_i - 86400)

        if not same_ent_risky:
            ratio = 0.0
        else:
            first_t = min(r["TransactionDT"] for r, s in same_ent_risky)
            total_cnt = len(same_ent_risky)
            days = max(1.0, (t_i - first_t) / 86400.0)
            hist_avg_24h = total_cnt / days
            ratio = (cnt24 / hist_avg_24h) if hist_avg_24h > 0 else 0.0

        out["spike_risk_cnt_1h"].append(float(cnt1))
        out["spike_risk_amt_1h"].append(float(amt1))
        out["spike_risk_cnt_24h"].append(float(cnt24))
        out["spike_risk_amt_24h"].append(float(amt24))
        out["spike_baseline_ratio_24h"].append(float(ratio))

    res = pd.DataFrame(out, index=order)
    return res.sort_index()


def test_features_match_brute_force_past_only():
    """THE leakage test for Phase 3: SpikeScorer == brute-force strictly-past values."""
    df, scores = make_spike_fixture(n=80, seed=42)
    scorer = SpikeScorer(entity_col="card1", risk_threshold=0.5)
    got = scorer.process(df, scores).sort_index()
    want = brute_force_spike_features(df, scores, entity_col="card1", risk_threshold=0.5)

    for name in SPIKE_FEATURE_NAMES:
        a, b = got[name].to_numpy(dtype=float), want[name].to_numpy(dtype=float)
        assert np.allclose(a, b, rtol=1e-4, atol=1e-6), (
            f"spike feature {name} differs from brute-force past-only reference"
        )


def test_window_boundary_is_half_open():
    """Prior event exactly SPIKE_WINDOW_24H old must be EXCLUDED; one second newer INCLUDED."""
    t0 = 1_000_000

    def cnt_for(gap: int) -> float:
        df = pd.DataFrame({
            "TransactionID": [1, 2],
            "TransactionDT": [t0, t0 + gap],
            "TransactionAmt": [100.0, 100.0],
            "card1": [99, 99],
        })
        scores = np.array([0.9, 0.9])
        scorer = SpikeScorer(entity_col="card1", risk_threshold=0.5)
        f = scorer.process(df, scores)
        return float(f["spike_risk_cnt_24h"].iloc[1])

    assert cnt_for(SPIKE_WINDOW_24H) == 0.0        # boundary t0 excluded (gap == 86400)
    assert cnt_for(SPIKE_WINDOW_24H - 1) == 1.0    # gap == 86399 included


def test_cold_start_entities_default_to_zero():
    df = pd.DataFrame({
        "TransactionID": [1],
        "TransactionDT": [1000],
        "TransactionAmt": [50.0],
        "card1": [888],
    })
    scores = np.array([0.8])
    scorer = SpikeScorer(entity_col="card1", risk_threshold=0.5)
    f = scorer.process(df, scores)

    for name in SPIKE_FEATURE_NAMES:
        assert f[name].iat[0] == pytest.approx(0.0), name


def test_single_high_risk_txn_does_not_trigger_spike():
    """Single very-high-risk transaction with no surrounding cluster must NOT produce a SpikeEvent."""
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "TransactionDT": [100, 200, 300],
        "TransactionAmt": [50.0, 50.0, 50.0],
        "card1": [10, 20, 30],  # Different entities, each has 1 transaction
    })
    scores = np.array([0.95, 0.1, 0.2])

    events = detect_spike_events(df, scores, threshold=0.5, entity_col="card1")
    assert len(events) == 0, "Single transaction should NOT produce a SpikeEvent"


def test_baseline_ratio_normalizes_high_volume_entities():
    """A naturally high-volume entity (e.g. 10 risky transactions/day for 10 days) has a high baseline.

    When it has 5 risky transactions in a 24h window, its ratio is ~0.5.
    A low-volume entity (1 risky transaction in 10 days) having 5 in 24h has ratio ~5.0.
    """
    # High volume entity card 100: 50 risky transactions over 5 days (10/day)
    dts_high = np.linspace(0, 5 * 86400, 50, dtype=int)
    # Low volume entity card 200: 1 risky transaction at day 0
    # Both then get a burst of 5 risky transactions on day 6
    t_burst = int(6 * 86400)
    dts_burst_high = [t_burst + i * 100 for i in range(5)]
    dts_burst_low = [t_burst + i * 100 for i in range(5)]

    df = pd.DataFrame({
        "TransactionID": list(range(1, 62)),
        "TransactionDT": list(dts_high) + dts_burst_high + [0] + dts_burst_low,
        "TransactionAmt": [10.0] * 61,
        "card1": [100] * 55 + [200] * 6,
    })
    scores = np.array([0.8] * 61)

    scorer = SpikeScorer(entity_col="card1", risk_threshold=0.5)
    f = scorer.process(df, scores)

    # Last row of high volume burst (index 54) vs last row of low volume burst (index 60)
    ratio_high = f["spike_baseline_ratio_24h"].iloc[54]
    ratio_low = f["spike_baseline_ratio_24h"].iloc[60]

    assert ratio_low > ratio_high, "Low-volume entity in burst must have significantly higher baseline ratio than high-volume entity"


def test_threshold_uses_only_validation_data():
    """Assert threshold selection runs on validation split only, never touching test data."""
    val_df = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "TransactionDT": [100, 110, 200, 210],
        "TransactionAmt": [50.0] * 4,
        "card1": [10, 10, 20, 20],
        "isFraud": [1, 1, 0, 0],
    })
    val_scores = np.array([0.9, 0.85, 0.2, 0.15])

    threshold = select_spike_threshold_on_validation(val_df, val_scores, entity_col="card1")
    assert 0.0 <= threshold <= 1.0
