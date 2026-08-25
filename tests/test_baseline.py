"""Phase-1 acceptance tests for the baseline model.

Headline guarantee: the held-out test split cannot influence training,
threshold choice, or evaluation inputs. Proven three ways:
  1. train_baseline() structurally accepts only (train, val);
     running it leaves the test frame byte-identical.
  2. Corrupting the FUTURE portion of raw data cannot change train/val
     frames produced by the time-based splitter (chronological isolation).
  3. The evaluation guard detects any tampering with recorded splits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.data.checksum import checksum_dataframe
from src.data.split import (
    SplitIntegrityError,
    build_manifest,
    time_based_split,
    verify_against_manifest,
    write_manifest,
)
from src.models.metrics import (
    classification_metrics,
    cost_metrics,
    select_threshold_on_validation,
)
from src.models.train import (
    compute_scale_pos_weight,
    sample_configs,
    train_baseline,
)

QUICK_KWARGS = dict(max_rounds=15, verbose=False)


@pytest.fixture(scope="module")
def fixture_splits(fixture_transactions):
    # module scope: one deterministic split shared by several tests
    return time_based_split(fixture_transactions)


# ------------------------------------------------------- untouchability --

def test_training_leaves_held_out_test_byte_identical(
        fixture_transactions, fixture_identity):
    train_tx, val_tx, test_tx = time_based_split(fixture_transactions)
    test_before = test_tx.copy(deep=True)
    id_train = fixture_identity[fixture_identity["TransactionID"]
                                .isin(train_tx["TransactionID"])]
    id_val = fixture_identity[fixture_identity["TransactionID"]
                              .isin(val_tx["TransactionID"])]

    train_baseline(train_tx, id_train, val_tx, id_val, **QUICK_KWARGS)

    assert checksum_dataframe(test_tx) == checksum_dataframe(test_before)


def test_future_corruption_cannot_affect_train_val_frames(fixture_transactions):
    """Chronological isolation: anything done to the most recent 15% of rows
    (the held-out region) leaves train/validation frames byte-identical."""
    corrupted = fixture_transactions.copy()
    n = len(corrupted)
    _, _, test_ref = time_based_split(corrupted)
    test_idx = corrupted.index[
        corrupted["TransactionID"].isin(test_ref["TransactionID"])]
    # Worst-case tampering: flip every label, scramble amounts, shuffle order.
    corrupted.loc[test_idx, "isFraud"] = (
        1 - corrupted.loc[test_idx, "isFraud"]).astype(int)
    corrupted.loc[test_idx, "TransactionAmt"] = np.random.default_rng(0).uniform(
        1, 10_000, len(test_idx))
    shuffled = corrupted.sample(frac=1.0, random_state=99)

    base = time_based_split(fixture_transactions)
    perturbed = time_based_split(shuffled)
    assert checksum_dataframe(base[0]) == checksum_dataframe(perturbed[0])  # train
    assert checksum_dataframe(base[1]) == checksum_dataframe(perturbed[1])  # val


def test_manifest_guard_detects_tampered_test_set(tmp_path, fixture_transactions):
    train_tx, val_tx, test_tx = time_based_split(fixture_transactions)
    manifest_path = tmp_path / "manifest.json"
    write_manifest(build_manifest(train_tx, val_tx, test_tx), manifest_path)

    verify_against_manifest(manifest_path, {
        "train": train_tx, "validation": val_tx, "test": test_tx})

    # Tamper ONE amount in the held-out set -> verification must fail loudly.
    tampered = test_tx.copy(deep=True)
    tampered.iloc[0, tampered.columns.get_loc("TransactionAmt")] += 0.01
    with pytest.raises(SplitIntegrityError, match="test"):
        verify_against_manifest(manifest_path, {
            "train": train_tx, "validation": val_tx, "test": tampered})


# ---------------------------------------------------------- reproducibility --

def test_same_seed_produces_identical_models_and_scores(
        fixture_transactions, fixture_identity):
    train_tx, val_tx, _ = time_based_split(fixture_transactions)
    id_train = fixture_identity[fixture_identity["TransactionID"]
                                .isin(train_tx["TransactionID"])]
    id_val = fixture_identity[fixture_identity["TransactionID"]
                              .isin(val_tx["TransactionID"])]

    model_a, meta_a = train_baseline(train_tx, id_train, val_tx, id_val,
                                     **QUICK_KWARGS)
    model_b, meta_b = train_baseline(train_tx, id_train, val_tx, id_val,
                                     **QUICK_KWARGS)

    assert meta_a["best_config"] == meta_b["best_config"]
    assert meta_a["threshold"] == meta_b["threshold"]
    assert meta_a["validation_metrics"]["auc_pr"] \
        == meta_b["validation_metrics"]["auc_pr"]

    from src.features.preprocess import build_features, join_identity
    X_val, _ = build_features(join_identity(val_tx, id_val))
    pa = model_a.predict_proba(X_val)[:, 1]
    pb = model_b.predict_proba(X_val)[:, 1]
    np.testing.assert_array_equal(pa, pb)  # bit-identical probabilities


def test_sampled_configs_are_deterministic():
    assert sample_configs(10) == sample_configs(10)


# ------------------------------------------------------------ metric math --

def test_cost_math_hand_computed():
    # Hand-checkable case: threshold 0.5 on 4 transactions.
    y_true = np.array([0, 0, 1, 1])
    proba = np.array([0.9, 0.2, 0.8, 0.3])
    amounts = np.array([100.0, 200.0, 400.0, 800.0])
    costs = cost_metrics(y_true, proba, threshold=0.5, amounts=amounts)

    # preds = [1, 0, 1, 0] -> TP: idx2 (400), FP: idx0 (100),
    # FN: idx3 (800), TN: idx1.
    assert costs["false_positive_count"] == 1
    assert costs["legitimate_value_disrupted"] == pytest.approx(100.0)
    assert costs["true_positive_count"] == 1
    assert costs["fraud_value_caught"] == pytest.approx(400.0)
    assert costs["missed_count"] == 1
    assert costs["fraud_value_missed"] == pytest.approx(800.0)
    assert costs["caught_to_disrupted_ratio"] == pytest.approx(4.0)

    m = classification_metrics(y_true, proba, threshold=0.5)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_threshold_uses_only_validation_data():
    """Threshold must be optimal for the validation scores passed in — the
    function signature has no test parameter, so prove optimality directly."""
    y_val = np.array([0] * 50 + [1] * 50)
    proba_val = np.concatenate([np.linspace(0.0, 0.45, 50),
                                np.linspace(0.55, 1.0, 50)])
    t = select_threshold_on_validation(proba_val, y_val)
    assert 0.45 <= t <= 0.55
    m = classification_metrics(y_val, proba_val, threshold=t)
    assert m["f1"] == pytest.approx(1.0)


def test_scale_pos_weight_from_train_only():
    y = np.array([0] * 90 + [1] * 10)
    assert compute_scale_pos_weight(y) == pytest.approx(9.0)
    with pytest.raises(ValueError, match="no positives"):
        compute_scale_pos_weight(np.array([0, 0, 0]))


# --------------------------------------------------- real-data smoke test --

@pytest.mark.skipif(not Path("data/raw/train_transaction.csv").exists(),
                    reason="real dataset not downloaded")
def test_real_pipeline_smoke_small(fixture_transactions):
    """End-to-end plumbing on REAL schema: subsample train/val, tiny model."""
    from src.data.loader import load_dataset

    tx, ident = load_dataset("data/raw", "train")
    tx = tx.iloc[:20_000]
    train_tx, val_tx, _ = time_based_split(tx)
    id_train = ident[ident["TransactionID"].isin(train_tx["TransactionID"])]
    id_val = ident[ident["TransactionID"].isin(val_tx["TransactionID"])]
    if train_tx["isFraud"].sum() == 0 or val_tx["isFraud"].sum() == 0:
        pytest.skip("subsample lacks both classes")
    model, meta = train_baseline(train_tx, id_train, val_tx, id_val,
                                 max_rounds=30, verbose=False)
    assert 0.0 < meta["threshold"] < 1.0
    assert meta["n_features"] > 300
