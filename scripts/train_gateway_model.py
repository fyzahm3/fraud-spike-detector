"""Train the gateway model: what is knowable at the instant of authorization.

Why a second model exists
-------------------------
The main model reaches AUC-PR 0.6732 using 443 features, most of which describe
an entity's accumulated history — device graphs, email-cluster co-occurrence,
rolling velocity. None of that exists at the moment a payment is authorized. A
webhook arriving from a payment rail carries an amount, a card network, a card
type, an email domain and a timestamp, and nothing else.

So this trains on the SAME real IEEE-CIS dataset and the SAME chronological
split, restricted to only those fields. It is deliberately weaker, and the gap
between its AUC-PR and the main model's is the measured cost of the missing
history — published rather than argued.

This is also how production fraud infrastructure is actually shaped: a fast
gateway pass at authorization, and a richer re-score once the entity history
that the strong features need has accumulated.

Protocol is identical to train.py and is not relaxed because the model is
smaller:
  * chronological split, verified against results/split_manifest.json
  * early stopping and threshold selection on VALIDATION only
  * the test split is touched exactly once, to report
  * SEED / N_THREADS pinned for bit-reproducibility

Usage:
    python scripts/train_gateway_model.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_dataset
from src.data.split import time_based_split, verify_against_manifest
from src.models.metrics import (
    classification_metrics,
    cost_metrics,
    select_threshold_on_validation,
)

SEED = 42
N_THREADS = 4
EARLY_STOPPING_ROUNDS = 50
MAX_ROUNDS = 600

# Exactly the fields a payment webhook actually carries. Nothing is included
# here because it improves the score; each entry has a named counterpart in a
# Razorpay payment payload, and anything without one is left out even where it
# would obviously help.
NUMERIC_FEATURES = ["TransactionAmt", "hour_of_day", "day_of_week"]
CATEGORICAL_FEATURES = ["card4", "card6", "P_emaildomain", "addr2"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

#: What each feature corresponds to on the wire. Published with the model so the
#: mapping is auditable rather than implied.
WEBHOOK_MAPPING = {
    "TransactionAmt": "payment.amount (major units)",
    "hour_of_day": "payment.created_at -> hour, UTC",
    "day_of_week": "payment.created_at -> weekday, Monday=0",
    "card4": "payment.card.network (visa / mastercard / amex / discover)",
    "card6": "payment.card.type (credit / debit)",
    "P_emaildomain": "payment.email -> domain part",
    "addr2": "payment.card.international / issuing country code",
}


class MissingSourceDataError(RuntimeError):
    """The real dataset is unavailable. Never substituted with generated rows."""


def build_frame(tx: pd.DataFrame) -> pd.DataFrame:
    """Project a transaction frame onto the webhook-obtainable columns."""
    out = pd.DataFrame(index=tx.index)
    out["TransactionAmt"] = pd.to_numeric(tx["TransactionAmt"], errors="coerce").astype("float32")

    # TransactionDT is seconds from a fixed reference. The absolute offset is
    # meaningless, but hour-of-day and weekday are real and a webhook has them.
    dt = pd.to_numeric(tx["TransactionDT"], errors="coerce")
    out["hour_of_day"] = ((dt // 3600) % 24).astype("float32")
    out["day_of_week"] = ((dt // 86400) % 7).astype("float32")

    for col in CATEGORICAL_FEATURES:
        # As strings: addr2 is numeric in the source, and XGBoost refuses a
        # category index with a floating-point dtype. NaN stays NaN so it is
        # treated as missing rather than as the literal string "nan".
        values = tx[col]
        out[col] = values.where(values.isna(), values.astype(str))
    return out


def fit_vocab(train: pd.DataFrame) -> dict[str, pd.CategoricalDtype]:
    """Category vocabulary from TRAIN alone; unseen values become NaN later."""
    return {
        col: pd.CategoricalDtype(categories=sorted(train[col].dropna().unique()))
        for col in CATEGORICAL_FEATURES
    }


def apply_vocab(frame: pd.DataFrame, vocab: dict[str, pd.CategoricalDtype]) -> pd.DataFrame:
    out = frame.copy()
    for col, dtype in vocab.items():
        out[col] = out[col].astype(dtype)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--manifest", type=Path, default=Path("results/split_manifest.json"))
    args = parser.parse_args(argv)

    for rel in ("train_transaction.csv",):
        if not (args.data_dir / rel).exists():
            raise MissingSourceDataError(
                f"{args.data_dir / rel} is absent. This script will NOT fabricate "
                "transactions; fetch the IEEE-CIS dataset into data/raw/."
            )

    print("Loading dataset and verifying split integrity ...")
    transactions, _identity = load_dataset(args.data_dir, "train")
    train_tx, val_tx, test_tx = time_based_split(transactions)
    verify_against_manifest(
        args.manifest, {"train": train_tx, "validation": val_tx, "test": test_tx}
    )

    y_train = train_tx["isFraud"].to_numpy(dtype=int)
    y_val = val_tx["isFraud"].to_numpy(dtype=int)
    y_test = test_tx["isFraud"].to_numpy(dtype=int)

    print(f"Projecting onto {len(FEATURES)} webhook-obtainable features ...")
    vocab = fit_vocab(build_frame(train_tx))
    X_train = apply_vocab(build_frame(train_tx), vocab)[FEATURES]
    X_val = apply_vocab(build_frame(val_tx), vocab)[FEATURES]
    X_test = apply_vocab(build_frame(test_tx), vocab)[FEATURES]

    positive_rate = float(y_train.mean())
    scale_pos_weight = (1.0 - positive_rate) / positive_rate

    model = xgb.XGBClassifier(
        n_estimators=MAX_ROUNDS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=5,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=SEED,
        n_jobs=N_THREADS,
    )

    print("Training (early stopping on VALIDATION aucpr) ...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    best_iteration = int(getattr(model, "best_iteration", MAX_ROUNDS))
    print(f"  best iteration: {best_iteration}")

    # Threshold from validation only. The test split is not consulted for any
    # choice made here — it is scored once, below, to report.
    val_proba = model.predict_proba(X_val)[:, 1]
    threshold = float(select_threshold_on_validation(val_proba, y_val))
    print(f"  threshold (validation): {threshold:.6f}")

    print("Scoring the held-out test split, once ...")
    test_proba = model.predict_proba(X_test)[:, 1]
    report = classification_metrics(y_test, test_proba, threshold)
    report.update(cost_metrics(
        y_test, test_proba, threshold, test_tx["TransactionAmt"].to_numpy(dtype=float)
    ))

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(args.artifacts_dir / "gateway_model.json"))

    meta = {
        "model": "gateway",
        "purpose": (
            "Scores a payment using only the fields available at the instant of "
            "authorization. Deliberately weaker than the full model; the gap is the "
            "measured cost of the entity history that does not exist yet."
        ),
        "seed": SEED,
        "n_jobs": N_THREADS,
        "best_iteration": best_iteration,
        "threshold": threshold,
        "feature_names": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categories": {c: [str(v) for v in vocab[c].categories] for c in CATEGORICAL_FEATURES},
        "webhook_mapping": WEBHOOK_MAPPING,
        "scale_pos_weight": scale_pos_weight,
        "training_domain": (
            "IEEE-CIS US card-not-present transactions, amounts in USD. A payment "
            "on another rail or in another currency is outside this training "
            "distribution."
        ),
    }
    with open(args.artifacts_dir / "gateway_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    with open(args.results_dir / "gateway_metrics.json", "w", encoding="utf-8") as fh:
        json.dump({
            "model": "gateway",
            "n_features": len(FEATURES),
            "feature_names": FEATURES,
            "threshold": threshold,
            "best_iteration": best_iteration,
            "split": "held-out test, chronological",
            "metrics": report,
        }, fh, indent=2)

    print("\n--- gateway model, held-out test split ---")
    for key in ("auc_pr", "auc_roc", "precision", "recall", "f1"):
        if key in report:
            print(f"  {key:<10} {report[key]:.4f}")
    print(f"\nwrote {args.artifacts_dir / 'gateway_model.json'}"
          f" ({(args.artifacts_dir / 'gateway_model.json').stat().st_size/1e6:.2f} MB)")
    print(f"wrote {args.results_dir / 'gateway_metrics.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MissingSourceDataError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
