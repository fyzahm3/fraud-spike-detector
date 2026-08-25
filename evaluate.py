#!/usr/bin/env python3
"""Evaluate the trained baseline on the held-out test split — exactly once.

Usage:
    python evaluate.py [--data-dir data/raw] [--artifacts-dir artifacts]

Guards:
  1. Refuses to run if the test split's checksum differs from the manifest
     written when the split was created (SplitIntegrityError otherwise).
  2. The threshold comes from training metadata (selected on validation).
  3. Writes results/baseline_metrics.{json,md}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.checksum import checksum_dataframe
from src.data.loader import load_dataset
from src.data.split import time_based_split, verify_against_manifest
from src.features.preprocess import build_features, fit_categorical_vocab, join_identity
from src.models.metrics import classification_metrics, cost_metrics, format_metrics_md
from src.models.train import load_artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--manifest", default="results/split_manifest.json")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args(argv)

    print("Loading data and reconstructing the deterministic time-based split ...")
    transactions, identity = load_dataset(args.data_dir, "train")
    train_tx, val_tx, test_tx = time_based_split(transactions)
    id_train = identity[identity["TransactionID"].isin(train_tx["TransactionID"])]
    id_test = identity[identity["TransactionID"].isin(test_tx["TransactionID"])]

    # Integrity guard: all three splits must byte-match the recorded manifest.
    verify_against_manifest(args.manifest,
                            {"train": train_tx, "validation": val_tx,
                             "test": test_tx})
    print(f"Split integrity verified against {args.manifest} "
          f"(test sha256={checksum_dataframe(test_tx)[:12]}…)")

    model, meta = load_artifacts(args.artifacts_dir)
    threshold = meta["threshold"]

    # Vocabulary fitted on the TRAIN-joined frame so feature space and
    # category sets are identical to training time.
    vocab = fit_categorical_vocab(join_identity(train_tx, id_train))
    X_test, y_test = build_features(join_identity(test_tx, id_test), vocab=vocab)
    # Column order must match training exactly.
    X_test = X_test[meta["feature_names"]]

    print(f"Scoring held-out test set once ({len(X_test):,} rows) ...")
    proba = model.predict_proba(X_test)[:, 1]
    amounts = test_tx["TransactionAmt"].to_numpy()

    report = {
        "model": meta.get("model"),
        "seed": meta.get("seed"),
        "xgboost_version": meta.get("xgboost_version"),
        "n_features": meta.get("n_features"),
        "scale_pos_weight": meta.get("scale_pos_weight"),
        "best_iteration": meta.get("best_iteration"),
        "test_split_sha256": checksum_dataframe(test_tx),
        "validation": meta["validation_metrics"],
        "test": classification_metrics(y_test.to_numpy(), proba, threshold),
        "costs": cost_metrics(y_test.to_numpy(), proba, threshold, amounts),
    }

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_metrics.json").write_text(json.dumps(report, indent=2))
    (out_dir / "baseline_metrics.md").write_text(format_metrics_md(report))

    t = report["test"]
    c = report["costs"]
    print(f"\nheld-out test @ threshold {t['threshold']:.6f}")
    print(f"  precision={t['precision']:.4f} recall={t['recall']:.4f} "
          f"F1={t['f1']:.4f}")
    print(f"  AUC-ROC={t['auc_roc']:.4f} AUC-PR={t['auc_pr']:.4f}")
    print(f"  confusion: TP={t['confusion_matrix']['tp']} "
          f"FP={t['confusion_matrix']['fp']} FN={t['confusion_matrix']['fn']} "
          f"TN={t['confusion_matrix']['tn']}")
    print(f"  FP cost: {c['false_positive_count']} legit txns disrupted, "
          f"value {c['legitimate_value_disrupted']:,.2f}")
    print(f"  caught : {c['true_positive_count']} fraud txns, "
          f"value {c['fraud_value_caught']:,.2f}")
    if c["caught_to_disrupted_ratio"] is not None:
        print(f"  ratio caught:disrupted = {c['caught_to_disrupted_ratio']:.3f}")

    print(f"\nReports written to {out_dir}/baseline_metrics.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
