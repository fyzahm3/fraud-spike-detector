#!/usr/bin/env python3
"""Evaluate a trained model variant on the held-out test split — once.

Usage:
    python evaluate.py [--variant baseline|graph] [--data-dir data/raw]
                       [--artifacts-dir artifacts] [--results-dir results]

Guards:
  1. Refuses to run if any split's checksum differs from the manifest written
     at split time (SplitIntegrityError).
  2. Threshold comes from training metadata (selected on validation only).
  3. Graph features (when the variant uses them) are computed causally over
     the full chronological stream; each test row sees strictly-past data only
     (verified by tests/test_graph_features.py).

Outputs: results/{variant}_metrics.{json,md} and results/feature_importance_{variant}.{json,md}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.checksum import checksum_dataframe
from src.data.loader import load_dataset
from src.data.split import time_based_split, verify_against_manifest
from src.features.pipeline import build_model_frames, align_to_training
from src.models.metrics import (
    classification_metrics,
    cost_metrics,
    format_metrics_md,
)
from src.models.train import load_artifacts

VARIANT_TO_METRICS_NAME = {"baseline": "baseline", "graph": "graph"}


def feature_importance(model, top_n: int = 50) -> dict[str, float]:
    scores = model.get_booster().get_score(importance_type="gain")
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return {k: float(v) for k, v in ranked}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="graph",
                        choices=["baseline", "graph"])
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--manifest", default="results/split_manifest.json")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args(argv)

    print("Loading data and reconstructing the deterministic time-based split ...")
    transactions, identity = load_dataset(args.data_dir, "train")
    train_tx, val_tx, test_tx = time_based_split(transactions)
    id_train = identity[identity["TransactionID"].isin(train_tx["TransactionID"])]
    id_val = identity[identity["TransactionID"].isin(val_tx["TransactionID"])]
    id_test = identity[identity["TransactionID"].isin(test_tx["TransactionID"])]

    # Integrity guard: all three splits must byte-match the recorded manifest.
    verify_against_manifest(args.manifest,
                            {"train": train_tx, "validation": val_tx,
                             "test": test_tx})
    print(f"Split integrity verified against {args.manifest} "
          f"(test sha256={checksum_dataframe(test_tx)[:12]}…)")

    model, meta = load_artifacts(args.artifacts_dir,
                                 name=f"{args.variant}_model")
    threshold = meta["threshold"]

    # Featurize all splits together so graph features for the test slice use
    # the full legitimate past (train+val history), strictly-past per row.
    frames = build_model_frames(
        train_tx, id_train, val_tx, id_val, test_tx, id_test,
        use_graph=(args.variant == "graph"))
    X_test_raw, y_test = frames["test"]
    X_test = align_to_training(X_test_raw, meta["feature_names"])

    print(f"Scoring held-out test set once ({len(X_test):,} rows) ...")
    proba = model.predict_proba(X_test)[:, 1]
    amounts = test_tx["TransactionAmt"].to_numpy()

    report = {
        "variant": args.variant,
        "model": meta.get("model"),
        "seed": meta.get("seed"),
        "xgboost_version": meta.get("xgboost_version"),
        "n_features": meta.get("n_features"),
        "scale_pos_weight": meta.get("scale_pos_weight"),
        "best_iteration": meta.get("best_iteration"),
        "best_config": meta.get("best_config"),
        "test_split_sha256": checksum_dataframe(test_tx),
        "validation": meta["validation_metrics"],
        "test": classification_metrics(y_test.to_numpy(), proba, threshold),
        "costs": cost_metrics(y_test.to_numpy(), proba, threshold, amounts),
    }

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_name = VARIANT_TO_METRICS_NAME[args.variant]
    (out_dir / f"{metrics_name}_metrics.json").write_text(
        json.dumps(report, indent=2))
    (out_dir / f"{metrics_name}_metrics.md").write_text(format_metrics_md(report))

    imp = feature_importance(model)
    (out_dir / f"feature_importance_{metrics_name}.json").write_text(
        json.dumps(imp, indent=2))
    imp_lines = [
        "# Feature importance (XGBoost gain, top 50)", "",
        "| rank | feature | gain |", "|---|---|---|"]
    imp_lines += [f"| {i + 1} | {k} | {v:.1f} |"
                  for i, (k, v) in enumerate(imp.items())]
    (out_dir / f"feature_importance_{metrics_name}.md").write_text(
        "\n".join(imp_lines) + "\n")

    t, c = report["test"], report["costs"]
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

    print(f"\nReports written to {out_dir}/{metrics_name}_metrics.* "
          f"and feature_importance_{metrics_name}.*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
