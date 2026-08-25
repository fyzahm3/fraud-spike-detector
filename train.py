#!/usr/bin/env python3
"""Train the Phase-1 baseline model.

Usage:
    python train.py [--data-dir data/raw] [--artifacts-dir artifacts]

Pipeline: load -> time-based split (refreshes the checksum manifest) ->
train on train split with validation-only tuning -> save artifacts.
The held-out test split is checksummed for the manifest but is NOT passed to
any training function (structural guarantee — see src/models/train.py).
"""

from __future__ import annotations

import argparse

from src.data.loader import load_dataset
from src.data.split import build_manifest, time_based_split, write_manifest
from src.models.train import save_artifacts, train_baseline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--manifest-out", default="results/split_manifest.json")
    args = parser.parse_args(argv)

    print("Loading data ...")
    transactions, identity = load_dataset(args.data_dir, "train")
    train_tx, val_tx, test_tx = time_based_split(transactions)
    # Identity rows follow their transactions into the same split.
    id_train = identity[identity["TransactionID"].isin(train_tx["TransactionID"])]
    id_val = identity[identity["TransactionID"].isin(val_tx["TransactionID"])]

    write_manifest(build_manifest(train_tx, val_tx, test_tx,
                                  source_path=args.data_dir),
                   args.manifest_out)
    print(f"Manifest refreshed at {args.manifest_out}")

    print("Training (tuning on validation only) ...")
    model, meta = train_baseline(train_tx, id_train, val_tx, id_val)
    save_artifacts(model, meta, args.artifacts_dir)

    m = meta["validation_metrics"]
    print(f"\nDone. best_config={meta['best_config']}")
    print(f"validation @ threshold {m['threshold']:.6f}: "
          f"precision={m['precision']:.4f} recall={m['recall']:.4f} "
          f"F1={m['f1']:.4f} AUC-ROC={m['auc_roc']:.4f} AUC-PR={m['auc_pr']:.4f}")
    print(f"Artifacts written to {args.artifacts_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
