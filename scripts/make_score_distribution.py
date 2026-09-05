#!/usr/bin/env python3
"""Commit a binned score histogram for the held-out test set (graph variant).

Reuses the exact evaluate.py loading/scoring path so the histogram reflects
real model output on real held-out rows, not a synthetic distribution.
Outputs results/score_distribution.json: fixed-width bins over [0, 1], each
carrying total/fraud/legit counts, plus the decision threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data.loader import load_dataset
from src.data.split import time_based_split, verify_against_manifest
from src.features.pipeline import build_model_frames, align_to_training
from src.models.train import load_artifacts

N_BINS = 25


def main() -> int:
    transactions, identity = load_dataset("data/raw", "train")
    train_tx, val_tx, test_tx = time_based_split(transactions)
    id_train = identity[identity["TransactionID"].isin(train_tx["TransactionID"])]
    id_val = identity[identity["TransactionID"].isin(val_tx["TransactionID"])]
    id_test = identity[identity["TransactionID"].isin(test_tx["TransactionID"])]

    verify_against_manifest("results/split_manifest.json",
                            {"train": train_tx, "validation": val_tx, "test": test_tx})

    model, meta = load_artifacts("artifacts", name="graph_model")
    threshold = meta["threshold"]

    frames = build_model_frames(train_tx, id_train, val_tx, id_val, test_tx, id_test,
                                 use_graph=True)
    X_test_raw, y_test = frames["test"]
    X_test = align_to_training(X_test_raw, meta["feature_names"])

    proba = model.predict_proba(X_test)[:, 1]
    y = y_test.to_numpy()

    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    bin_idx = np.clip(np.digitize(proba, edges[1:-1]), 0, N_BINS - 1)

    bins = []
    for b in range(N_BINS):
        mask = bin_idx == b
        n_total = int(mask.sum())
        n_fraud = int(y[mask].sum())
        bins.append({
            "lo": round(float(edges[b]), 4),
            "hi": round(float(edges[b + 1]), 4),
            "total": n_total,
            "fraud": n_fraud,
            "legit": n_total - n_fraud,
        })

    out = {
        "variant": "graph",
        "n_rows": int(len(proba)),
        "threshold": threshold,
        "n_bins": N_BINS,
        "bins": bins,
        "max_bin_total": max(b["total"] for b in bins),
    }
    Path("results/score_distribution.json").write_text(json.dumps(out, indent=2))
    print(f"wrote results/score_distribution.json ({len(proba):,} rows, "
          f"{N_BINS} bins, threshold={threshold:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
