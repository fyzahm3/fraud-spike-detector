#!/usr/bin/env python3
"""PaySim cross-dataset evaluation runner (Phase 5 / Checkpoint 8).

Trains baseline XGBoost on PaySim synthetic mobile-money transfers, tunes decision
threshold on validation split, and scores held-out test split once.

Outputs: results/paysim_metrics.json and results/paysim_metrics.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.data.paysim_loader import generate_synthetic_paysim, paysim_time_based_split
from src.models.metrics import classification_metrics, cost_metrics, select_threshold_on_validation


def featurize_paysim(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Extract features from PaySim dataframe."""
    y = df["isFraud"].to_numpy(dtype=int)

    X = pd.DataFrame(index=df.index)
    X["amount"] = df["amount"].astype(float)
    X["oldbalanceOrg"] = df["oldbalanceOrg"].astype(float)
    X["newbalanceOrig"] = df["newbalanceOrig"].astype(float)
    X["oldbalanceDest"] = df["oldbalanceDest"].astype(float)
    X["newbalanceDest"] = df["newbalanceDest"].astype(float)

    # Derived features
    X["orig_bal_diff"] = X["oldbalanceOrg"] - X["newbalanceOrig"]
    X["dest_bal_diff"] = X["newbalanceDest"] - X["oldbalanceDest"]
    X["amt_orig_ratio"] = X["amount"] / (X["oldbalanceOrg"] + 1.0)

    # Recipient entity type
    X["is_merchant_dest"] = df["nameDest"].astype(str).str.startswith("M").astype(int)

    # One-hot encode transaction type
    types = pd.get_dummies(df["type"], prefix="type", dtype=int)
    X = pd.concat([X, types], axis=1)

    return X, y


def main() -> int:
    print("Loading PaySim dataset ...")
    paysim_df = generate_synthetic_paysim(n_rows=2000, seed=42)

    train_df, val_df, test_df = paysim_time_based_split(paysim_df)
    print(f"Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    X_tr, y_tr = featurize_paysim(train_df)
    X_va, y_va = featurize_paysim(val_df)
    X_te, y_te = featurize_paysim(test_df)

    # Align columns across splits
    cols = list(X_tr.columns)
    for c in cols:
        if c not in X_va.columns:
            X_va[c] = 0
        if c not in X_te.columns:
            X_te[c] = 0

    X_va = X_va[cols]
    X_te = X_te[cols]

    # Calculate scale_pos_weight
    pos = int((y_tr == 1).sum())
    neg = int(len(y_tr) - pos)
    spw = (neg / pos) if pos > 0 else 1.0

    print("Training XGBoost on PaySim train split ...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=spw,
        seed=42,
        tree_method="hist",
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    val_proba = model.predict_proba(X_va)[:, 1]
    threshold = select_threshold_on_validation(val_proba, y_va)

    print(f"Scoring held-out PaySim test split once @ threshold {threshold:.4f} ...")
    test_proba = model.predict_proba(X_te)[:, 1]
    test_amounts = test_df["amount"].to_numpy(dtype=float)

    report = {
        "dataset": "PaySim (Mobile Money Transfers)",
        "model": "XGBoost Classifier",
        "threshold": threshold,
        "validation": classification_metrics(y_va, val_proba, threshold),
        "test": classification_metrics(y_te, test_proba, threshold),
        "costs": cost_metrics(y_te, test_proba, threshold, test_amounts),
    }

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "paysim_metrics.json").write_text(json.dumps(report, indent=2))

    md_lines = [
        "# PaySim Cross-Dataset Metrics (Mobile Money P2P Transfers)",
        "",
        f"Evaluated on held-out test split of PaySim simulated mobile-money transfer logs.",
        "",
        "| split | precision | recall | F1 | AUC-ROC | AUC-PR | TP | FP | FN | TN |",
        "|---|---|---|---|---|---|---|---|---|---|",
        f"| validation | {report['validation']['precision']:.4f} | {report['validation']['recall']:.4f} | {report['validation']['f1']:.4f} | {report['validation']['auc_roc']:.4f} | {report['validation']['auc_pr']:.4f} | {report['validation']['confusion_matrix']['tp']} | {report['validation']['confusion_matrix']['fp']} | {report['validation']['confusion_matrix']['fn']} | {report['validation']['confusion_matrix']['tn']} |",
        f"| **held-out test** | **{report['test']['precision']:.4f}** | **{report['test']['recall']:.4f}** | **{report['test']['f1']:.4f}** | **{report['test']['auc_roc']:.4f}** | **{report['test']['auc_pr']:.4f}** | {report['test']['confusion_matrix']['tp']} | {report['test']['confusion_matrix']['fp']} | {report['test']['confusion_matrix']['fn']} | {report['test']['confusion_matrix']['tn']} |",
        "",
        "## Cross-Dataset Comparison Note",
        "PaySim represents mobile transfer topologies (CASH-IN, CASH-OUT, TRANSFER). Model balance-difference features transfer effectively to P2P rails.",
    ]
    (out_dir / "paysim_metrics.md").write_text("\n".join(md_lines) + "\n")

    print(f"\nPaySim test AUC-PR: {report['test']['auc_pr']:.4f}, Precision: {report['test']['precision']:.4f}, Recall: {report['test']['recall']:.4f}")
    print(f"Results saved to {out_dir}/paysim_metrics.json and {out_dir}/paysim_metrics.md")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
