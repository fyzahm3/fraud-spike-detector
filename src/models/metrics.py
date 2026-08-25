"""Metric computation, including honest false-positive cost accounting.

The FP-cost framing required by the project:
  - every flagged-but-legitimate transaction disrupts real customer value
    (sum of TransactionAmt over false positives),
  - every caught fraud protects value (sum of TransactionAmt over true
    positives),
  - the ratio "fraud value caught / legitimate value disrupted" makes the
    precision/recall tradeoff concrete in currency terms.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, proba: np.ndarray,
                           threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    preds = (proba >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": float(roc_auc_score(y_true, proba)),
        # AUC-PR is the more informative headline number at ~3.5% positives.
        "auc_pr": float(average_precision_score(y_true, proba)),
        "confusion_matrix": {
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        },
        "n_flagged": int(preds.sum()),
        "n_rows": int(len(y_true)),
    }


def cost_metrics(y_true: np.ndarray, proba: np.ndarray,
                 threshold: float, amounts: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    preds = (proba >= threshold).astype(int)
    amounts = np.asarray(amounts, dtype=float)

    fp = (preds == 1) & (y_true == 0)
    tp = (preds == 1) & (y_true == 1)
    fn = (preds == 0) & (y_true == 1)

    legit_value_disrupted = float(amounts[fp].sum())
    fraud_value_caught = float(amounts[tp].sum())
    fraud_value_missed = float(amounts[fn].sum())

    ratio = (fraud_value_caught / legit_value_disrupted
             if legit_value_disrupted > 0 else None)
    return {
        "false_positive_count": int(fp.sum()),
        "legitimate_value_disrupted": legit_value_disrupted,
        "true_positive_count": int(tp.sum()),
        "fraud_value_caught": fraud_value_caught,
        "missed_count": int(fn.sum()),
        "fraud_value_missed": fraud_value_missed,
        "caught_to_disrupted_ratio": ratio,
    }


def select_threshold_on_validation(proba: np.ndarray, y_true: np.ndarray,
                                   grid_size: int = 199) -> float:
    """Pick the F1-maximizing threshold from a fixed quantile grid.

    Uses validation data ONLY (the caller must guarantee that); the grid is a
    deterministic quantile sweep rather than an arbitrary constant like 0.5,
    which is meaningless under class imbalance and scale_pos_weighting.
    """
    lo, hi = float(np.min(proba)), float(np.max(proba))
    candidates = np.linspace(lo, hi, grid_size)
    best_t, best_f1 = hi, -1.0
    y_true = np.asarray(y_true).astype(int)
    for t in candidates:
        preds = (proba >= t)
        tp = int((preds & (y_true == 1)).sum())
        fp = int((preds & (y_true == 0)).sum())
        fn = int((~preds & (y_true == 1)).sum())
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def format_metrics_md(report: dict) -> str:
    """Render the metrics report as human-readable markdown."""
    lines = ["# Baseline model — held-out test metrics", ""]

    def row(label, m):
        cm = m["confusion_matrix"]
        return (f"| {label} | {m['precision']:.4f} | {m['recall']:.4f} | "
                f"{m['f1']:.4f} | {m['auc_roc']:.4f} | {m['auc_pr']:.4f} | "
                f"{cm['tp']} | {cm['fp']} | {cm['fn']} | {cm['tn']} |")

    lines += [
        f"Threshold **{report['test']['threshold']:.6f}** selected on the "
        "validation split only (max F1 on a quantile grid); held-out test "
        "was scored once.",
        "",
        "| split | precision | recall | F1 | AUC-ROC | AUC-PR | TP | FP | FN | TN |",
        "|---|---|---|---|---|---|---|---|---|---|",
        row("validation (tuning)", report["validation"]),
        row("**held-out test**", report["test"]),
        "",
        "## False-positive cost (held-out test)",
        "",
        f"- Flagged transactions that were legitimate: "
        f"**{report['costs']['false_positive_count']}**, disrupting "
        f"**{report['costs']['legitimate_value_disrupted']:,.2f}** in transaction value",
        f"- Fraud correctly caught: **{report['costs']['true_positive_count']}**, "
        f"protecting **{report['costs']['fraud_value_caught']:,.2f}**",
        f"- Fraud missed (false negatives): **{report['costs']['missed_count']}**, "
        f"value **{report['costs']['fraud_value_missed']:,.2f}**",
        f"- Ratio fraud-value-caught : legitimate-value-disrupted = "
        f"**{report['costs']['caught_to_disrupted_ratio']:.3f}**",
        "",
        "_Defense-only reminder: these flags are advisory. Nothing is blocked, "
        "cancelled, or held automatically._",
    ]
    return "\n".join(lines) + "\n"
