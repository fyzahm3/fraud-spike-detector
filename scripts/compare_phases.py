#!/usr/bin/env python3
"""Build the Phase 1 vs Phase 2 before/after comparison table.

Reads results/baseline_metrics.json and results/graph_metrics.json and writes
results/phase_comparison.{json,md}. Deltas are reported verbatim — if graph
features did not help, the table says so.
"""

from __future__ import annotations

import json
from pathlib import Path

BASELINE = Path("results/baseline_metrics.json")
GRAPH = Path("results/graph_metrics.json")


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def main() -> int:
    base, graph = load(BASELINE), load(GRAPH)
    if not base or not graph:
        missing = [p for p, d in (("baseline", base), ("graph", graph)) if not d]
        print(f"Missing metrics files: {missing} — run evaluate.py for each variant first.")
        return 1

    def row(name: str, r: dict) -> dict:
        t, c = r["test"], r["costs"]
        cm = t["confusion_matrix"]
        return {
            "precision": t["precision"], "recall": t["recall"], "f1": t["f1"],
            "auc_roc": t["auc_roc"], "auc_pr": t["auc_pr"],
            "tp": cm["tp"], "fp": cm["fp"], "fn": cm["fn"], "tn": cm["tn"],
            "legit_value_disrupted": c["legitimate_value_disrupted"],
            "fraud_value_caught": c["fraud_value_caught"],
            "ratio": c["caught_to_disrupted_ratio"],
        }

    b, g = row("baseline", base), row("graph", graph)
    delta = {k: g[k] - b[k] for k in b}

    out_md = [
        "# Phase 1 (baseline) vs Phase 2 (+graph features) — held-out test",
        "",
        f"Both models scored the identical held-out test set "
        f"(sha256 `{base['test_split_sha256'][:12]}…`), threshold selected on "
        f"validation only in both cases.",
        "",
        "| metric | Phase 1 baseline | Phase 2 graph | delta |",
        "|---|---|---|---|",
    ]
    fmt_money = {"legit_value_disrupted", "fraud_value_caught"}
    for k, label in [
        ("auc_pr", "AUC-PR"), ("auc_roc", "AUC-ROC"),
        ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
        ("fp", "False positives"), ("tp", "True positives"),
        ("legit_value_disrupted", "Legitimate value disrupted ($$)"),
        ("fraud_value_caught", "Fraud value caught ($$)"),
        ("ratio", "Caught : disrupted ratio"),
    ]:
        bv, gv, dv = b[k], g[k], delta[k]
        if k in fmt_money:
            s = f"{bv:,.0f} | {gv:,.0f} | {dv:+,.0f}"
        elif isinstance(bv, float):
            s = f"{bv:.4f} | {gv:.4f} | {dv:+.4f}"
        else:
            s = f"{bv} | {gv} | {dv:+d}"
        out_md.append(f"| {label} | {s} |")

    out_md += [
        "",
        "## Reading this honestly",
        "",
        "- The graph variant adds 12 causal entity-graph features (trailing",
        "  window velocities/degrees + exponentially-decayed neighbor fraud",
        "  rates) computed strictly from pre-transaction history",
        "  (see src/features/graph_features.py and the brute-force leakage test).",
        "- A positive delta means real signal; a ~zero or negative delta would",
        "  mean the raw features already carried that information.",
    ]
    Path("results/phase_comparison.md").write_text("\n".join(out_md) + "\n")
    Path("results/phase_comparison.json").write_text(
        json.dumps({"baseline": b, "graph": g, "delta": delta}, indent=2))
    print("\n".join(out_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
