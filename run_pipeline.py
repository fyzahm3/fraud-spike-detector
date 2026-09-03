#!/usr/bin/env python3
"""Integration pipeline & end-to-end execution (Phase 6).

Runs full fraud-spike detector pipeline:
1. Data loading + manifest integrity check (or optional custom batch CSV)
2. Feature extraction (baseline or causal graph features)
3. Model risk scoring (XGBoost)
4. Rolling window spike scoring (Phase 3 SpikeScorer + detect_spike_events)
5. Explainable escalation & review queue enqueueing (Phase 4 RiskBrief + ReviewQueue)
6. Latency & throughput benchmarking -> results/pipeline_run_summary.json

Usage:
    python run_pipeline.py [--variant baseline|graph] [--data-dir data/raw]
                           [--artifacts-dir artifacts] [--results-dir results]
                           [--batch path/to/batch.csv]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.data.loader import load_dataset
from src.data.split import time_based_split, verify_against_manifest
from src.explain.queue import ReviewQueue
from src.explain.risk_brief import ContributingFactor, generate_risk_brief
from src.features.pipeline import align_to_training, build_model_frames
from src.models.train import load_artifacts
from src.spike.spike_scorer import SpikeScorer, detect_spike_events


def run_pipeline_on_data(
    test_tx: pd.DataFrame,
    id_test: pd.DataFrame,
    train_tx: pd.DataFrame,
    id_train: pd.DataFrame,
    val_tx: pd.DataFrame,
    id_val: pd.DataFrame,
    variant: str = "graph",
    artifacts_dir: str = "artifacts",
    results_dir: str = "results",
) -> dict:
    model, meta = load_artifacts(artifacts_dir, name=f"{variant}_model")
    threshold = float(meta["threshold"])

    print(f"Featurizing and scoring {len(test_tx):,} transactions ({variant} variant) ...")
    t0 = time.perf_counter()

    # 1. Featurize
    frames = build_model_frames(
        train_tx, id_train, val_tx, id_val, test_tx, id_test,
        use_graph=(variant == "graph")
    )
    X_test_raw, y_test = frames["test"]
    X_test = align_to_training(X_test_raw, meta["feature_names"])

    # 2. Predict risk scores
    proba = model.predict_proba(X_test)[:, 1]
    t1 = time.perf_counter()

    scoring_duration = t1 - t0
    throughput_tps = len(test_tx) / scoring_duration if scoring_duration > 0 else 0.0

    print(f"Scoring completed in {scoring_duration:.3f}s ({throughput_tps:,.1f} txns/sec)")

    # 3. Spike Scoring
    scorer = SpikeScorer(entity_col="card1", risk_threshold=threshold)
    spike_feats = scorer.process(test_tx, proba)

    spike_events = detect_spike_events(
        test_tx, proba, threshold=threshold, entity_col="card1"
    )
    print(f"Detected {len(spike_events)} multi-transaction spike events")

    # 4. Explainability & Review Queue
    db_path = Path(results_dir) / "review_queue.db"
    queue = ReviewQueue(db_path=db_path)

    flagged_mask = proba >= threshold
    flagged_indices = np.where(flagged_mask)[0]
    print(f"Total transactions flagged at threshold {threshold:.4f}: {len(flagged_indices)}")

    queue_ids: list[int] = []

    # Helper for top contributing factors per row
    feature_names = list(X_test.columns)

    # Process flagged transactions (up to 20 for brief generation sample in end-to-end run)
    sample_flagged_indices = flagged_indices[:20] if len(flagged_indices) > 20 else flagged_indices

    api_key_set = bool(os.getenv("GEMINI_API_KEY"))

    for idx in sample_flagged_indices:
        row_dict = test_tx.iloc[idx].to_dict()
        row_dict["model_score"] = float(proba[idx])
        row_dict["flagged_type"] = "transaction"

        row_feats = X_test.iloc[idx]
        # Top 3 highest magnitude numeric features
        num_feats = row_feats.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        num_vals = num_feats.to_numpy(dtype=float)
        top_idx = np.argsort(np.abs(num_vals))[::-1][:3]
        factors = [
            ContributingFactor(
                feature=feature_names[i],
                value=float(num_vals[i]),
                direction="increases_risk" if num_vals[i] > 0 else "decreases_risk",
            )
            for i in top_idx
        ]

        amt = float(row_dict.get("TransactionAmt", 50.0))

        if api_key_set:
            try:
                brief = generate_risk_brief(row_dict, factors, cost_estimate=amt, risk_threshold=threshold)
                qid = queue.enqueue(brief)
                queue_ids.append(qid)
            except Exception as e:
                print(f"Warning: Brief generation failed for transaction: {e}")
        else:
            # Fallback mock brief for queue test when no API key provided
            from src.explain.risk_brief import RiskBrief
            brief = RiskBrief(
                entity_id=str(row_dict.get("card1", "unknown")),
                flagged_type="transaction",
                model_score=float(proba[idx]),
                top_factors=factors,
                confidence="high" if abs(float(proba[idx]) - threshold) >= 0.25 else "medium",
                estimated_fp_cost=amt,
                recommended_action="hold_for_review",
                summary_text=f"Flagged transaction for card {row_dict.get('card1')} with risk score {proba[idx]:.4f}.",
            )
            qid = queue.enqueue(brief)
            queue_ids.append(qid)

    # Process spike events
    for spike in spike_events[:10]:
        factors = [
            ContributingFactor(feature="spike_baseline_ratio_24h", value=round(spike.baseline_ratio, 2), direction="increases_risk"),
            ContributingFactor(feature="spike_transaction_count", value=len(spike.transaction_ids), direction="increases_risk"),
        ]

        if api_key_set:
            try:
                brief = generate_risk_brief(spike, factors, cost_estimate=1000.0, risk_threshold=threshold)
                qid = queue.enqueue(brief)
                queue_ids.append(qid)
            except Exception as e:
                print(f"Warning: Brief generation failed for spike: {e}")
        else:
            from src.explain.risk_brief import RiskBrief
            brief = RiskBrief(
                entity_id=str(spike.entity_id),
                flagged_type="spike",
                model_score=float(spike.aggregate_risk_score),
                top_factors=factors,
                confidence="high",
                estimated_fp_cost=1000.0,
                recommended_action="hold_for_review",
                summary_text=f"Spike event for entity {spike.entity_id} spanning {len(spike.transaction_ids)} transactions.",
            )
            qid = queue.enqueue(brief)
            queue_ids.append(qid)

    summary = {
        "variant": variant,
        "total_rows_scored": len(test_tx),
        "total_flagged_transactions": len(flagged_indices),
        "total_spikes_detected": len(spike_events),
        "scoring_duration_seconds": round(scoring_duration, 4),
        "throughput_txns_per_sec": round(throughput_tps, 2),
        "risk_threshold": threshold,
        "enqueued_briefs_count": len(queue_ids),
        "sample_queue_ids": queue_ids[:5],
    }

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="graph", choices=["baseline", "graph"])
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--manifest", default="results/split_manifest.json")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--batch", default=None, help="Optional CSV path for custom batch scoring demo")

    args = parser.parse_args(argv)

    if args.batch and Path(args.batch).exists():
        print(f"Loading custom batch from {args.batch} ...")
        batch_df = pd.read_csv(args.batch)
        id_df = pd.DataFrame({"TransactionID": batch_df["TransactionID"]})
        # Use batch as test slice, minimal synthetic train/val slices
        summary = run_pipeline_on_data(
            batch_df, id_df, batch_df, id_df, batch_df, id_df,
            variant=args.variant, artifacts_dir=args.artifacts_dir, results_dir=args.results_dir
        )
    else:
        print("Loading dataset and verifying split integrity ...")
        transactions, identity = load_dataset(args.data_dir, "train")
        train_tx, val_tx, test_tx = time_based_split(transactions)
        id_train = identity[identity["TransactionID"].isin(train_tx["TransactionID"])]
        id_val = identity[identity["TransactionID"].isin(val_tx["TransactionID"])]
        id_test = identity[identity["TransactionID"].isin(test_tx["TransactionID"])]

        verify_against_manifest(
            args.manifest,
            {"train": train_tx, "validation": val_tx, "test": test_tx}
        )

        summary = run_pipeline_on_data(
            test_tx, id_test, train_tx, id_train, val_tx, id_val,
            variant=args.variant, artifacts_dir=args.artifacts_dir, results_dir=args.results_dir
        )

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "pipeline_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\nPipeline Run Summary written to {summary_path}:")
    print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
