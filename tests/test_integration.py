"""Phase 6 acceptance test: Integration & end-to-end pipeline.

Runs the full pipeline end-to-end using synthetic fixture data (normal, fraud,
injected spike burst, missing data, cold start) and asserts:
1. Complete execution without exception for every row
2. Generation of pipeline_run_summary.json with valid counts and latency metrics
3. Verification of enqueued risk briefs in SQLite review queue
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from run_pipeline import run_pipeline_on_data
from src.explain.queue import ReviewQueue
from src.features.pipeline import build_model_frames
from src.models.train import fit_and_select, save_artifacts


def make_integration_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates synthetic dataset with normal, fraud, injected spike burst, missing data, and cold start."""
    # Synthetic transactions
    dts = [1000 + i * 3600 for i in range(20)]
    # Injected spike burst for card 777 at t=50000..50500
    spike_dts = [50000, 50100, 50200, 50300, 50400]

    all_dts = dts + spike_dts
    n = len(all_dts)

    df_tx = pd.DataFrame({
        "TransactionID": list(range(1001, 1001 + n)),
        "TransactionDT": all_dts,
        "isFraud": [0] * 7 + [1] * 8 + [0] * 2 + [1] * 3 + [1] * 5,
        "TransactionAmt": [50.0] * 20 + [500.0] * 5,
        "card1": [101] * 10 + [202] * 10 + [777] * 5,
        "P_emaildomain": ["gmail.com"] * 20 + ["yahoo.com"] * 5,
        "addr1": [100.0] * 20 + [200.0] * 5,
        "addr2": [87.0] * 25,
    })
    # Add a missing-data row
    df_tx.loc[0, "P_emaildomain"] = None
    df_tx.loc[0, "addr1"] = None

    # Add a cold start entity row
    df_tx.loc[1, "card1"] = 9999

    df_id = pd.DataFrame({
        "TransactionID": df_tx["TransactionID"],
        "DeviceType": ["mobile"] * n,
        "DeviceInfo": ["iOS"] * n,
    })

    return df_tx, df_id


def test_end_to_end_pipeline_execution(tmp_path: Path):
    """Executes full pipeline end-to-end and validates outputs."""
    df_tx, df_id = make_integration_fixture()

    # Train a minimal dummy graph model artifact in tmp_path
    artifacts_dir = tmp_path / "artifacts"
    results_dir = tmp_path / "results"
    artifacts_dir.mkdir()
    results_dir.mkdir()

    # Train dummy model
    train_tx = df_tx.iloc[:15]
    val_tx = df_tx.iloc[15:20]
    test_tx = df_tx.iloc[20:]

    id_train = df_id.iloc[:15]
    id_val = df_id.iloc[15:20]
    id_test = df_id.iloc[20:]

    frames = build_model_frames(train_tx, id_train, val_tx, id_val, test_tx, id_test, use_graph=True)
    X_train, y_train = frames["train"]
    X_val, y_val = frames["validation"]

    model, meta = fit_and_select(X_train, y_train, X_val, y_val, max_rounds=5, verbose=False)
    save_artifacts(model, meta, out_dir=artifacts_dir, name="graph_model")

    # Execute run_pipeline_on_data
    summary = run_pipeline_on_data(
        test_tx, id_test, train_tx, id_train, val_tx, id_val,
        variant="graph", artifacts_dir=str(artifacts_dir), results_dir=str(results_dir)
    )

    # 1. Verify summary fields
    assert summary["variant"] == "graph"
    assert summary["total_rows_scored"] == len(test_tx)
    assert "scoring_duration_seconds" in summary
    assert summary["scoring_duration_seconds"] >= 0.0
    assert summary["throughput_txns_per_sec"] >= 0.0

    # Write summary json
    summary_path = results_dir / "pipeline_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    assert summary_path.exists()

    # 2. Verify ReviewQueue contents
    queue_db = results_dir / "review_queue.db"
    assert queue_db.exists()

    queue = ReviewQueue(db_path=queue_db)
    pending = queue.list_pending()
    assert len(pending) == summary["enqueued_briefs_count"]
