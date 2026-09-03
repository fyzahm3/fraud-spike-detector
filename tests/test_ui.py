"""Acceptance tests for Review Queue UI Dashboard (app.py).

Verifies:
1. UI Flask app initializes and serves pending queue items cleanly.
2. Resolve endpoint updates SQLite review queue status and appends to audit log.
3. Defense-only security scan: UI module contains zero payment-action calls.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from app import app as flask_app
from src.explain.queue import ReviewQueue
from src.explain.risk_brief import ContributingFactor, RiskBrief


@pytest.fixture
def client(tmp_path: Path):
    db_path = tmp_path / "ui_test_queue.db"
    queue = ReviewQueue(db_path=db_path)

    # Enqueue a test brief
    brief = RiskBrief(
        entity_id="7890",
        flagged_type="spike",
        model_score=0.91,
        top_factors=[ContributingFactor("g_card_cnt_24h", 10, "increases_risk")],
        confidence="high",
        estimated_fp_cost=450.0,
        recommended_action="hold_for_review",
        summary_text="UI test brief summary.",
    )
    qid = queue.enqueue(brief)

    import app as app_module
    app_module.DB_PATH = db_path

    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client, qid, db_path


def test_ui_index_and_pending_api(client):
    c, qid, db_path = client

    # Test index rendering
    res_index = c.get("/")
    assert res_index.status_code == 200
    assert b"Fraud-Spike Review Queue" in res_index.data

    # Test pending API
    res_api = c.get("/api/pending")
    assert res_api.status_code == 200
    data = res_api.get_json()
    assert len(data) == 1
    assert data[0]["id"] == qid
    assert data[0]["entity_id"] == "7890"


def test_ui_resolve_endpoint(client):
    c, qid, db_path = client

    # Resolve via API endpoint
    res = c.post(f"/api/resolve/{qid}", json={"action": "resolved_true_positive", "note": "Confirmed in UI"})
    assert res.status_code == 200
    assert res.get_json()["status"] == "success"

    # Confirm pending queue is now empty
    res_api = c.get("/api/pending")
    assert len(res_api.get_json()) == 0

    # Confirm audit log entry
    queue = ReviewQueue(db_path=db_path)
    logs = queue.get_audit_log(qid)
    assert len(logs) == 1
    assert logs[0]["reviewer_action"] == "resolved_true_positive"
    assert logs[0]["note"] == "Confirmed in UI"


def test_ui_no_blocking_payment_actions():
    """Security check: app.py contains no forbidden payment blocking strings."""
    content = Path("app.py").read_text().lower()
    forbidden_terms = ["block_card", "cancel_transaction", "hold_funds", "execute_payment"]
    for term in forbidden_terms:
        assert term not in content, f"Forbidden term '{term}' found in app.py"
