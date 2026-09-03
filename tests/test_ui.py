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


# ---------------------------------------------------------------------------
# Deployment surface: health check, basic auth, read-only gating.
# ---------------------------------------------------------------------------


def test_health_endpoint_reports_ok(client):
    """Platform readiness probe: 200 plus a reachable-database signal."""
    c, _, db_path = client

    res = c.get("/health")
    assert res.status_code == 200
    body = res.get_json()
    assert body["status"] == "ok"
    assert body["database_reachable"] is True
    assert body["database"] == str(db_path)


def test_health_reports_503_when_database_unreadable(client, monkeypatch):
    """A boot with a broken snapshot must not pass readiness and take traffic."""
    c, _, _ = client
    import app as app_module

    def _boom(*args, **kwargs):
        raise RuntimeError("snapshot missing")

    monkeypatch.setattr(app_module, "ReviewQueue", _boom)
    res = c.get("/health")
    assert res.status_code == 503
    assert res.get_json()["status"] == "degraded"
    assert res.get_json()["database_reachable"] is False


def test_auth_disabled_when_credentials_absent(client, monkeypatch):
    """Local development and this suite must keep working with no login."""
    c, _, _ = client
    monkeypatch.delenv("DEMO_USER", raising=False)
    monkeypatch.delenv("DEMO_PASSWORD", raising=False)

    assert c.get("/").status_code == 200
    assert c.get("/api/pending").status_code == 200


def test_auth_required_on_all_routes_except_health(client, monkeypatch):
    c, qid, _ = client
    monkeypatch.setenv("DEMO_USER", "demo")
    monkeypatch.setenv("DEMO_PASSWORD", "s3cret")

    for path in ("/", "/api/pending"):
        res = c.get(path)
        assert res.status_code == 401, f"{path} should be gated"
        assert "Basic" in res.headers.get("WWW-Authenticate", "")

    res = c.post(f"/api/resolve/{qid}", json={"action": "escalated"})
    assert res.status_code == 401

    # The platform's probe is unauthenticated; a 401 here fails the deploy.
    assert c.get("/health").status_code == 200


def test_auth_accepts_correct_credentials_and_rejects_wrong_ones(client, monkeypatch):
    from base64 import b64encode

    c, _, _ = client
    monkeypatch.setenv("DEMO_USER", "demo")
    monkeypatch.setenv("DEMO_PASSWORD", "s3cret")

    def header(user: str, password: str) -> dict:
        token = b64encode(f"{user}:{password}".encode()).decode()
        return {"Authorization": f"Basic {token}"}

    assert c.get("/api/pending", headers=header("demo", "s3cret")).status_code == 200
    assert c.get("/api/pending", headers=header("demo", "wrong")).status_code == 401
    assert c.get("/api/pending", headers=header("wrong", "s3cret")).status_code == 401


def test_read_only_mode_blocks_resolution(client, monkeypatch):
    """READ_ONLY freezes the queue; it is independent of DEMO_MODE."""
    c, qid, db_path = client
    import app as app_module

    monkeypatch.setattr(app_module, "READ_ONLY", True)
    res = c.post(f"/api/resolve/{qid}", json={"action": "resolved_true_positive"})
    assert res.status_code == 403
    assert res.get_json()["status"] == "read_only"

    # Nothing was written: the item is still pending and has no audit entry.
    assert len(c.get("/api/pending").get_json()) == 1
    assert ReviewQueue(db_path=db_path).get_audit_log(qid) == []


def test_demo_mode_selects_the_committed_snapshot(monkeypatch):
    """DEMO_MODE is what points the hosted instance at the seeded database."""
    import app as app_module

    monkeypatch.delenv("REVIEW_DB_PATH", raising=False)

    monkeypatch.setenv("DEMO_MODE", "1")
    assert app_module._resolve_db_path() == app_module.DEMO_DB_PATH

    monkeypatch.delenv("DEMO_MODE", raising=False)
    assert app_module._resolve_db_path() == app_module.DEFAULT_DB_PATH

    # An explicit path always wins over the flag.
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setenv("REVIEW_DB_PATH", "/tmp/elsewhere.db")
    assert app_module._resolve_db_path() == Path("/tmp/elsewhere.db")


def test_committed_demo_snapshot_is_real_and_representative():
    """Guards the honesty contract on the shipped demo database.

    The snapshot must exist, carry provenance tracing to the real held-out
    split, and contain a genuine error surface — not only clean true positives.
    """
    db = Path("data/demo_review_queue.db")
    prov_path = Path("results/demo_seed_provenance.json")
    if not db.exists() or not prov_path.exists():
        pytest.skip("demo snapshot not generated; run scripts/seed_demo_db.py")

    prov = json.loads(prov_path.read_text())
    assert "IEEE-CIS" in prov["source_dataset"]
    assert prov["seeded_false_positives"] >= 1, "demo must show the real error surface"
    assert prov["seeded_true_positives"] >= 1

    items = ReviewQueue(db_path=db).list_pending()
    assert len(items) == prov["seeded_brief_count"]
    assert {i["flagged_type"] for i in items} == {"transaction", "spike"}
    # Borderline calls are the point of the stratified sample.
    assert any(i["confidence"] == "low" for i in items)

    # Ground truth must NOT leak into what a reviewer sees.
    for item in items:
        assert "isFraud" not in item["summary_text"]
        assert "ground_truth" not in json.dumps(item)
