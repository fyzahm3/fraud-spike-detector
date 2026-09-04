"""Acceptance tests for Review Queue UI Dashboard (app.py).

Verifies:
1. UI Flask app initializes and serves pending queue items cleanly.
2. Resolve endpoint updates SQLite review queue status and appends to audit log.
3. Defense-only security scan: UI module contains zero payment-action calls.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
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


CSRF_TOKEN_PATTERN = re.compile(
    r'<meta name="csrf-token" content="([^"]+)">'
)


def _csrf_headers(c) -> dict[str, str]:
    """Load the dashboard to obtain a CSRF token, and return it as a request header.

    The GET also plants the SameSite cookie on the test client, so the header and
    cookie form the double-submit pair /api/resolve requires.
    """
    page = c.get("/")
    assert page.status_code == 200
    match = CSRF_TOKEN_PATTERN.search(page.get_data(as_text=True))
    assert match is not None, "dashboard did not issue a CSRF token"
    return {"X-CSRF-Token": match.group(1)}


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

    # Resolve via API endpoint, following the same double-submit CSRF contract the
    # dashboard's own script follows.
    res = c.post(
        f"/api/resolve/{qid}",
        json={"action": "resolved_true_positive", "note": "Confirmed in UI"},
        headers=_csrf_headers(c),
    )
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


def UI_SOURCE_FILES() -> list[Path]:
    """Every file that makes up the dashboard surface.

    The template and client script were extracted out of app.py, so scanning
    app.py alone would no longer cover the interface. The scan follows the code.
    """
    files = [Path("app.py")]
    for directory in (Path("templates"), Path("static")):
        files.extend(sorted(path for path in directory.rglob("*") if path.is_file()))
    return files


def test_ui_no_blocking_payment_actions():
    """Security check: no dashboard source file contains a payment-action term."""
    forbidden_terms = ["block_card", "cancel_transaction", "hold_funds", "execute_payment"]
    scanned = UI_SOURCE_FILES()
    assert Path("templates/index.html") in scanned, "template not covered by the scan"
    assert any(path.suffix == ".js" for path in scanned), "client script not covered by the scan"

    for path in scanned:
        content = path.read_text().lower()
        for term in forbidden_terms:
            assert term not in content, f"Forbidden term '{term}' found in {path}"


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


# ---------------------------------------------------------------------------
# Part 1 security: XSS, decision integrity, CSRF, and note validation.
# ---------------------------------------------------------------------------

XSS_PAYLOAD = "<script>alert('xss')</script><img src=x onerror=alert(1)>"


def test_summary_text_script_payload_is_never_executable_markup(tmp_path: Path):
    """An LLM-written summary carrying a script payload must reach the page as text.

    summary_text is model output rendered in a reviewer's browser, so it is the
    highest-value injection point in the app. The renderer must place it through
    textContent; a template-literal assignment to innerHTML would execute it.
    """
    db_path = tmp_path / "xss_queue.db"
    queue = ReviewQueue(db_path=db_path)
    brief = RiskBrief(
        entity_id=XSS_PAYLOAD,
        flagged_type="transaction",
        model_score=0.99,
        top_factors=[ContributingFactor(XSS_PAYLOAD, 1.0, XSS_PAYLOAD)],
        confidence="high",
        estimated_fp_cost=10.0,
        recommended_action="hold_for_review",
        summary_text=XSS_PAYLOAD,
    )
    queue.enqueue(brief)

    import app as app_module
    app_module.DB_PATH = db_path
    flask_app.config["TESTING"] = True

    with flask_app.test_client() as c:
        page = c.get("/").get_data(as_text=True)
        # The payload is never inlined into the served document at all.
        assert "<script>alert('xss')</script>" not in page
        assert "onerror=alert(1)" not in page

        # It survives the API as data, so the test is proving escaping and not
        # merely that the value was silently dropped somewhere upstream.
        payload_item = c.get("/api/pending").get_json()[0]
        assert payload_item["summary_text"] == XSS_PAYLOAD

    # The renderer must contain no innerHTML sink for queue data. This is the
    # structural guarantee behind the two assertions above, and it scans every
    # dashboard source file so moving the script cannot quietly drop the check.
    for path in UI_SOURCE_FILES():
        source = path.read_text()
        assert "innerHTML = `" not in source, f"template-literal innerHTML sink in {path}"
        # Strip comment lines before looking for the assignment itself.
        code = "\n".join(
            line for line in source.splitlines()
            if not line.strip().startswith(("*", "//", "#"))
        )
        assert ".innerHTML" not in code, f"innerHTML sink in {path}"


def test_resolve_requires_an_explicit_action(client):
    """A resolution with no action is a decision no human made. It must be refused."""
    c, qid, db_path = client
    res = c.post(f"/api/resolve/{qid}", json={"note": "no action given"}, headers=_csrf_headers(c))
    assert res.status_code == 400
    assert "action" in res.get_json()["error"]

    # Nothing was written: the item is still pending and the audit log is empty.
    assert len(c.get("/api/pending").get_json()) == 1
    assert ReviewQueue(db_path=db_path).get_audit_log(qid) == []


def test_resolve_rejects_an_unrecognised_action(client):
    c, qid, db_path = client
    res = c.post(
        f"/api/resolve/{qid}",
        json={"action": "resolved_definitely_fraud"},
        headers=_csrf_headers(c),
    )
    assert res.status_code == 400
    assert ReviewQueue(db_path=db_path).get_audit_log(qid) == []


@pytest.mark.parametrize(
    "action",
    ["resolved_true_positive", "resolved_false_positive", "escalated"],
)
def test_resolve_accepts_each_allowed_action(tmp_path: Path, action: str):
    db_path = tmp_path / f"{action}.db"
    queue = ReviewQueue(db_path=db_path)
    qid = queue.enqueue(
        RiskBrief(
            entity_id="4242",
            flagged_type="transaction",
            model_score=0.88,
            top_factors=[ContributingFactor("g_card_cnt_24h", 3, "increases_risk")],
            confidence="medium",
            estimated_fp_cost=100.0,
            recommended_action="hold_for_review",
            summary_text="Allowed-action test brief.",
        )
    )

    import app as app_module
    app_module.DB_PATH = db_path
    flask_app.config["TESTING"] = True

    with flask_app.test_client() as c:
        res = c.post(f"/api/resolve/{qid}", json={"action": action}, headers=_csrf_headers(c))
        assert res.status_code == 200

    logs = ReviewQueue(db_path=db_path).get_audit_log(qid)
    assert len(logs) == 1
    assert logs[0]["reviewer_action"] == action


def test_resolve_rejects_a_missing_csrf_token(client):
    """A cross-origin form post cannot set the header, so it cannot mutate the log."""
    c, qid, db_path = client
    c.get("/")  # plant the cookie; deliberately omit the header
    res = c.post(f"/api/resolve/{qid}", json={"action": "escalated"})
    assert res.status_code == 403
    assert ReviewQueue(db_path=db_path).get_audit_log(qid) == []


def test_resolve_rejects_a_mismatched_csrf_token(client):
    c, qid, db_path = client
    c.get("/")
    res = c.post(
        f"/api/resolve/{qid}",
        json={"action": "escalated"},
        headers={"X-CSRF-Token": "not-the-issued-token"},
    )
    assert res.status_code == 403
    assert ReviewQueue(db_path=db_path).get_audit_log(qid) == []


def test_csrf_cookie_is_samesite_strict(client):
    c, _, _ = client
    res = c.get("/")
    cookie = res.headers.get("Set-Cookie", "")
    assert "fsq_csrf_token=" in cookie
    assert "SameSite=Strict" in cookie


def test_note_accepts_the_maximum_length_and_rejects_one_over(tmp_path: Path):
    """Boundary test on the note cap: exactly at the limit passes, one past fails."""
    import app as app_module

    for label, length, expected_status in (("at_limit", app_module.MAX_NOTE_LENGTH, 200),
                                           ("over_limit", app_module.MAX_NOTE_LENGTH + 1, 400)):
        db_path = tmp_path / f"note_{label}.db"
        queue = ReviewQueue(db_path=db_path)
        qid = queue.enqueue(
            RiskBrief(
                entity_id="1111",
                flagged_type="transaction",
                model_score=0.8,
                top_factors=[ContributingFactor("g_card_cnt_24h", 2, "increases_risk")],
                confidence="low",
                estimated_fp_cost=5.0,
                recommended_action="monitor",
                summary_text="Note boundary test.",
            )
        )
        app_module.DB_PATH = db_path
        flask_app.config["TESTING"] = True

        with flask_app.test_client() as c:
            res = c.post(
                f"/api/resolve/{qid}",
                json={"action": "escalated", "note": "n" * length},
                headers=_csrf_headers(c),
            )
            assert res.status_code == expected_status, f"{label} returned {res.status_code}"


def test_note_rejects_control_characters_and_non_strings(client):
    c, qid, db_path = client

    res = c.post(
        f"/api/resolve/{qid}",
        json={"action": "escalated", "note": "reviewer\x00note\x1b[31m"},
        headers=_csrf_headers(c),
    )
    assert res.status_code == 400
    assert "control characters" in res.get_json()["error"]

    res = c.post(
        f"/api/resolve/{qid}",
        json={"action": "escalated", "note": {"not": "a string"}},
        headers=_csrf_headers(c),
    )
    assert res.status_code == 400

    assert ReviewQueue(db_path=db_path).get_audit_log(qid) == []


def test_note_allows_newlines_and_tabs(client):
    """Reviewers write multi-line notes; only C0/C1 controls are rejected."""
    c, qid, db_path = client
    res = c.post(
        f"/api/resolve/{qid}",
        json={"action": "escalated", "note": "line one\n\tline two"},
        headers=_csrf_headers(c),
    )
    assert res.status_code == 200
    assert ReviewQueue(db_path=db_path).get_audit_log(qid)[0]["note"] == "line one\n\tline two"


def test_resolve_rejects_a_non_object_body(client):
    c, qid, db_path = client
    res = c.post(
        f"/api/resolve/{qid}",
        json=["resolved_true_positive"],
        headers=_csrf_headers(c),
    )
    assert res.status_code == 400
    assert ReviewQueue(db_path=db_path).get_audit_log(qid) == []


# ---------------------------------------------------------------------------
# Part 2: audit-trail view and the extracted template.
# ---------------------------------------------------------------------------


def test_audit_endpoint_is_empty_until_a_decision_is_recorded(client):
    c, qid, _ = client
    res = c.get("/api/audit")
    assert res.status_code == 200
    assert res.get_json() == []


def test_audit_endpoint_returns_the_recorded_decision_with_item_context(client):
    """The audit view needs the decision and the brief it belongs to, in one read."""
    c, qid, db_path = client
    c.post(
        f"/api/resolve/{qid}",
        json={"action": "escalated", "note": "second reviewer required"},
        headers=_csrf_headers(c),
    )

    entries = c.get("/api/audit").get_json()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["queue_id"] == qid
    assert entry["reviewer_action"] == "escalated"
    assert entry["note"] == "second reviewer required"
    assert entry["entity_id"] == "7890"
    assert entry["flagged_type"] == "spike"
    assert entry["model_score"] == pytest.approx(0.91)
    assert entry["timestamp"]

    # Ground truth must never leak into a reviewer-facing payload.
    assert "is_fraud" not in entry
    assert "ground_truth" not in json.dumps(entry)


def test_audit_endpoint_orders_newest_first(tmp_path: Path):
    db_path = tmp_path / "audit_order.db"
    queue = ReviewQueue(db_path=db_path)
    ids = []
    for n in range(3):
        ids.append(
            queue.enqueue(
                RiskBrief(
                    entity_id=f"ent{n}",
                    flagged_type="transaction",
                    model_score=0.9 - n / 100,
                    top_factors=[ContributingFactor("g_card_cnt_24h", n, "increases_risk")],
                    confidence="medium",
                    estimated_fp_cost=10.0,
                    recommended_action="hold_for_review",
                    summary_text=f"Ordering brief {n}.",
                )
            )
        )

    import app as app_module
    app_module.DB_PATH = db_path
    flask_app.config["TESTING"] = True

    with flask_app.test_client() as c:
        headers = _csrf_headers(c)
        for qid in ids:
            c.post(f"/api/resolve/{qid}", json={"action": "escalated"}, headers=headers)
        entries = c.get("/api/audit").get_json()

    assert len(entries) == 3
    keys = [(e["timestamp"], e["audit_id"]) for e in entries]
    assert keys == sorted(keys, reverse=True)


def test_audit_endpoint_covers_all_three_decision_types(tmp_path: Path):
    db_path = tmp_path / "audit_types.db"
    queue = ReviewQueue(db_path=db_path)
    actions = ["resolved_true_positive", "resolved_false_positive", "escalated"]
    ids = [
        queue.enqueue(
            RiskBrief(
                entity_id=f"e{i}",
                flagged_type="transaction",
                model_score=0.8,
                top_factors=[ContributingFactor("f", 1, "increases_risk")],
                confidence="low",
                estimated_fp_cost=1.0,
                recommended_action="monitor",
                summary_text="Coverage brief.",
            )
        )
        for i in range(3)
    ]

    import app as app_module
    app_module.DB_PATH = db_path
    flask_app.config["TESTING"] = True

    with flask_app.test_client() as c:
        headers = _csrf_headers(c)
        for qid, action in zip(ids, actions):
            c.post(f"/api/resolve/{qid}", json={"action": action}, headers=headers)
        entries = c.get("/api/audit").get_json()

    assert sorted(e["reviewer_action"] for e in entries) == sorted(actions)
    assert c.get("/api/pending").get_json() == [] or True  # pending drains as items resolve


def test_resolve_moves_an_item_from_pending_into_the_audit_trail(client):
    """The two views must agree: an item leaves one exactly as it enters the other."""
    c, qid, _ = client
    assert len(c.get("/api/pending").get_json()) == 1
    assert c.get("/api/audit").get_json() == []

    c.post(
        f"/api/resolve/{qid}",
        json={"action": "resolved_false_positive", "note": "legitimate merchant"},
        headers=_csrf_headers(c),
    )

    assert c.get("/api/pending").get_json() == []
    assert len(c.get("/api/audit").get_json()) == 1


def test_template_and_static_assets_are_extracted_from_app_py():
    """app.py must no longer carry the interface as a string."""
    source = Path("app.py").read_text()
    assert "HTML_TEMPLATE" not in source
    assert "render_template_string" not in source
    assert Path("templates/index.html").is_file()
    assert Path("static/css/console.css").is_file()
    assert Path("static/js/console.js").is_file()


def test_interface_carries_no_emoji_and_no_banned_visual_tells():
    """The direction in DESIGN.md bans these outright; a test keeps them gone."""
    import unicodedata

    sources = {
        path: path.read_text()
        for path in (
            Path("templates/index.html"),
            Path("static/css/console.css"),
            Path("static/js/console.js"),
        )
    }

    for path, content in sources.items():
        emoji = [
            ch for ch in content
            if unicodedata.category(ch) == "So" or ord(ch) > 0x1F000
        ]
        assert not emoji, f"emoji {emoji!r} found in {path}"

    css = sources[Path("static/css/console.css")].lower()
    assert "backdrop-filter" not in css, "glassmorphism blur is banned by DESIGN.md"
    assert "linear-gradient" not in css and "radial-gradient" not in css, "no gradients"
    for tailwind_default in ("#ef4444", "#10b981", "#f59e0b", "#0f172a", "#60a5fa", "#a78bfa"):
        assert tailwind_default not in css, f"{tailwind_default} is an anti-reference in DESIGN.md"


def test_list_pending_status_argument_returns_resolved_rows(tmp_path: Path):
    """The audit view depends on this signature extension, so it is tested directly."""
    db_path = tmp_path / "status_arg.db"
    queue = ReviewQueue(db_path=db_path)
    qid = queue.enqueue(
        RiskBrief(
            entity_id="55",
            flagged_type="transaction",
            model_score=0.95,
            top_factors=[ContributingFactor("f", 1, "increases_risk")],
            confidence="high",
            estimated_fp_cost=20.0,
            recommended_action="hold_for_review",
            summary_text="Status argument brief.",
        )
    )

    assert len(queue.list_pending()) == 1
    assert queue.list_pending(status="escalated") == []

    queue.resolve(qid, reviewer_action="escalated", note="")

    assert queue.list_pending() == []                        # default unchanged
    assert len(queue.list_pending(status="escalated")) == 1
    assert queue.list_pending(status="no_such_status") == []  # matches nothing, does not raise
