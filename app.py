#!/usr/bin/env python3
"""Review Queue Dashboard (UI Track / Section 15).

Reviewer console for human analysts to inspect flagged transaction & spike risk briefs,
view LLM-generated summaries and top contributing factors, and log resolution decisions.

Defense-only: Contains ZERO transaction execution, blocking, or payment modification endpoints.
Resolutions update SQLite review queue status and append to the audit log only.

Usage:
    python app.py [--port 5050] [--db results/review_queue.db]     # local dev server
    gunicorn app:app --bind 0.0.0.0:$PORT                          # production

Configuration is environment-first so the same module serves both: a hosting
platform injects PORT and the demo credentials, while the CLI flags stay the
convenient path locally. See `_env_*` helpers below.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import secrets

from flask import Flask, Response, jsonify, make_response, render_template, request

from src.explain.queue import ReviewQueue

app = Flask(__name__)

# Committed snapshot of real briefs, used when DEMO_MODE is on. The hosted
# instance cannot regenerate this: run_pipeline.py needs the ~650MB dataset and
# the trained artifacts, and free tiers wipe the filesystem on redeploy.
DEMO_DB_PATH = Path("data/demo_review_queue.db")
DEFAULT_DB_PATH = Path("results/review_queue.db")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_db_path() -> Path:
    """Pick the database this process serves.

    Explicit REVIEW_DB_PATH always wins; otherwise DEMO_MODE selects the
    committed snapshot, which is what the hosted instance runs on.
    """
    explicit = os.environ.get("REVIEW_DB_PATH")
    if explicit:
        return Path(explicit)
    return DEMO_DB_PATH if _env_flag("DEMO_MODE") else DEFAULT_DB_PATH


# Module-level so tests and `main()` can point the app at another database by
# assignment, which is the existing contract.
DB_PATH = _resolve_db_path()

# The three decisions a human reviewer can record. This mirrors ReviewQueue.resolve's
# own whitelist rather than trusting it: a decision that reaches the append-only audit
# log must have been made by a person, so the value is required and validated here at
# the edge, never defaulted. There is no fallback action, by design.
ALLOWED_REVIEWER_ACTIONS = frozenset(
    {"resolved_true_positive", "resolved_false_positive", "escalated"}
)

# Reviewer notes are free text written by a human and replayed in the audit view.
# Cap the length and reject C0/C7 control characters (newline and tab excepted, since
# a reviewer legitimately writes multi-line notes) so nothing can smuggle terminal
# escapes or NULs into a permanent record.
MAX_NOTE_LENGTH = 2000
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Double-submit CSRF: the token is issued as a SameSite=Strict cookie and echoed into
# the page, and a mutation must present it in a header that a cross-origin form post
# cannot set. No framework, no server-side session store.
CSRF_COOKIE_NAME = "fsq_csrf_token"
CSRF_HEADER_NAME = "X-CSRF-Token"


def _validate_note(raw: object) -> tuple[str | None, str | None]:
    """Return (note, error). Rejects non-strings, overlong text, and control bytes."""
    if raw is None:
        return "", None
    if not isinstance(raw, str):
        return None, "Field 'note' must be a string."
    if len(raw) > MAX_NOTE_LENGTH:
        return None, f"Field 'note' exceeds the {MAX_NOTE_LENGTH}-character limit."
    if _CONTROL_CHARACTERS.search(raw):
        return None, "Field 'note' contains disallowed control characters."
    return raw, None


def _csrf_token_is_valid() -> bool:
    """Header token must be present and equal to the cookie token."""
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME, "")
    header_token = request.headers.get(CSRF_HEADER_NAME, "")
    if not cookie_token or not header_token:
        return False
    return secrets.compare_digest(cookie_token, header_token)


# Read-only gating is deliberately SEPARATE from DEMO_MODE: the pitch demo needs
# a working resolve button, and the hosted filesystem is ephemeral anyway, so
# demo writes disappear on redeploy. Set READ_ONLY=1 to freeze the queue.
READ_ONLY = _env_flag("READ_ONLY")


def _auth_configured() -> tuple[str, str] | None:
    """Basic-auth credentials, or None when auth is disabled.

    Both variables must be set and non-empty. Absent them the app stays open,
    which keeps local development and the test suite unchanged — the hosted
    deployment is the place that sets them.
    """
    user = os.environ.get("DEMO_USER", "").strip()
    password = os.environ.get("DEMO_PASSWORD", "")
    if user and password:
        return user, password
    return None


@app.before_request
def _require_basic_auth() -> Response | None:
    """Gate every route except /health behind HTTP basic auth.

    Demo gating, not an identity system: one shared credential pair from the
    environment. /health is exempt because the platform's readiness probe is
    unauthenticated and a 401 there would fail the deploy.
    """
    if request.path == "/health":
        return None

    expected = _auth_configured()
    if expected is None:
        return None

    auth = request.authorization
    if auth and auth.username is not None and auth.password is not None:
        # compare_digest on both halves so neither is short-circuited.
        user_ok = secrets.compare_digest(auth.username, expected[0])
        pass_ok = secrets.compare_digest(auth.password, expected[1])
        if user_ok and pass_ok:
            return None

    return Response(
        "Authentication required.",
        401,
        {"WWW-Authenticate": 'Basic realm="Fraud-Spike Review Queue"'},
    )

@app.route("/")
def index():
    """Serve the dashboard and issue the CSRF token it will echo back.

    The same value goes into a SameSite=Strict cookie and into a meta tag, which
    is the double-submit pair /api/resolve checks.
    """
    token = request.cookies.get(CSRF_COOKIE_NAME) or secrets.token_urlsafe(32)
    response = make_response(render_template("index.html", csrf_token=token))
    response.set_cookie(
        CSRF_COOKIE_NAME,
        token,
        samesite="Strict",
        httponly=False,  # the page's own script must read it to echo it back
        secure=request.is_secure,
        path="/",
    )
    return response


@app.route("/api/pending")
def api_pending():
    queue = ReviewQueue(db_path=DB_PATH)
    items = queue.list_pending()
    return jsonify(items)


@app.route("/api/audit")
def api_audit():
    """Flattened audit trail: one entry per recorded decision, newest first.

    The audit log is a named requirement of the track and the strongest part of
    this architecture, so the console shows it as a peer view rather than hiding
    it behind a row. Item context is joined in here so the client renders a table
    without a second round trip per row.

    Reads only. The queue's public surface is fixed at four methods, so resolved
    rows come from list_pending's `status` argument rather than a new method.
    """
    queue = ReviewQueue(db_path=DB_PATH)
    entries: list[dict] = []
    for status in sorted(ALLOWED_REVIEWER_ACTIONS):
        for item in queue.list_pending(status=status):
            for record in queue.get_audit_log(item["id"]):
                entries.append({
                    "audit_id": record["id"],
                    "queue_id": item["id"],
                    "entity_id": item["entity_id"],
                    "flagged_type": item["flagged_type"],
                    "model_score": item["model_score"],
                    "reviewer_action": record["reviewer_action"],
                    "note": record["note"],
                    "timestamp": record["timestamp"],
                })

    # Newest first, with audit_id as the tiebreak: timestamp has one-second
    # granularity, so several decisions in the same second need a stable order.
    entries.sort(key=lambda e: (e["timestamp"], e["audit_id"]), reverse=True)
    return jsonify(entries)


@app.route("/api/resolve/<int:queue_id>", methods=["POST"])
def api_resolve(queue_id: int):
    if READ_ONLY:
        return jsonify({
            "status": "read_only",
            "error": "This instance is running read-only; resolutions are disabled.",
        }), 403

    if not _csrf_token_is_valid():
        return jsonify({
            "status": "error",
            "error": "Missing or invalid CSRF token.",
        }), 403

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({
            "status": "error",
            "error": "Request body must be a JSON object.",
        }), 400

    # No default. A resolution recorded without an explicit human choice would be a
    # decision nobody made, written to an append-only log — the one failure this
    # project cannot ship. Absent or unrecognised action is a 400, never a guess.
    action = data.get("action")
    if action not in ALLOWED_REVIEWER_ACTIONS:
        return jsonify({
            "status": "error",
            "error": "Field 'action' is required and must be one of: "
                     + ", ".join(sorted(ALLOWED_REVIEWER_ACTIONS)),
        }), 400

    note, note_error = _validate_note(data.get("note", ""))
    if note_error is not None:
        return jsonify({"status": "error", "error": note_error}), 400

    queue = ReviewQueue(db_path=DB_PATH)
    queue.resolve(queue_id, reviewer_action=action, note=note)
    return jsonify({"status": "success", "queue_id": queue_id, "action": action})


@app.route("/health")
def health():
    """Unauthenticated readiness probe for the hosting platform.

    Reports 200 as long as the process is up and the queue is reachable. The
    database check is what makes this a readiness signal rather than a liveness
    one: a boot with a missing or unreadable snapshot should not take traffic.
    """
    status = {
        "status": "ok",
        "demo_mode": _env_flag("DEMO_MODE"),
        "read_only": READ_ONLY,
        "auth_enabled": _auth_configured() is not None,
        "database": str(DB_PATH),
    }
    try:
        # Full count, not a limit=1 probe: this field is read as queue depth
        # (the README quotes it), so it has to be the number it claims to be.
        status["pending_items"] = len(ReviewQueue(db_path=DB_PATH).list_pending())
        status["database_reachable"] = True
    except Exception as exc:  # surface the reason rather than a bare 500
        status["status"] = "degraded"
        status["database_reachable"] = False
        status["error"] = str(exc)
        return jsonify(status), 503
    return jsonify(status)


def main() -> int:
    """Local development entry point only.

    Production runs `gunicorn app:app`, which imports the module and never
    reaches this function — hence the env-first configuration above. The CLI
    flags remain the ergonomic local path and take precedence when passed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("PORT", 5050)),
        help="Defaults to $PORT when set (hosting platforms inject it), else 5050.",
    )
    parser.add_argument(
        "--host", default=os.environ.get("HOST", "127.0.0.1"),
        help="Defaults to $HOST, else loopback. Use 0.0.0.0 to expose on the LAN.",
    )
    parser.add_argument(
        "--db", default=None,
        help="Queue database. Defaults to $REVIEW_DB_PATH, or the committed demo "
             "snapshot when DEMO_MODE is set, else results/review_queue.db.",
    )
    args = parser.parse_args()

    global DB_PATH
    if args.db:
        DB_PATH = Path(args.db)

    print(f"Starting Review Queue Dashboard on http://{args.host}:{args.port} ...")
    print(f"  database : {DB_PATH}")
    print(f"  auth     : {'enabled' if _auth_configured() else 'disabled (set DEMO_USER/DEMO_PASSWORD)'}")
    print(f"  read-only: {READ_ONLY}")
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
