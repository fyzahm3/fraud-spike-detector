#!/usr/bin/env python3
"""Review Queue Dashboard (UI Track / Section 15).

Minimal, modern web interface for human reviewers to inspect flagged transaction & spike risk briefs,
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
import secrets

from flask import Flask, Response, jsonify, render_template_string, request

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

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fraud-Spike Review Queue | Human-in-the-Loop Escalation</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --border-color: rgba(255, 255, 255, 0.1);
            --text-primary: #f8fafc;
            --text-muted: #94a3b8;
            --accent-red: #ef4444;
            --accent-amber: #f59e0b;
            --accent-green: #10b981;
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-primary);
            margin: 0;
            padding: 24px;
            line-height: 1.5;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 28px;
        }

        .title-group h1 {
            margin: 0;
            font-size: 1.75rem;
            font-weight: 700;
            background: linear-gradient(135deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .title-group p {
            margin: 4px 0 0 0;
            color: var(--text-muted);
            font-size: 0.875rem;
        }

        .badge-defense {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(52, 211, 153, 0.3);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }

        .stat-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }

        .stat-card .label {
            font-size: 0.75rem;
            text-transform: uppercase;
            color: var(--text-muted);
            font-weight: 600;
        }

        .stat-card .value {
            font-size: 1.875rem;
            font-weight: 700;
            margin-top: 6px;
        }

        .queue-section h2 {
            font-size: 1.25rem;
            margin-bottom: 16px;
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 20px;
            backdrop-filter: blur(12px);
            transition: transform 0.2s, border-color 0.2s;
        }

        .card:hover {
            border-color: rgba(96, 165, 250, 0.4);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }

        .entity-info {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .entity-id {
            font-size: 1.1rem;
            font-weight: 700;
        }

        .type-tag {
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .type-tag.spike {
            background: rgba(239, 68, 68, 0.2);
            color: #fca5a5;
            border: 1px solid rgba(239, 68, 68, 0.4);
        }

        .type-tag.transaction {
            background: rgba(59, 130, 246, 0.2);
            color: #93c5fd;
            border: 1px solid rgba(59, 130, 246, 0.4);
        }

        .score-box {
            text-align: right;
        }

        .score-value {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--accent-red);
        }

        .score-label {
            font-size: 0.7rem;
            color: var(--text-muted);
        }

        .brief-box {
            background: rgba(15, 23, 42, 0.6);
            border-left: 4px solid var(--accent-purple);
            padding: 14px 16px;
            border-radius: 0 8px 8px 0;
            margin-bottom: 20px;
        }

        .brief-box p {
            margin: 0;
            font-size: 0.925rem;
            color: #e2e8f0;
        }

        .factors-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
            font-size: 0.85rem;
        }

        .factors-table th {
            text-align: left;
            color: var(--text-muted);
            padding: 8px 12px;
            border-bottom: 1px solid var(--border-color);
            font-weight: 600;
        }

        .factors-table td {
            padding: 8px 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }

        .action-bar {
            display: flex;
            gap: 12px;
            align-items: center;
            justify-content: flex-end;
            padding-top: 12px;
            border-top: 1px solid var(--border-color);
        }

        .btn {
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            border: none;
            transition: opacity 0.2s, transform 0.1s;
        }

        .btn:hover {
            opacity: 0.9;
        }

        .btn:active {
            transform: scale(0.97);
        }

        .btn-tp {
            background: var(--accent-red);
            color: white;
        }

        .btn-fp {
            background: rgba(255, 255, 255, 0.1);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
        }

        .btn-escalate {
            background: var(--accent-purple);
            color: white;
        }

        .empty-state {
            text-align: center;
            padding: 48px;
            color: var(--text-muted);
            background: var(--card-bg);
            border-radius: 12px;
            border: 1px dashed var(--border-color);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="title-group">
                <h1>Fraud-Spike Review Queue</h1>
                <p>Human-Gated Escalation Dashboard • Defense-Only Audit Trail</p>
            </div>
            <div class="badge-defense">🛡️ Defense-Only System</div>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="label">Pending Queue Items</div>
                <div class="value" id="stat-pending">0</div>
            </div>
            <div class="stat-card">
                <div class="label">Total Risk Score Avg</div>
                <div class="value" id="stat-avg-score">0.00</div>
            </div>
            <div class="stat-card">
                <div class="label">Estimated FP Cost at Risk</div>
                <div class="value" id="stat-fp-cost">$0</div>
            </div>
        </div>

        <div class="queue-section">
            <h2>Pending Items requiring Review</h2>
            <div id="queue-container">Loading review queue...</div>
        </div>
    </div>

    <script>
        async function fetchQueue() {
            const res = await fetch('/api/pending');
            const data = await res.json();
            renderQueue(data);
        }

        function renderQueue(items) {
            document.getElementById('stat-pending').innerText = items.length;

            if (items.length === 0) {
                document.getElementById('stat-avg-score').innerText = '0.00';
                document.getElementById('stat-fp-cost').innerText = '$0';
                document.getElementById('queue-container').innerHTML = `
                    <div class="empty-state">
                        <h3>No Pending Reviews</h3>
                        <p>All flagged transactions and spike events have been reviewed.</p>
                    </div>
                `;
                return;
            }

            const avgScore = (items.reduce((sum, item) => sum + item.model_score, 0) / items.length).toFixed(3);
            const totalCost = items.reduce((sum, item) => sum + item.estimated_fp_cost, 0).toLocaleString('en-US', {style: 'currency', currency: 'USD'});

            document.getElementById('stat-avg-score').innerText = avgScore;
            document.getElementById('stat-fp-cost').innerText = totalCost;

            const html = items.map(item => `
                <div class="card" id="item-${item.id}">
                    <div class="card-header">
                        <div class="entity-info">
                            <span class="entity-id">Entity ${item.entity_id}</span>
                            <span class="type-tag ${item.flagged_type}">${item.flagged_type}</span>
                            <span style="font-size:0.8rem; color:var(--text-muted);">Confidence: <b>${item.confidence}</b></span>
                        </div>
                        <div class="score-box">
                            <div class="score-value">${item.model_score.toFixed(4)}</div>
                            <div class="score-label">Risk Score</div>
                        </div>
                    </div>

                    <div class="brief-box">
                        <p><strong>LLM Risk Brief:</strong> ${item.summary_text}</p>
                    </div>

                    <table class="factors-table">
                        <thead>
                            <tr>
                                <th>Top Contributing Feature</th>
                                <th>Value</th>
                                <th>Direction</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${item.top_factors.map(f => `
                                <tr>
                                    <td><code>${f.feature}</code></td>
                                    <td>${typeof f.value === 'number' ? f.value.toFixed(2) : f.value}</td>
                                    <td style="color:${f.direction === 'increases_risk' ? '#f87171' : '#6ee7b7'}">${f.direction}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>

                    <div class="action-bar">
                        <span style="font-size:0.8rem; color:var(--text-muted); margin-right:auto;">
                            Rec. Action: <strong>${item.recommended_action}</strong> | FP Cost Est: $${item.estimated_fp_cost.toLocaleString()}
                        </span>
                        <button class="btn btn-fp" onclick="resolveItem(${item.id}, 'resolved_false_positive')">Dismiss False Positive</button>
                        <button class="btn btn-escalate" onclick="resolveItem(${item.id}, 'escalated')">Escalate for Review</button>
                        <button class="btn btn-tp" onclick="resolveItem(${item.id}, 'resolved_true_positive')">Confirm True Positive</button>
                    </div>
                </div>
            `).join('');

            document.getElementById('queue-container').innerHTML = html;
        }

        async function resolveItem(queueId, action) {
            const note = prompt(`Optional reviewer note for ${action}:`) || "";
            const res = await fetch(`/api/resolve/${queueId}`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: action, note: note})
            });
            if (res.ok) {
                fetchQueue();
            } else {
                alert('Failed to resolve item.');
            }
        }

        fetchQueue();
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/pending")
def api_pending():
    queue = ReviewQueue(db_path=DB_PATH)
    items = queue.list_pending()
    return jsonify(items)


@app.route("/api/resolve/<int:queue_id>", methods=["POST"])
def api_resolve(queue_id: int):
    if READ_ONLY:
        return jsonify({
            "status": "read_only",
            "error": "This instance is running read-only; resolutions are disabled.",
        }), 403

    data = request.get_json() or {}
    action = data.get("action", "resolved_true_positive")
    note = data.get("note", "")

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
        status["pending_items"] = len(ReviewQueue(db_path=DB_PATH).list_pending(limit=1))
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
