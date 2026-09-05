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
import json
import os
from pathlib import Path
import re
import secrets

from flask import Flask, Response, jsonify, make_response, render_template, request

from src.explain.queue import ReviewQueue
from src.serve.scorer import (
    ScoringUnavailableError,
    is_available as scoring_is_available,
    list_samples,
    score as score_sample,
)
from src.ingest.razorpay_live import (
    LIVE_DEMO_FLAGGED_TYPE,
    SIGNATURE_HEADER,
    UNSCORED_MODEL_SCORE,
    LiveIngestionError,
    RazorpayConfigError,
    build_live_demo_brief,
    create_test_order,
    extract_event_id,
    load_credentials,
    record_event_id,
    verify_webhook_signature,
)

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

# Razorpay signs its webhook deliveries; it cannot present basic-auth
# credentials. This path is therefore exempt from the shared demo password and
# authenticates by HMAC signature instead, which is the stronger of the two.
WEBHOOK_PATH = "/api/webhook/razorpay"

# ---------------------------------------------------------------------------
# Contextual help.
#
# Written prose, held in one place and rendered into the page by Jinja. There is
# no LLM call behind the "?" affordances and there is deliberately no way to add
# one: an explanation of this system's own methodology is exactly the text that
# must not be improvised, because a plausible-sounding wrong answer about the
# evaluation protocol is worse than no answer at all. Deterministic, fast, and
# incapable of hallucinating.
#
# Each entry is (title, body). Bodies are escaped by Jinja on render and reach
# the page as text nodes, never as markup.
# ---------------------------------------------------------------------------

HELP_TOPICS: dict[str, dict[str, str]] = {
    "auc_pr": {
        "title": "Why AUC-PR, not accuracy",
        "body": (
            "Fraud is rare: 3,083 of the 88,581 transactions in the held-out test "
            "split are fraudulent, about 3.5%. A model that calls every single "
            "transaction legitimate is therefore 96.5% accurate while catching "
            "nothing, so accuracy cannot distinguish a useful model from a useless "
            "one here. AUC-PR summarises the precision/recall trade-off across "
            "every possible threshold and uses the rare class as its reference "
            "point, so it moves when the model actually gets better at the job. "
            "AUC-ROC is reported alongside it, but on data this imbalanced ROC "
            "flatters weak models, because the large legitimate class dominates "
            "its false-positive axis."
        ),
    },
    "cost_ratio": {
        "title": "What the value ratio means",
        "body": (
            "Currency caught for every unit of legitimate currency disrupted, "
            "measured on the held-out test split. Precision and recall count "
            "transactions and treat a 5 payment and a 5,000 payment as equal "
            "events; this metric weighs them by the amount actually at stake, "
            "which is closer to what the outcome is worth. Above 1.0 the flagged "
            "set carries more fraudulent value than legitimate value. The "
            "baseline sits below 1.0 and the graph variant above it, which is the "
            "single clearest statement of what the graph features changed."
        ),
    },
    "chronological_split": {
        "title": "Why the split is chronological",
        "body": (
            "The data is cut by time: the earliest 70% of transactions train the "
            "model, the next 15% tune it, and the final 15% test it. A random "
            "shuffle would let the model learn from transactions that happened "
            "after the ones it is scored on, which no deployed system can do, and "
            "the resulting metric would be optimistic in a way that never survives "
            "production. Each boundary is snapped past its tied-timestamp block so "
            "no single instant straddles two splits, and every split is SHA-256 "
            "checksummed into results/split_manifest.json — the evaluation scripts "
            "refuse to run if a checksum has drifted."
        ),
    },
    "leakage": {
        "title": "How leakage is prevented structurally",
        "body": (
            "Not by discipline, by construction. No function in the training code "
            "accepts the test frame at all, so a threshold or a hyperparameter "
            "cannot be chosen using it — both come from the validation split only. "
            "The streaming features carry the same rule down to the row: every "
            "feature for a given transaction is computed from strictly earlier "
            "transactions, on half-open windows, with ties broken deterministically. "
            "Tests verify this against brute-force references and against "
            "deliberately future-corrupted data."
        ),
    },
    "graph_features": {
        "title": "The causal graph features",
        "body": (
            "Twelve features describing how a card, email, device, or address has "
            "co-occurred with others over the previous 24 hours and 7 days, plus a "
            "48-hour exponentially-decayed measure of fraud among an entity's "
            "neighbours. They are built by a single stateful pass that sees train, "
            "then validation, then test in order, so each row's features reflect "
            "only its own past. A first-time entity gets zeros rather than a "
            "guess. This is the difference between the two variants compared on "
            "the evidence page."
        ),
    },
    "threshold": {
        "title": "Where the decision threshold comes from",
        "body": (
            "Chosen on the validation split and then frozen before the test split "
            "is scored even once. Tuning it against test results would be choosing "
            "the answer after seeing the mark scheme, and the reported figures "
            "would no longer describe unseen data. The committed value is in "
            "results/pipeline_run_summary.json and in the model metadata."
        ),
    },
    "llm_prose": {
        "title": "Why the LLM only writes prose",
        "body": (
            "Everything a reader might act on is computed in Python before the "
            "language model is called: the risk score, the confidence band, the "
            "estimated cost of a false positive, and the recommendation, which is "
            "one of a fixed set of values. The model receives those and writes the "
            "summary paragraph — nothing else. It cannot change a score, pick a "
            "recommendation, or introduce a number, so a hallucination costs you a "
            "clumsy sentence rather than a wrong decision. When no API key is "
            "configured the pipeline substitutes a deterministic template instead "
            "of quietly degrading."
        ),
    },
    "live_unscored": {
        "title": "Why the live transaction carries no score",
        "body": (
            "The model was trained on the IEEE-CIS feature space: roughly 430 "
            "engineered columns, including a large block of masked proprietary "
            "features and the graph features derived from an entity's own history. "
            "A payment webhook carries an amount, a currency, a method, an order "
            "id and a timestamp. Those do not overlap. Padding the ~430 missing "
            "features with zeros would return a number, but it would describe the "
            "padding rather than the payment. This project already found and "
            "deleted one metric that was an artifact of its data rather than its "
            "model; it will not manufacture a second one to make a demo look "
            "complete. So the item is stored and displayed as explicitly unscored."
        ),
    },
    "human_decides": {
        "title": "What the system does and does not do",
        "body": (
            "It scores, groups, explains, and queues for review. It has no code "
            "path that can act on a payment in any direction, and automated tests "
            "scan the source for one on every run. A recorded decision updates a "
            "status field and appends a row to an audit log that is never edited "
            "or deleted; a correction is a new row. The reviewer holds the "
            "decision, and the record shows who made it and when."
        ),
    },
    "spike": {
        "title": "Transactions versus spikes",
        "body": (
            "A transaction item is one payment the model scored highly on its own. "
            "A spike item is a burst: several risky transactions on the same card, "
            "email, device, or address inside a rolling window, compared against "
            "that entity's own earlier baseline rather than a global average. "
            "Coordinated abuse tends to appear as the second shape, which a "
            "per-transaction score alone reads as unrelated events."
        ),
    },
    "live_scoring": {
        "body": (
            "The trained XGBoost model is loaded in this web process and asked for a "
            "prediction when you press the button, so the number you see was computed "
            "then, not looked up. What is committed alongside the model is the input: "
            "the feature vector the pipeline produced for a real held-out transaction. "
            "That is a deployment constraint rather than a shortcut \u2014 the ~650MB "
            "dataset and the sequential feature build that turns a raw transaction into "
            "443 model features cannot run on a free instance. Ground truth is shown "
            "here because this is an evaluation surface; the reviewer's queue still "
            "contains no labels, so a reviewer is never shown the answer."
        ),
        "title": "What \u201crun the model\u201d means here",
    },
    "dataset_gap": {
        "title": "The India-market gap, stated plainly",
        "body": (
            "IEEE-CIS is real US card-not-present data. India's rails are "
            "dominated by UPI, which differs in entity topology (virtual payment "
            "addresses and device identifiers rather than card numbers and billing "
            "addresses), in settlement speed (instant and around the clock, so "
            "bursts unfold in seconds rather than hours), and in typology. No "
            "synthetic UPI data was generated to paper over this, because "
            "fabricated data would produce fabricated metrics. The gap is stated "
            "rather than filled."
        ),
    },
}

# Reading this site is public; changing its state is not.
#
# The gate is drawn on the HTTP method rather than on a list of public paths,
# and that direction matters. A path allowlist fails open: add a route, forget
# to list it, and it is unprotected. This fails closed — a new route is public
# only for as long as it is read-only, and the moment it accepts a POST it is
# behind the password automatically. /health needs no special case any more; it
# is a GET like every other public surface.
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


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

    Both variables must be set and non-empty. Absent them mutations are open
    too, which keeps local development and the test suite unchanged — the
    hosted deployment is the place that sets them.
    """
    user = os.environ.get("DEMO_USER", "").strip()
    password = os.environ.get("DEMO_PASSWORD", "")
    if user and password:
        return user, password
    return None


@app.before_request
def _require_basic_auth() -> Response | None:
    """Gate state-changing requests behind HTTP basic auth. Reading is public.

    Every read surface — the landing page, the evidence page, the review queue,
    the live-ingestion page, their JSON APIs, and the readiness probe — answers
    without credentials. A reviewer opening the link is looking at the product
    in the same second, which is the entire point of publishing it.

    Authentication applies to the two routes that change something: recording a
    reviewer decision, and creating a test-mode order. Demo gating, not an
    identity system — one shared credential pair from the environment.

    The Razorpay webhook is exempt because Razorpay presents an HMAC signature
    rather than a password. That is not a hole: the signature is verified on
    every request to it, which is the stronger of the two checks.
    """
    if request.method in SAFE_METHODS:
        return None

    if request.path == WEBHOOK_PATH:
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

_GATEWAY_CONTEXT: dict = {}


def gateway_context() -> dict:
    """Measured facts about the gateway model, cached per process.

    Served alongside every gateway score so the number never appears without
    the two things that qualify it: how many features it had, and how it
    actually performs against the full model on the same held-out data.
    """
    if _GATEWAY_CONTEXT:
        return _GATEWAY_CONTEXT
    context: dict = {}
    try:
        with open(RESULTS_DIR / "gateway_metrics.json", encoding="utf-8") as handle:
            gateway = json.load(handle)
        context["threshold"] = gateway.get("threshold")
        context["auc_pr"] = (gateway.get("metrics") or {}).get("auc_pr")
        context["n_features"] = gateway.get("n_features")
    except (OSError, ValueError):
        pass
    try:
        with open(RESULTS_DIR / "phase_comparison.json", encoding="utf-8") as handle:
            context["full_model_auc_pr"] = json.load(handle)["graph"]["auc_pr"]
    except (OSError, ValueError, KeyError):
        pass
    context.setdefault("full_model_n_features", 443)
    _GATEWAY_CONTEXT.update(context)
    return _GATEWAY_CONTEXT


def _serialize_item(item: dict) -> dict:
    """Shape a queue row for the API, with unscored items marked as such.

    Two derived fields, both about the same thing:

    - `scored` is False for a live-ingested item. The client keys its whole
      presentation off this flag, so an unscored item cannot pick up a scored
      item's rendering by accident.
    - `model_score` becomes JSON null for those items rather than the storage
      sentinel. review_queue.model_score is REAL NOT NULL, so the row has to
      hold *some* number; what a reviewer must never see is a number that looks
      like a risk score when no risk score exists. Null is the honest wire
      value, and it is also unusable as one — arithmetic on it fails loudly
      instead of silently averaging a fake score into a metric.
    """
    out = dict(item)
    scored = out.get("flagged_type") != LIVE_DEMO_FLAGGED_TYPE
    out["scored"] = scored

    gateway_score = out.pop("gateway_score", None)
    if not scored:
        out["model_score"] = None
        # The gateway model's score travels in its own field with its own
        # measured performance attached, so the client cannot render it in the
        # place where a full-model score would go, and cannot present it without
        # the context that makes it honest.
        if gateway_score is not None:
            out["gateway"] = {
                "score": float(gateway_score),
                "threshold": gateway_context().get("threshold"),
                "auc_pr": gateway_context().get("auc_pr"),
                "full_model_auc_pr": gateway_context().get("full_model_auc_pr"),
                "n_features": gateway_context().get("n_features"),
                "full_model_n_features": gateway_context().get("full_model_n_features"),
            }
    return out


# ---------------------------------------------------------------------------
# Committed evidence, read from results/ at request time.
#
# Every number the site displays comes from one of these files. Nothing is
# computed here and nothing is hard-coded in a template: if a figure is not in
# a committed artifact, it does not appear on the site. That is the same
# no-fabricated-results rule the README states, enforced by having no other
# source of numbers available to the page.
#
# Reading is stdlib json on three small files, so the dashboard's import chain
# stays free of pandas and requirements-web.txt stays Flask + gunicorn.
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results")

EVIDENCE_FILES = {
    "phase_comparison": "phase_comparison.json",
    "split_manifest": "split_manifest.json",
    "pipeline_run": "pipeline_run_summary.json",
    "gateway": "gateway_metrics.json",
}


def load_evidence() -> dict:
    """Read the committed result artifacts. A missing file yields None, not a 500.

    The evidence page degrades to "not available in this deployment" for any
    artifact it cannot read. An absent file is a deployment fact worth showing
    plainly; it is never a reason to invent a number to fill the gap, and never
    a reason to take the whole page down.
    """
    evidence: dict[str, object] = {}
    for key, filename in EVIDENCE_FILES.items():
        try:
            with open(RESULTS_DIR / filename, encoding="utf-8") as handle:
                evidence[key] = json.load(handle)
        except (OSError, ValueError):
            evidence[key] = None
    return evidence


def _render_page(template: str, **context):
    """Render a page and plant the CSRF token it may need to echo back.

    The same value goes into a SameSite=Strict cookie and into a meta tag; that
    double-submit pair is what the mutation routes check. Every page carries it
    rather than only the ones with buttons, so a control can move between
    surfaces without silently losing its token.
    """
    token = request.cookies.get(CSRF_COOKIE_NAME) or secrets.token_urlsafe(32)
    response = make_response(render_template(
        template,
        csrf_token=token,
        help_topics=HELP_TOPICS,
        **context,
    ))
    response.set_cookie(
        CSRF_COOKIE_NAME,
        token,
        samesite="Strict",
        httponly=False,  # the page's own script must read it to echo it back
        secure=request.is_secure,
        path="/",
    )
    return response


@app.route("/")
def index():
    """Public landing page: what this is, and the evidence behind the claim."""
    return _render_page("landing.html", evidence=load_evidence(), active="home")


@app.route("/metrics")
def metrics_page():
    """Public evidence page. Every figure traces to a file under results/."""
    return _render_page(
        "metrics.html",
        evidence=load_evidence(),
        active="metrics",
        scoring_available=scoring_is_available(),
    )


@app.route("/demo")
def demo_page():
    """Public, read-only view of the review queue. Deciding still needs auth."""
    return _render_page("demo.html", active="demo")


@app.route("/live")
def live_page():
    """Public explanation of live ingestion, plus the test-mode trigger."""
    return _render_page(
        "live.html",
        active="live",
        evidence=load_evidence(),
        gateway=gateway_context(),
    )


@app.route("/api/pending")
def api_pending():
    queue = ReviewQueue(db_path=DB_PATH)
    items = [_serialize_item(item) for item in queue.list_pending()]
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
            shaped = _serialize_item(item)
            for record in queue.get_audit_log(item["id"]):
                entries.append({
                    "audit_id": record["id"],
                    "queue_id": item["id"],
                    "entity_id": item["entity_id"],
                    "flagged_type": shaped["flagged_type"],
                    "scored": shaped["scored"],
                    "model_score": shaped["model_score"],
                    # A live item was reviewed with the gateway model's score in
                    # front of the reviewer, so the record of that decision
                    # carries the same number rather than reporting "not scored".
                    "gateway": shaped.get("gateway"),
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


# ---------------------------------------------------------------------------
# Live test-mode ingestion (proof-of-concept).
#
# Two routes, and deliberately no more: one creates a single test-mode order,
# one receives the signed webhook that the resulting payment produces. Neither
# can capture, refund, cancel, or pay out — the enforced boundary is documented
# in src/ingest/razorpay_live.py. Items ingested here are never scored by the
# model; see that module for why a score would be dishonest rather than merely
# unavailable.
# ---------------------------------------------------------------------------


@app.route("/api/live/trigger", methods=["POST"])
def api_live_trigger():
    """Create ONE Razorpay test-mode order so a real payment can be ingested.

    An order is an intent to be paid, not a movement of funds, and it is the
    only outbound payment-rail call this application is capable of making.

    The response carries the order id and the *publishable* key id. That key id
    is designed to be public — Razorpay's hosted checkout needs it in the
    browser — and it only ever leaves the server behind whatever auth this
    instance is running. The key secret and the webhook signing secret never do.
    """
    if READ_ONLY:
        return jsonify({
            "status": "read_only",
            "error": "This instance is running read-only; live ingestion is disabled.",
        }), 403

    if not _csrf_token_is_valid():
        return jsonify({
            "status": "error",
            "error": "Missing or invalid CSRF token.",
        }), 403

    try:
        # Reads the environment and enforces the test-mode prefix before any
        # socket is opened, so a live-mode key cannot produce even one call.
        credentials = load_credentials(require_webhook_secret=False)
        order = create_test_order()
    except RazorpayConfigError as exc:
        # Operator configuration fault, not a client fault: credentials absent,
        # or a live-mode key refused. 503 rather than 500 — the capability is
        # unavailable on this instance, the application is fine.
        app.logger.warning("Live ingestion unavailable: %s", exc)
        return jsonify({"status": "unavailable", "error": str(exc)}), 503
    except LiveIngestionError as exc:
        app.logger.warning("Razorpay order creation failed: %s", exc)
        return jsonify({"status": "error", "error": str(exc)}), 502

    return jsonify({
        "status": "success",
        "order_id": order.get("id"),
        "amount": order.get("amount"),
        "currency": order.get("currency"),
        "key_id": credentials.key_id,
    })


@app.route(WEBHOOK_PATH, methods=["POST"])
def api_razorpay_webhook():
    """Ingest one signature-verified test-mode payment as an UNSCORED item.

    The order of operations is itself the security property: verify the HMAC
    over the raw bytes, then claim the event id, and only then parse. Nothing
    about an unverified payload is parsed, enqueued, or written to a log.
    """
    # Raw bytes, never request.get_json(): the signature covers the body exactly
    # as sent, and a parse-and-re-serialise round trip would invalidate it.
    raw_body = request.get_data()

    secret = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "").strip()
    if not secret:
        app.logger.warning(
            "Razorpay webhook rejected: RAZORPAY_WEBHOOK_SECRET is not set on this instance."
        )
        return jsonify({
            "status": "unavailable",
            "error": "Webhook signing secret is not configured.",
        }), 503

    if not verify_webhook_signature(raw_body, request.headers.get(SIGNATURE_HEADER), secret):
        # Recorded without the body and without the presented signature. An
        # unverified request is attacker-controlled input; the only fact worth
        # keeping is that one arrived and was refused.
        app.logger.warning(
            "Razorpay webhook rejected: signature verification failed (%d byte body).",
            len(raw_body),
        )
        return jsonify({
            "status": "error",
            "error": "Signature verification failed.",
        }), 400

    if READ_ONLY:
        return jsonify({
            "status": "read_only",
            "error": "This instance is running read-only; live ingestion is disabled.",
        }), 503

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        payload = None
    if not isinstance(payload, dict):
        return jsonify({
            "status": "error",
            "error": "Request body must be a JSON object.",
        }), 400

    event_id = extract_event_id(request.headers, payload)
    if not event_id:
        return jsonify({
            "status": "error",
            "error": "Webhook carried no event id and no payment id; refusing to ingest.",
        }), 400

    if not record_event_id(DB_PATH, event_id):
        # Razorpay retries any delivery it did not get a 2xx for, so a replay
        # must answer 200 — a 4xx here would make it retry the duplicate
        # indefinitely. The point is that nothing is enqueued a second time.
        return jsonify({"status": "duplicate_ignored", "event_id": event_id}), 200

    brief = build_live_demo_brief(payload)
    queue_id = ReviewQueue(db_path=DB_PATH).enqueue(brief)
    return jsonify({
        "status": "success",
        "queue_id": queue_id,
        "event_id": event_id,
        "flagged_type": brief.flagged_type,
        # Stated on the wire, not only inferred from flagged_type, so a consumer
        # cannot mistake this item for a scored one.
        "scored": False,
        "model_score": None,
    })


# ---------------------------------------------------------------------------
# Live scoring.
#
# The rest of the site reports what the model produced during evaluation. These
# two routes make the model run: the real XGBoost booster, loaded in this
# process, predicting on the committed feature vector of a real held-out
# transaction. See src/serve/scorer.py for what is committed and what is
# computed.
# ---------------------------------------------------------------------------


@app.route("/api/score/samples")
def api_score_samples():
    """The catalogue of real transactions available to score. No scores in it."""
    try:
        return jsonify({"status": "ok", "samples": list_samples()})
    except ScoringUnavailableError as exc:
        app.logger.warning("Live scoring unavailable: %s", exc)
        return jsonify({"status": "unavailable", "error": str(exc)}), 503


@app.route("/api/score/<sample_id>")
def api_score(sample_id: str):
    """Run the trained model over one real transaction and return its output.

    A GET, because it changes nothing — which also keeps it public under the
    method-based auth gate, so a visitor can exercise the model without a
    password.

    503 rather than a default score when the model cannot load: a number
    returned without the model behind it would read as a prediction, and this
    project does not publish those.
    """
    try:
        return jsonify({"status": "ok", "result": score_sample(sample_id)})
    except KeyError:
        return jsonify({"status": "error", "error": "Unknown sample id."}), 404
    except ScoringUnavailableError as exc:
        app.logger.warning("Live scoring unavailable: %s", exc)
        return jsonify({"status": "unavailable", "error": str(exc)}), 503


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
