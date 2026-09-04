"""Razorpay test-mode live ingestion (proof-of-concept).

What this module does, in full:

1. Creates ONE small test-mode order via Razorpay's Orders API.
2. Verifies the signature on the webhook that order's payment produces.
3. Turns that verified payload into an UNSCORED review-queue item.

What it deliberately cannot do
------------------------------
There is no code path here that refunds, pays out, captures, cancels, or
subscribes. `create_test_order` is the only outbound call in the package and it
posts to exactly one endpoint, `POST /v1/orders`. That is the defense-only rule
applied to a live payment rail: the system observes a payment, it never moves
one.

Why the ingested item carries no risk score
-------------------------------------------
The trained model expects the IEEE-CIS feature space — ~430 engineered columns
including the masked V1-V339 block, plus the causal co-occurrence features that
src/features/graph_features.py derives from an entity's own transaction history.
A Razorpay webhook payload carries an amount, a currency, a method, an order id
and a timestamp. Those two spaces do not overlap, so there is no honest way to
produce a model score from one.

Defaulting the missing ~430 features to zeros and calling predict() WOULD return
a number. That number would be a property of the padding, not of the payment.
This project already shipped one metric that was an artifact of its data rather
than its model (the PaySim label-leakage incident recorded in the README), found
it, and deleted it. Producing a second one on purpose, to make a demo look
complete, is not a trade this codebase makes.

So the item is stored with flagged_type="live_demo_unscored" and a sentinel
model_score that the API layer renders as null, never as a number.

No new dependencies
-------------------
Order creation uses urllib and signature verification uses hmac, both stdlib,
so requirements-web.txt stays Flask + gunicorn and the free-tier build budget is
untouched. That constraint is documented in CLAUDE.md.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any
import urllib.error
import urllib.request

from src.explain.risk_brief import ContributingFactor, RiskBrief

# --- Constants -------------------------------------------------------------

#: flagged_type for a live-ingested item. Deliberately not "transaction" or
#: "spike": those two mean "the model scored this". This one means the opposite,
#: and every consumer keys off the difference.
LIVE_DEMO_FLAGGED_TYPE = "live_demo_unscored"

#: review_queue.model_score is REAL NOT NULL, so an unscored row still needs a
#: value. A negative number is impossible for a probability, which makes this
#: unambiguously a sentinel rather than a plausible-looking score. app.py maps
#: it to JSON null before it ever reaches a reviewer.
UNSCORED_MODEL_SCORE = -1.0

#: Fixed summary for every live-ingested item. Hard-coded, never LLM-generated:
#: the whole point of this text is that it states the absence of a score
#: accurately, and an improvised paraphrase is exactly the failure mode to
#: avoid. src/explain/risk_brief.py is not called for these items at all.
LIVE_DEMO_SUMMARY = (
    "Live ingestion proof-of-concept. This item is a real Razorpay test-mode "
    "payment received over a signature-verified webhook, and it is NOT scored "
    "by the fraud model. No risk score is assigned because the payment payload "
    "does not contain the model's feature space: the model was trained on the "
    "IEEE-CIS schema (~430 engineered features, including the masked V1-V339 "
    "block and causal graph features derived from an entity's transaction "
    "history), and a webhook payload carries only an amount, a currency, a "
    "payment method and a timestamp. Any number shown here would describe "
    "placeholder values rather than this payment, so none is shown. This item "
    "demonstrates ingestion and human review, not detection."
)

RAZORPAY_ORDERS_URL = "https://api.razorpay.com/v1/orders"

#: Razorpay key ids are environment-prefixed: rzp_test_* for the sandbox,
#: rzp_live_* for production. Verified against Razorpay's API-keys docs, Sept
#: 2026. This module accepts the first prefix and nothing else.
TEST_KEY_PREFIX = "rzp_test_"

#: Header names, verified against Razorpay's webhook-validation docs, Sept 2026.
SIGNATURE_HEADER = "X-Razorpay-Signature"
EVENT_ID_HEADER = "X-Razorpay-Event-Id"

#: The single test order this proof-of-concept creates: one rupee, fixed. The
#: amount is not caller-supplied, so no request can enlarge it.
TEST_ORDER_AMOUNT_PAISE = 100
TEST_ORDER_CURRENCY = "INR"

ORDER_REQUEST_TIMEOUT_SECONDS = 15


class LiveIngestionError(RuntimeError):
    """A live-ingestion step failed. Raised loudly; never swallowed."""


class RazorpayConfigError(LiveIngestionError):
    """Credentials are missing, malformed, or not test-mode."""


@dataclass(frozen=True)
class RazorpayCredentials:
    key_id: str
    key_secret: str
    webhook_secret: str


# --- Credentials -----------------------------------------------------------

def load_credentials(require_webhook_secret: bool = True) -> RazorpayCredentials:
    """Read Razorpay credentials from the environment, or fail loudly.

    Mirrors generate_risk_brief's contract for GEMINI_API_KEY: absent
    configuration raises rather than degrading to some silent default. There is
    no fallback key and no hard-coded value anywhere in this package.

    The test-mode check happens here, before any caller can reach the network,
    so a live-mode key cannot be used to make even one API call. That is a
    correctness requirement of the defense-only rule, not a convenience: this
    project must not be capable of touching real money.
    """
    key_id = os.environ.get("RAZORPAY_KEY_ID", "").strip()
    key_secret = os.environ.get("RAZORPAY_KEY_SECRET", "").strip()
    webhook_secret = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "").strip()

    missing = [
        name for name, value in (
            ("RAZORPAY_KEY_ID", key_id),
            ("RAZORPAY_KEY_SECRET", key_secret),
        ) if not value
    ]
    if require_webhook_secret and not webhook_secret:
        missing.append("RAZORPAY_WEBHOOK_SECRET")
    if missing:
        raise RazorpayConfigError(
            "Razorpay environment variable(s) not set: " + ", ".join(missing)
            + ". Live ingestion requires test-mode credentials; see .env.example."
        )

    assert_test_mode_key(key_id)
    return RazorpayCredentials(key_id, key_secret, webhook_secret)


def assert_test_mode_key(key_id: str) -> None:
    """Reject anything that is not a Razorpay test-mode key id.

    Whitelist, not blacklist: an unrecognised prefix is refused rather than
    assumed safe, so a future key format cannot quietly pass as test mode. The
    key id itself is never included in the error — it is a credential.
    """
    if not key_id.startswith(TEST_KEY_PREFIX):
        raise RazorpayConfigError(
            f"RAZORPAY_KEY_ID is not a test-mode key (expected a {TEST_KEY_PREFIX!r} "
            "prefix). This project is defense-only and must never be able to reach "
            "a live payment environment. Refusing to make any API call."
        )


# --- Order creation (the only outbound call in this package) ---------------

def create_test_order(receipt: str | None = None) -> dict[str, Any]:
    """Create one small test-mode order. Returns Razorpay's order object.

    This is the entire outbound surface of the live-ingestion feature: one POST
    to /v1/orders with a fixed one-rupee amount. Nothing here can capture,
    refund, or pay out, and the amount is not a parameter.
    """
    creds = load_credentials(require_webhook_secret=False)

    receipt = receipt or f"fsd-live-{int(time.time())}"
    body = json.dumps({
        "amount": TEST_ORDER_AMOUNT_PAISE,
        "currency": TEST_ORDER_CURRENCY,
        # Razorpay caps receipt at 40 characters.
        "receipt": receipt[:40],
        "notes": {"purpose": "fraud-spike-detector live ingestion proof-of-concept"},
    }).encode("utf-8")

    request = urllib.request.Request(
        RAZORPAY_ORDERS_URL, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    # Basic auth, per Razorpay's API authentication scheme.
    handler = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    handler.add_password(None, RAZORPAY_ORDERS_URL, creds.key_id, creds.key_secret)
    opener = urllib.request.build_opener(urllib.request.HTTPBasicAuthHandler(handler))

    try:
        with opener.open(request, timeout=ORDER_REQUEST_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Razorpay's error body describes the rejection and contains no secret,
        # but the request that produced it did carry one, so nothing about the
        # request is echoed back here.
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise LiveIngestionError(
            f"Razorpay order creation failed with HTTP {exc.code}: {detail}"
        ) from None
    except urllib.error.URLError as exc:
        raise LiveIngestionError(
            f"Could not reach the Razorpay API: {exc.reason}"
        ) from None


# --- Webhook signature verification ----------------------------------------

def verify_webhook_signature(raw_body: bytes, signature: str | None, secret: str) -> bool:
    """HMAC-SHA256 over the RAW request body, hex digest, compared constant-time.

    Scheme and header name confirmed against Razorpay's webhook-validation docs
    (Sept 2026) rather than assumed. The body must be the bytes as received: a
    parse-and-re-serialise round trip changes whitespace and key order and would
    invalidate every real signature.

    Returns False rather than raising, so the caller decides the status code —
    but a False here means the payload is not processed, ever.
    """
    if not signature or not secret:
        return False
    expected = hmac.new(
        secret.encode("utf-8"), raw_body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature.strip())


# --- Replay protection ------------------------------------------------------

_EVENTS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS live_ingestion_events (
    event_id TEXT PRIMARY KEY,
    received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def record_event_id(db_path: str | Path, event_id: str) -> bool:
    """Claim an event id. True if newly seen, False if this is a replay.

    Razorpay retries a webhook it did not get a 2xx for, so the same event can
    arrive several times legitimately. The PRIMARY KEY does the work: the claim
    and the check are one atomic INSERT, so two concurrent deliveries of the
    same event cannot both win the race and enqueue a duplicate item.

    Kept out of ReviewQueue on purpose. That class's public API is fixed at four
    methods and asserted by tests/test_explain.py; this is a separate concern
    with its own table, not a fifth queue method.
    """
    if not event_id:
        raise ValueError("event_id must be a non-empty string")

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5.0)
    try:
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute(_EVENTS_TABLE_DDL)
        conn.commit()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO live_ingestion_events (event_id) VALUES (?);",
                    (event_id,),
                )
            return True
        except sqlite3.IntegrityError:
            return False
    finally:
        conn.close()


# --- Payload -> unscored review item ---------------------------------------

def extract_event_id(headers: Any, payload: dict[str, Any]) -> str:
    """Razorpay's own event id, else the payment id. Empty string if neither."""
    header_value = ""
    if headers is not None:
        try:
            header_value = (headers.get(EVENT_ID_HEADER) or "").strip()
        except AttributeError:
            header_value = ""
    if header_value:
        return header_value
    entity = _payment_entity(payload)
    return str(entity.get("id", "")).strip()


def _payment_entity(payload: dict[str, Any]) -> dict[str, Any]:
    entity = (
        payload.get("payload", {})
        .get("payment", {})
        .get("entity", {})
    )
    return entity if isinstance(entity, dict) else {}


def build_live_demo_brief(payload: dict[str, Any]) -> RiskBrief:
    """Turn a verified webhook payload into an UNSCORED review-queue item.

    Every risk-bearing field is neutralised rather than estimated:

    - `model_score` is the sentinel, which the API renders as null.
    - `confidence` is "low" because there is no scored judgement to be
      confident about; the UI suppresses the tag for this flagged_type so the
      word never appears next to a live item.
    - `recommended_action` is "monitor" — the only one of the three enum values
      that asserts nothing about risk.
    - `summary_text` is the fixed LIVE_DEMO_SUMMARY constant. No LLM call is
      made for these items, so there is nothing for a model to improvise around.

    `top_factors` are observed payload fields, and every one is tagged
    "not_a_model_input" rather than a risk direction, because none of them is a
    feature this model was trained on.
    """
    entity = _payment_entity(payload)
    amount_paise = entity.get("amount", 0)
    try:
        amount_display = f"{int(amount_paise) / 100:.2f}"
    except (TypeError, ValueError):
        amount_display = "unknown"

    observed = [
        ("payment_id", str(entity.get("id", "unknown"))),
        ("order_id", str(entity.get("order_id", "unknown"))),
        ("amount", f"{amount_display} {entity.get('currency', '')}".strip()),
        ("method", str(entity.get("method", "unknown"))),
        ("event", str(payload.get("event", "unknown"))),
    ]
    factors = [
        ContributingFactor(
            feature=name,
            value=value,
            # Not "increases_risk"/"decreases_risk": these are observed payload
            # fields, not model inputs, and must not read as risk evidence.
            direction="not_a_model_input",
        )
        for name, value in observed
    ]

    return RiskBrief(
        entity_id=str(entity.get("id") or payload.get("event") or "live-demo"),
        flagged_type=LIVE_DEMO_FLAGGED_TYPE,
        model_score=UNSCORED_MODEL_SCORE,
        top_factors=factors,
        confidence="low",
        estimated_fp_cost=0.0,
        recommended_action="monitor",
        summary_text=LIVE_DEMO_SUMMARY,
    )
