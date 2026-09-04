"""Live test-mode payment ingestion (proof-of-concept).

Ingestion only. Nothing in this package can move money, refund, or hold a
payment — see src/ingest/razorpay_live.py for the enforced boundary.
"""

from src.ingest.razorpay_live import (
    LIVE_DEMO_FLAGGED_TYPE,
    UNSCORED_MODEL_SCORE,
    LiveIngestionError,
    RazorpayConfigError,
    build_live_demo_brief,
    create_test_order,
    load_credentials,
    record_event_id,
    verify_webhook_signature,
)

__all__ = [
    "LIVE_DEMO_FLAGGED_TYPE",
    "UNSCORED_MODEL_SCORE",
    "LiveIngestionError",
    "RazorpayConfigError",
    "build_live_demo_brief",
    "create_test_order",
    "load_credentials",
    "record_event_id",
    "verify_webhook_signature",
]
