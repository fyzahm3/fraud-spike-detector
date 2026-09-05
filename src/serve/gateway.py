"""The gateway model: scoring a payment with what is knowable at authorization.

The full model reaches AUC-PR 0.6732 using 443 features, and most of its
strength comes from an entity's accumulated history — device graphs, email
co-occurrence, rolling velocity. At the instant a payment is authorized none of
that exists yet. The webhook carries an amount, a card network, a card type, an
email domain and a timestamp.

So this scores that payment with a model trained on the same real IEEE-CIS
dataset and the same chronological split, restricted to exactly those fields.
It reaches AUC-PR 0.1430 on the same held-out data. That number is not a
disappointment to be hidden; it is the measured cost of the history that does
not exist yet, and it is why production fraud systems run a fast gateway pass at
authorization and a richer re-score once the history has accumulated.

Two rules this module exists to enforce:

1. The gateway score is never presented as the full model's score. They are
   separate fields, separately labelled, each carrying its own measured
   performance.
2. A payment on another rail or in another currency is outside this model's
   training distribution. The score is real and really computed; what it is
   evidence *of* is stated alongside it rather than assumed.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any
from urllib.parse import urlparse

MODEL_PATH = Path("artifacts/gateway_model.json")
META_PATH = Path("artifacts/gateway_meta.json")
METRICS_PATH = Path("results/gateway_metrics.json")

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {}


class GatewayUnavailableError(RuntimeError):
    """The gateway model could not be loaded. Never substituted with a default."""


def _load() -> dict[str, Any]:
    if _STATE:
        return _STATE
    with _LOCK:
        if _STATE:
            return _STATE
        try:
            import numpy as np
            import xgboost as xgb
        except ImportError as exc:
            raise GatewayUnavailableError(f"ML stack unavailable: {exc}") from None

        for path in (MODEL_PATH, META_PATH):
            if not path.exists():
                raise GatewayUnavailableError(
                    f"{path} is absent. Run scripts/train_gateway_model.py."
                )

        booster = xgb.Booster()
        booster.load_model(str(MODEL_PATH))
        booster.set_param({"nthread": 1})

        with open(META_PATH, encoding="utf-8") as fh:
            meta = json.load(fh)

        metrics = None
        if METRICS_PATH.exists():
            try:
                with open(METRICS_PATH, encoding="utf-8") as fh:
                    metrics = json.load(fh).get("metrics")
            except (OSError, ValueError):
                metrics = None

        _STATE.update({
            "np": np, "xgb": xgb, "booster": booster, "meta": meta, "metrics": metrics,
            "features": meta["feature_names"],
            "categories": {k: list(v) for k, v in meta["categories"].items()},
            "threshold": float(meta["threshold"]),
        })
    return _STATE


def is_available() -> bool:
    try:
        _load()
        return True
    except GatewayUnavailableError:
        return False


def _email_domain(email: str | None) -> str | None:
    if not email or "@" not in email:
        return None
    return email.rsplit("@", 1)[1].strip().lower() or None


def map_payload(payment: dict[str, Any]) -> dict[str, Any]:
    """Project a Razorpay payment entity onto the model's feature space.

    Every value here is read from the payload. Nothing is inferred, defaulted to
    a plausible-looking value, or filled in to improve the score: a field the
    webhook did not carry stays missing, and XGBoost treats it as missing, which
    is exactly what it is.
    """
    import datetime as _dt

    card = payment.get("card") or {}

    amount_paise = payment.get("amount")
    try:
        amount = float(amount_paise) / 100.0
    except (TypeError, ValueError):
        amount = None

    created = payment.get("created_at")
    hour = weekday = None
    if isinstance(created, (int, float)):
        moment = _dt.datetime.fromtimestamp(float(created), tz=_dt.timezone.utc)
        hour, weekday = float(moment.hour), float(moment.weekday())

    network = (card.get("network") or "").strip().lower() or None
    card_type = (card.get("type") or "").strip().lower() or None

    return {
        "TransactionAmt": amount,
        "hour_of_day": hour,
        "day_of_week": weekday,
        "card4": network,
        "card6": card_type,
        "P_emaildomain": _email_domain(payment.get("email")),
        # The dataset's addr2 is a numeric issuing-country code; a webhook gives
        # a boolean "international" instead, which does not map onto it. Left
        # missing rather than guessed.
        "addr2": None,
    }


def score_payment(payment: dict[str, Any]) -> dict[str, Any]:
    """Score one payment entity. Returns the score and how it was reached."""
    state = _load()
    np, xgb = state["np"], state["xgb"]
    features = state["features"]
    mapped = map_payload(payment)

    row = []
    for name in features:
        value = mapped.get(name)
        if name in state["categories"]:
            # Category codes must match the training vocabulary; a value the
            # model never saw is missing, not a new category.
            try:
                row.append(float(state["categories"][name].index(str(value))))
            except (ValueError, AttributeError):
                row.append(float("nan"))
        else:
            row.append(float("nan") if value is None else float(value))

    matrix = xgb.DMatrix(
        np.array([row], dtype=np.float32),
        feature_names=state["booster"].feature_names,
        feature_types=state["booster"].feature_types,
        enable_categorical=True,
    )
    probability = float(state["booster"].predict(matrix)[0])
    threshold = state["threshold"]

    observed = {k: v for k, v in mapped.items() if v is not None}
    return {
        "gateway_score": probability,
        "threshold": threshold,
        "flagged": probability >= threshold,
        "n_features": len(features),
        "features_present": len(observed),
        "observed": mapped,
        "auc_pr": (state["metrics"] or {}).get("auc_pr"),
        "training_domain": state["meta"].get("training_domain"),
    }
