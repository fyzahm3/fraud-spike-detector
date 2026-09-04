"""Score a real held-out transaction with the real trained model, on request.

The rest of the site reports numbers the model produced during evaluation. This
module makes the model itself run in the web process: the actual XGBoost booster
from artifacts/, loaded once per worker, predicting on the committed feature
vectors of real held-out transactions.

Why the real library rather than a reimplementation
---------------------------------------------------
The model and its inference stack are the substance of the project, so the
hosted demo runs the same XGBoost that produced the committed metrics — not a
port of it that would have to be argued equivalent. The cost is that the web
tier now carries xgboost and numpy: about 250MB resident once the booster is
loaded, against the instance's 512MB. That is a deliberate trade, and the
budget is checked in tests/test_scoring.py rather than assumed.

What is committed, and what is computed
---------------------------------------
Committed: the trained model, and the feature vectors of real held-out
transactions (the ~650MB dataset and its sequential feature pipeline cannot be
deployed). Computed here, per request: the score. The site therefore shows a
number this process produced, and `reference_score` in the sample file lets a
test prove that number equals what the full local pipeline produced.

Nothing here degrades. If the model or the samples are missing, requests fail
loudly — a scoring endpoint that answers with a plausible default would be
publishing a fabricated prediction, which is the one thing this project does
not do.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any

MODEL_PATH = Path("artifacts/graph_model.json")
SAMPLES_PATH = Path("data/score_samples.json")

# One booster per process, built on first use rather than at import. gunicorn
# runs a single worker with several threads here, so the ~250MB is paid once;
# building it lazily keeps a missing artifact from taking the whole app down at
# boot, when every other page would still work.
_LOCK = threading.Lock()
_STATE: dict[str, Any] = {}


class ScoringUnavailableError(RuntimeError):
    """The model or the sample set could not be loaded.

    Surfaced to the client as an explicit failure. There is deliberately no
    fallback score: a number returned without the model behind it would be
    indistinguishable, to a reader, from a real prediction.
    """


def _load() -> dict[str, Any]:
    if _STATE:
        return _STATE
    with _LOCK:
        if _STATE:  # another thread won the race while this one waited
            return _STATE
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ScoringUnavailableError(
                f"xgboost is not installed in this environment ({exc}). Live scoring "
                "requires it; see requirements-web.txt."
            ) from None

        if not MODEL_PATH.exists():
            raise ScoringUnavailableError(
                f"Trained model not found at {MODEL_PATH}. Run `python train.py`."
            )
        if not SAMPLES_PATH.exists():
            raise ScoringUnavailableError(
                f"Score samples not found at {SAMPLES_PATH}. Run "
                "`python scripts/make_score_samples.py`."
            )

        booster = xgb.Booster()
        booster.load_model(str(MODEL_PATH))
        # The instance is 0.1 CPU. Letting XGBoost fan out across imagined cores
        # costs more in contention than it saves, and a single row's prediction
        # is a couple of milliseconds single-threaded anyway.
        booster.set_param({"nthread": 1})

        with open(SAMPLES_PATH, encoding="utf-8") as handle:
            payload = json.load(handle)

        samples = {s["id"]: s for s in payload["samples"]}
        _STATE.update({
            "xgb": xgb,
            "booster": booster,
            "samples": samples,
            "order": [s["id"] for s in payload["samples"]],
            "threshold": float(payload["threshold"]),
            "feature_names": payload["feature_names"],
            "variant": payload.get("variant", "graph"),
        })
    return _STATE


def is_available() -> bool:
    """Whether live scoring can run here, without raising."""
    try:
        _load()
        return True
    except ScoringUnavailableError:
        return False


def list_samples() -> list[dict[str, Any]]:
    """The catalogue the page offers, with no scores in it.

    `reference_score` is withheld on purpose: the number the page displays must
    be the one this process just computed, so the client is never handed a
    pre-computed score it could show instead.
    """
    state = _load()
    return [
        {
            "id": s["id"],
            "transaction_id": s["transaction_id"],
            "stratum": s["stratum"],
            "amount": s["amount"],
        }
        for s in (state["samples"][i] for i in state["order"])
    ]


def score(sample_id: str) -> dict[str, Any]:
    """Run the real model over one real transaction, now."""
    state = _load()
    sample = state["samples"].get(sample_id)
    if sample is None:
        raise KeyError(sample_id)

    xgb = state["xgb"]
    booster = state["booster"]

    # None is JSON's stand-in for the NaN that XGBoost reads as missing.
    row = [[float("nan") if v is None else float(v) for v in sample["features"]]]
    matrix = xgb.DMatrix(
        row,
        feature_names=booster.feature_names,
        feature_types=booster.feature_types,
        enable_categorical=True,
    )
    probability = float(booster.predict(matrix)[0])
    threshold = state["threshold"]
    flagged = probability >= threshold

    label = int(sample["label"])
    if flagged and label == 1:
        outcome = "true_positive"
    elif flagged and label == 0:
        outcome = "false_positive"
    elif not flagged and label == 1:
        outcome = "false_negative"
    else:
        outcome = "true_negative"

    return {
        "id": sample["id"],
        "transaction_id": sample["transaction_id"],
        "stratum": sample["stratum"],
        "amount": sample["amount"],
        "model_score": probability,
        "threshold": threshold,
        "flagged": flagged,
        # Ground truth is shown here because this is an evaluation surface, not
        # the reviewer's queue: the point is to let a visitor check the model
        # against reality, including where it is wrong. The review queue still
        # holds no labels, so a reviewer is never shown the answer.
        "actual_label": label,
        "outcome": outcome,
        "computed_live": True,
        "variant": state["variant"],
        "n_features": len(sample["features"]),
    }
