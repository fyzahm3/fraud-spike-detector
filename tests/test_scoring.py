"""Live model scoring: the site runs the trained model, not a lookup table.

The property that matters is that the number the website shows is the number the
real model produces. `data/score_samples.json` carries, for each committed
transaction, the score the full local pipeline computed with real XGBoost; these
tests require the hosted path to reproduce it exactly. If it ever does not, the
site is displaying something other than the model's answer, which is the failure
this whole surface exists to avoid.

The second property is that there is no fallback. A scoring endpoint that
answers with a plausible default when the model is missing would publish a
fabricated prediction under the appearance of a real one, so absence must be an
error the visitor can see.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import app as app_module
from app import app as flask_app
from src.serve import scorer

SAMPLES_PATH = Path("data/score_samples.json")
MODEL_PATH = Path("artifacts/graph_model.json")

xgboost = pytest.importorskip("xgboost", reason="live scoring needs the ML stack")

pytestmark = pytest.mark.skipif(
    not (SAMPLES_PATH.exists() and MODEL_PATH.exists()),
    reason="trained model or committed score samples are absent",
)


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


def _samples() -> dict:
    with open(SAMPLES_PATH, encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# The score is the model's own
# ---------------------------------------------------------------------------


def test_hosted_score_matches_the_pipelines_own_score():
    """Every committed transaction must score here exactly as it did locally.

    `reference_score` was produced by the full pipeline: real dataset, real
    feature build, real booster. The hosted path starts from the committed
    feature vector instead, and must land on the same number. Equality is
    required to 1e-12 rather than "close": the same model over the same input is
    deterministic, so any drift means the input or the model is not what it
    claims to be.
    """
    payload = _samples()
    for sample in payload["samples"]:
        result = scorer.score(sample["id"])
        assert result["model_score"] == pytest.approx(sample["reference_score"], abs=1e-12), (
            f"{sample['id']} scored {result['model_score']!r} here but "
            f"{sample['reference_score']!r} in the pipeline"
        )


def test_scoring_is_deterministic_across_calls():
    first = scorer.score("txn-01")["model_score"]
    for _ in range(3):
        assert scorer.score("txn-01")["model_score"] == first


def test_every_sample_uses_the_full_feature_space():
    """443 features in, or the vector is not the transaction it claims to be."""
    payload = _samples()
    expected = len(payload["feature_names"])
    assert expected == 443
    for sample in payload["samples"]:
        assert len(sample["features"]) == expected, sample["id"]


def test_flagging_follows_the_committed_threshold():
    payload = _samples()
    threshold = payload["threshold"]
    for sample in payload["samples"]:
        result = scorer.score(sample["id"])
        assert result["threshold"] == pytest.approx(threshold)
        assert result["flagged"] is (result["model_score"] >= threshold)


def test_outcome_labels_are_consistent_with_score_and_truth():
    for sample in _samples()["samples"]:
        r = scorer.score(sample["id"])
        expected = {
            (True, 1): "true_positive",
            (True, 0): "false_positive",
            (False, 1): "false_negative",
            (False, 0): "true_negative",
        }[(r["flagged"], r["actual_label"])]
        assert r["outcome"] == expected, sample["id"]


# ---------------------------------------------------------------------------
# The sample set is honest about the model
# ---------------------------------------------------------------------------


def test_samples_include_the_models_real_mistakes():
    """A demo that only shows clean hits misrepresents the model.

    The committed set must contain genuine false positives and genuine misses,
    because those are the cases a technical reviewer will ask about and the
    ones the system is honest for showing.
    """
    strata = {s["stratum"] for s in _samples()["samples"]}
    for required in ("top_fraud", "borderline", "false_positive", "false_negative"):
        assert required in strata, f"no {required} transactions committed"


def test_samples_are_real_held_out_transactions():
    payload = _samples()
    assert "IEEE-CIS" in payload["source"]
    assert "held-out test split" in payload["source"]
    ids = [s["transaction_id"] for s in payload["samples"]]
    assert len(ids) == len(set(ids)), "duplicate transactions committed"
    assert all(isinstance(i, int) and i > 0 for i in ids)


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


def test_score_api_returns_a_live_computed_result(client):
    res = client.get("/api/score/txn-01")
    assert res.status_code == 200
    body = res.get_json()
    assert body["status"] == "ok"
    result = body["result"]
    assert result["computed_live"] is True
    assert isinstance(result["model_score"], float)
    assert 0.0 <= result["model_score"] <= 1.0
    assert result["n_features"] == 443


def test_score_catalogue_withholds_the_precomputed_score(client):
    """The page must show what this process computed, not a shipped number.

    If the catalogue carried `reference_score`, a client could render it and the
    displayed figure would no longer be evidence that the model ran.
    """
    res = client.get("/api/score/samples")
    assert res.status_code == 200
    for sample in res.get_json()["samples"]:
        assert "reference_score" not in sample
        assert "model_score" not in sample
        assert "features" not in sample
        assert "label" not in sample


def test_unknown_sample_is_404_not_a_guess(client):
    res = client.get("/api/score/does-not-exist")
    assert res.status_code == 404
    assert "model_score" not in res.get_data(as_text=True)


def test_scoring_routes_are_public_reads(client, monkeypatch):
    """GETs, so the auth gate leaves them open and a visitor can run the model."""
    monkeypatch.setenv("DEMO_USER", "demo")
    monkeypatch.setenv("DEMO_PASSWORD", "s3cret")
    for path in ("/api/score/samples", "/api/score/txn-01"):
        res = client.get(path)
        assert res.status_code == 200
        assert "WWW-Authenticate" not in res.headers


# ---------------------------------------------------------------------------
# No fallback
# ---------------------------------------------------------------------------


def test_missing_model_fails_loudly_instead_of_returning_a_number(client, monkeypatch, tmp_path):
    """503 and an explanation — never a default score.

    A plausible number returned without the model behind it is indistinguishable
    to a reader from a real prediction, which makes it the most dangerous thing
    this endpoint could do.
    """
    monkeypatch.setattr(scorer, "MODEL_PATH", tmp_path / "absent_model.json")
    monkeypatch.setattr(scorer, "_STATE", {})

    res = client.get("/api/score/txn-01")
    assert res.status_code == 503
    body = res.get_json()
    assert body["status"] == "unavailable"
    assert "model_score" not in res.get_data(as_text=True)

    assert scorer.is_available() is False
    with pytest.raises(scorer.ScoringUnavailableError):
        scorer.score("txn-01")


def test_metrics_page_offers_the_scorer_and_names_its_source(client):
    body = client.get("/metrics").get_data(as_text=True)
    assert 'id="scorer-select"' in body
    assert "artifacts/graph_model.json" in body
    # The claim on the page must be that the model runs, not that a figure is recalled.
    assert "scores it in this process" in body
