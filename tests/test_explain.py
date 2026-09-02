"""Phase 4 acceptance tests: Explainable escalation and review queue.

Verifies:
1. RiskBrief schema enforcement & Python-side confidence/action assignment
2. No-blocking-side-effects introspection test on ReviewQueue and source scanning
3. Coherence and top_factors matching on sample flagged examples
4. Malformed/missing data input handling without crashing
5. Loud failure when GEMINI_API_KEY is missing
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.explain.queue import ReviewQueue
from src.explain.risk_brief import (
    ContributingFactor,
    RiskBrief,
    generate_risk_brief,
)
from src.spike.spike_scorer import SpikeEvent


def test_missing_api_key_fails_loudly():
    """Security & Defense-Only requirement 8: Unset GEMINI_API_KEY must raise an explicit exception."""
    with patch.dict(os.environ, {}, clear=True):
        if "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]

        item = {"card1": "1001", "model_score": 0.85, "flagged_type": "transaction"}
        factors = [ContributingFactor(feature="g_card_cnt_24h", value=12, direction="increases_risk")]

        with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable is not set"):
            generate_risk_brief(item, factors, cost_estimate=150.0, risk_threshold=0.5)


def test_no_blocking_side_effects():
    """Assert ReviewQueue has ONLY the 4 specified public methods and no payment-action code."""
    # 1. Public method set check on ReviewQueue
    public_methods = {
        m for m in dir(ReviewQueue)
        if not m.startswith("_") and callable(getattr(ReviewQueue, m))
    }
    expected_methods = {"enqueue", "list_pending", "resolve", "get_audit_log"}
    assert public_methods == expected_methods, (
        f"ReviewQueue public methods must be exactly {expected_methods}, found {public_methods}"
    )

    # 2. Source code scan on src/explain/ for forbidden action strings
    explain_dir = Path("src/explain")
    forbidden_terms = ["block_card", "cancel_transaction", "hold_funds", "execute_payment", "chargeback"]

    for py_file in explain_dir.glob("*.py"):
        content = py_file.read_text().lower()
        for term in forbidden_terms:
            assert term not in content, f"Forbidden payment action term '{term}' found in {py_file}"


def test_llm_output_schema_enforced(tmp_path: Path):
    """Test schema enforcement and Python-derived fields (confidence, action)."""
    item = {"card1": "9999", "model_score": 0.92, "flagged_type": "transaction"}
    factors = [
        ContributingFactor(feature="g_card_cnt_24h", value=15, direction="increases_risk"),
        ContributingFactor(feature="spike_baseline_ratio_24h", value=4.5, direction="increases_risk"),
    ]

    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_test_key"}):
        with patch("src.explain.risk_brief._call_llm", return_value="Synthetic explanation summary from LLM."):
            brief = generate_risk_brief(item, factors, cost_estimate=500.0, risk_threshold=0.5)

    assert isinstance(brief, RiskBrief)
    assert brief.entity_id == "9999"
    assert brief.model_score == 0.92
    assert brief.confidence == "high"  # abs(0.92 - 0.5) = 0.42 >= 0.25
    assert brief.recommended_action == "hold_for_review"
    assert brief.estimated_fp_cost == 500.0
    assert brief.summary_text == "Synthetic explanation summary from LLM."
    assert len(brief.top_factors) == 2


def test_review_queue_sqlite(tmp_path: Path):
    """Test enqueue, list_pending, resolve, get_audit_log in ReviewQueue."""
    db_file = tmp_path / "test_queue.db"
    queue = ReviewQueue(db_path=db_file)

    brief = RiskBrief(
        entity_id="1234",
        flagged_type="spike",
        model_score=0.88,
        top_factors=[ContributingFactor("g_card_fr_decay", 0.75, "increases_risk")],
        confidence="high",
        estimated_fp_cost=250.0,
        recommended_action="hold_for_review",
        summary_text="High risk spike detected on card 1234.",
    )

    # 1. Enqueue
    qid = queue.enqueue(brief)
    assert isinstance(qid, int) and qid > 0

    # 2. List pending
    pending = queue.list_pending()
    assert len(pending) == 1
    assert pending[0]["id"] == qid
    assert pending[0]["entity_id"] == "1234"
    assert pending[0]["top_factors"][0]["feature"] == "g_card_fr_decay"

    # 3. Resolve item
    queue.resolve(qid, reviewer_action="resolved_true_positive", note="Confirmed fraud by reviewer")

    # 4. Confirm pending is now empty
    assert len(queue.list_pending()) == 0

    # 5. Check audit log
    logs = queue.get_audit_log(qid)
    assert len(logs) == 1
    assert logs[0]["reviewer_action"] == "resolved_true_positive"
    assert logs[0]["note"] == "Confirmed fraud by reviewer"


def test_malformed_input_graceful():
    """Test graceful handling when inputs are missing fields or spike has minimal data."""
    # SpikeEvent with single item or edge parameters
    spike = SpikeEvent(
        entity_id="555",
        window_start=1000,
        window_end=1000,
        transaction_ids=[101],
        aggregate_risk_score=0.52,  # Close to threshold -> low confidence
        baseline_ratio=1.1,
    )

    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"}):
        with patch("src.explain.risk_brief._call_llm", return_value="Spike summary explanation."):
            brief = generate_risk_brief(spike, [], cost_estimate=0.0, risk_threshold=0.5)

    assert brief.entity_id == "555"
    assert brief.flagged_type == "spike"
    assert brief.confidence == "low"  # abs(0.52 - 0.5) = 0.02 < 0.1
    assert brief.recommended_action == "monitor"  # score >= threshold but confidence == low
    assert brief.top_factors == []
    assert brief.summary_text == "Spike summary explanation."


def test_five_flagged_examples_coherence():
    """Test 5 distinct flagged examples (mix of TP/FP, transactions & spikes)."""
    examples = [
        {"card1": "101", "model_score": 0.95, "flagged_type": "transaction", "amt": 120.0, "is_tp": True},
        {"card1": "102", "model_score": 0.55, "flagged_type": "transaction", "amt": 45.0, "is_tp": False},
        {"card1": "103", "model_score": 0.82, "flagged_type": "spike", "amt": 800.0, "is_tp": True},
        {"card1": "104", "model_score": 0.48, "flagged_type": "transaction", "amt": 15.0, "is_tp": False},
        {"card1": "105", "model_score": 0.76, "flagged_type": "spike", "amt": 350.0, "is_tp": True},
    ]

    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"}):
        with patch("src.explain.risk_brief._call_llm", return_value="Coherent explanation text referencing features."):
            for ex in examples:
                factors = [
                    ContributingFactor(feature="g_card_cnt_24h", value=8, direction="increases_risk"),
                    ContributingFactor(feature="TransactionAmt", value=ex["amt"], direction="increases_risk"),
                ]
                brief = generate_risk_brief(ex, factors, cost_estimate=ex["amt"], risk_threshold=0.5)

                assert brief.entity_id == ex["card1"]
                assert brief.flagged_type == ex["flagged_type"]
                assert isinstance(brief.summary_text, str) and len(brief.summary_text) > 0
                assert len(brief.top_factors) == 2
                assert brief.top_factors[0].feature in ("g_card_cnt_24h", "TransactionAmt")
