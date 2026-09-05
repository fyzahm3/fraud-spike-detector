"""Risk brief generation using schema-constrained LLM explanations (Phase 4).

Defense-only proof point: When a transaction or spike is flagged, generate a
structured, human-readable risk brief — never an automated action.

Confidence, estimated_fp_cost, and recommended_action are computed in Python
BEFORE the LLM call — the LLM explains them in natural language summary_text,
it does NOT decide them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os

from typing import Literal

try:
    from google import genai
    from google.genai import types
    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False


@dataclass
class ContributingFactor:
    feature: str
    value: float | str
    # "not_a_model_input" is used only by src/ingest: a live-ingested payment
    # carries observed payload fields, which are not features this model was
    # trained on and must not be presented as risk evidence in either direction.
    direction: Literal[
        "increases_risk", "decreases_risk",
        "not_a_model_input",
        # A field the GATEWAY model consumed. Distinct from the two risk
        # directions because it says what the value was used for, not which way
        # it pushed the score, and distinct from not_a_model_input because it
        # was in fact a model input — just not one of the full model's 443.
        "gateway_model_input",
    ]


@dataclass
class RiskBrief:
    entity_id: str
    # "transaction" and "spike" both mean the model scored this item.
    # "live_demo_unscored" (src/ingest) means the opposite, and model_score
    # carries a sentinel the API renders as null rather than a number.
    flagged_type: Literal["transaction", "spike", "live_demo_unscored"]
    model_score: float
    top_factors: list[ContributingFactor]
    confidence: Literal["low", "medium", "high"]      # derived from score distance from threshold
    estimated_fp_cost: float                            # from cost_metrics-style calc
    recommended_action: Literal["hold_for_review", "monitor", "dismiss_low_priority"]  # enum only — never free text
    summary_text: str                                    # natural language summary from LLM

    # Set only for live-ingested items, by src/ingest. It is the GATEWAY
    # model's score — a separate, deliberately weaker model trained on the
    # handful of fields a payment webhook actually carries — and it is never
    # the same number as model_score, which belongs to the full model. Two
    # models, two fields, so neither can be mistaken for the other.
    gateway_score: float | None = None


def generate_risk_brief(
    flagged_item,
    contributing_features: list[ContributingFactor],
    cost_estimate: float,
    risk_threshold: float = 0.5,
) -> RiskBrief:
    """Generate a structured RiskBrief for a flagged transaction or spike.

    Fails loudly if GEMINI_API_KEY environment variable is not set.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set. Loud failure required for defense-only explainer.")

    # Extract entity_id, flagged_type, model_score
    if hasattr(flagged_item, "entity_id") and hasattr(flagged_item, "aggregate_risk_score"):
        # SpikeEvent
        entity_id = str(flagged_item.entity_id)
        flagged_type = "spike"
        model_score = float(flagged_item.aggregate_risk_score)
    elif isinstance(flagged_item, dict):
        entity_id = str(flagged_item.get("card1", flagged_item.get("entity_id", "unknown")))
        flagged_type = str(flagged_item.get("flagged_type", "transaction"))
        model_score = float(flagged_item.get("model_score", 0.0))
    elif hasattr(flagged_item, "to_dict"):
        d = flagged_item.to_dict()
        entity_id = str(d.get("card1", d.get("entity_id", "unknown")))
        flagged_type = "transaction"
        model_score = float(d.get("model_score", 0.0))
    else:
        entity_id = "unknown"
        flagged_type = "transaction"
        model_score = 0.0

    # 1. Compute confidence in Python
    dist = abs(model_score - risk_threshold)
    if dist < 0.1:
        confidence = "low"
    elif dist < 0.25:
        confidence = "medium"
    else:
        confidence = "high"

    # 2. Compute estimated_fp_cost in Python
    estimated_fp_cost = float(cost_estimate)

    # 3. Compute recommended_action in Python
    if model_score >= risk_threshold:
        recommended_action = "hold_for_review" if confidence in ("high", "medium") else "monitor"
    else:
        recommended_action = "dismiss_low_priority"

    # 4. Generate natural language summary via LLM
    top_factors_formatted = [
        f"{f.feature}={f.value} ({f.direction})" for f in contributing_features[:5]
    ]

    prompt = (
        f"You are a defense-only risk analyst assistant. Write a concise 2-3 sentence summary explaining why this "
        f"{flagged_type} was flagged for review.\n\n"
        f"Details:\n"
        f"- Entity ID: {entity_id}\n"
        f"- Flagged Type: {flagged_type}\n"
        f"- Model Risk Score: {model_score:.4f} (Threshold: {risk_threshold})\n"
        f"- System Confidence: {confidence}\n"
        f"- Recommended System Action: {recommended_action}\n"
        f"- Estimated False Positive Cost: ${estimated_fp_cost:,.2f}\n"
        f"- Top Contributing Risk Factors: {', '.join(top_factors_formatted)}\n\n"
        f"Write ONLY the natural language summary explanation text."
    )

    summary_text = _call_llm(prompt, api_key, entity_id, flagged_type, model_score)

    return RiskBrief(
        entity_id=entity_id,
        flagged_type=flagged_type,
        model_score=model_score,
        top_factors=contributing_features,
        confidence=confidence,
        estimated_fp_cost=estimated_fp_cost,
        recommended_action=recommended_action,
        summary_text=summary_text,
    )


def _call_llm(prompt: str, api_key: str, entity_id: str, flagged_type: str, model_score: float) -> str:
    """Invokes Gemini API via google.genai client."""
    if not _HAS_GENAI:
        return f"Entity {entity_id} flagged ({flagged_type}) with risk score {model_score:.4f} based on elevated feature risk."

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        if response and response.text:
            return response.text.strip()
    except Exception as e:
        # Fallback if API call fails (e.g. rate limit or network issue during test)
        return (
            f"Entity {entity_id} flagged ({flagged_type}) with model risk score {model_score:.4f}. "
            f"Automated explanation fallback due to API status: {str(e)}"
        )

    return f"Entity {entity_id} flagged ({flagged_type}) with model risk score {model_score:.4f}."
