"""The four public surfaces: landing, evidence, review queue, live ingestion.

Two properties carry most of the weight here.

The first is that **every number the site shows traces to a committed artifact**.
That is the README's no-fabricated-results rule applied to marketing copy, which
is exactly where the temptation to round a figure up lives. The test reads the
artifacts, formats the values the way the templates do, and requires the page to
contain those exact strings — so a hard-coded figure that drifts from its source
fails rather than quietly misrepresenting the work.

The second is that **reading is public**. Auth coverage lives in tests/test_ui.py;
what is asserted here is that the pages themselves render without credentials.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

import app as app_module
from app import app as flask_app

PAGES = ("/", "/metrics", "/demo", "/live")
TEMPLATE_DIR = Path("templates")


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    from src.explain.queue import ReviewQueue
    db_path = tmp_path / "site.db"
    ReviewQueue(db_path=db_path)
    monkeypatch.setattr(app_module, "DB_PATH", db_path)
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


def _text(client, path: str) -> str:
    res = client.get(path)
    assert res.status_code == 200, f"{path} returned {res.status_code}"
    return res.get_data(as_text=True)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_four_surfaces_render_with_persistent_navigation(client):
    """Four real routes in one app, each linking to the other three."""
    for path in PAGES:
        body = _text(client, path)
        for href in ('href="/"', 'href="/metrics"', 'href="/demo"', 'href="/live"'):
            assert href in body, f"{path} is missing {href}"
        assert 'aria-current="page"' in body, f"{path} does not mark the active nav item"


def test_pages_are_public_without_credentials(client, monkeypatch):
    """A reviewer opening the link sees the product, not a password prompt."""
    monkeypatch.setenv("DEMO_USER", "demo")
    monkeypatch.setenv("DEMO_PASSWORD", "s3cret")
    for path in PAGES:
        res = client.get(path)
        assert res.status_code == 200
        assert "WWW-Authenticate" not in res.headers


# ---------------------------------------------------------------------------
# Every number traces to a committed artifact
# ---------------------------------------------------------------------------


def _artifact(name: str) -> dict:
    with open(Path("results") / name, encoding="utf-8") as handle:
        return json.load(handle)


def test_every_displayed_metric_matches_the_committed_artifact(client):
    """The site may not state a figure that differs from its source file.

    Formatted here exactly as the templates format it, then required verbatim in
    the response. A figure typed into a template by hand, or one that drifted
    after a rerun, fails this rather than misrepresenting the result.
    """
    pc = _artifact("phase_comparison.json")
    run = _artifact("pipeline_run_summary.json")
    split = _artifact("split_manifest.json")

    metrics = _text(client, "/metrics")
    landing = _text(client, "/")

    # Classification metrics, both variants, four decimals.
    for variant in ("baseline", "graph"):
        for key in ("auc_pr", "auc_roc", "precision", "recall", "f1", "ratio"):
            expected = f"{pc[variant][key]:.4f}"
            assert expected in metrics, f"/metrics is missing {variant}.{key} = {expected}"

    # Confusion-matrix counts, thousands-separated.
    for variant in ("baseline", "graph"):
        for key in ("tp", "fp", "fn", "tn"):
            expected = f"{pc[variant][key]:,}"
            assert expected in metrics, f"/metrics is missing {variant}.{key} = {expected}"

    # Run summary.
    assert f"{run['total_rows_scored']:,}" in metrics
    assert f"{run['total_flagged_transactions']:,}" in metrics
    assert f"{run['risk_threshold']:.4f}" in metrics

    # Split manifest, including the real checksum prefixes.
    for name in ("train", "validation", "test"):
        part = split["splits"][name]
        assert f"{part['n_rows']:,}" in metrics
        assert f"{part['n_fraud']:,}" in metrics
        assert part["sha256"][:16] in metrics, f"{name} checksum prefix not shown"

    # The landing page's headline figures come from the same file.
    assert f"{pc['graph']['auc_pr']:.4f}" in landing
    assert f"{pc['baseline']['auc_pr']:.4f}" in landing
    assert f"{pc['graph']['ratio']:.2f}" in landing
    assert f"{pc['graph']['fp']:,}" in landing


def test_no_metric_is_hard_coded_into_a_template():
    """Templates carry format expressions, not literal results.

    A four-decimal literal in a template is a number that cannot be re-derived
    and will not change when the model is retrained — the precise shape of the
    problem this project already had once.
    """
    literal = re.compile(r"(?<![\w.])0\.\d{4}(?![\w])")
    for path in TEMPLATE_DIR.rglob("*.html"):
        found = literal.findall(path.read_text())
        assert not found, f"hard-coded metric literal {found} in {path}"


def test_metrics_page_degrades_rather_than_inventing_or_crashing(client, tmp_path, monkeypatch):
    """A missing artifact is stated plainly; it is never filled in from memory."""
    monkeypatch.setattr(app_module, "RESULTS_DIR", tmp_path)

    res = client.get("/metrics")
    assert res.status_code == 200, "a missing artifact must not take the page down"
    body = res.get_data(as_text=True)
    assert "not available in this deployment" in body

    # And with no source file, no result figure appears anywhere on the page.
    pc = _artifact("phase_comparison.json")
    assert f"{pc['graph']['auc_pr']:.4f}" not in body

    # The landing page's evidence blocks disappear rather than rendering empty.
    landing = client.get("/")
    assert landing.status_code == 200


def test_evidence_loader_reports_missing_files_as_none(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "RESULTS_DIR", tmp_path)
    evidence = app_module.load_evidence()
    assert set(evidence) == set(app_module.EVIDENCE_FILES)
    assert all(value is None for value in evidence.values())


# ---------------------------------------------------------------------------
# Contextual help
# ---------------------------------------------------------------------------


REQUIRED_HELP_TOPICS = {
    "auc_pr", "cost_ratio", "chronological_split", "llm_prose", "live_unscored",
}


def test_help_covers_the_required_topics_with_written_content():
    for topic in REQUIRED_HELP_TOPICS:
        assert topic in app_module.HELP_TOPICS, f"no help written for {topic}"
        entry = app_module.HELP_TOPICS[topic]
        assert entry["title"].strip()
        # Long enough to be an explanation rather than a label.
        assert len(entry["body"]) > 200, f"{topic} help is too thin to be useful"


def test_help_is_static_and_cannot_call_a_model():
    """No LLM behind the '?' — deterministic, fast, and unable to hallucinate.

    An explanation of this system's own evaluation protocol is the last text
    that should be improvised: a plausible-sounding wrong answer about the
    methodology is worse than no answer.
    """
    source = Path("app.py").read_text()
    for forbidden in ("genai", "generate_risk_brief", "GEMINI"):
        assert forbidden not in source, f"app.py reaches for {forbidden}"

    script = Path("static/js/console.js").read_text()
    help_section = script[script.index("function initHelp"):script.index("function initMotion")]
    assert "fetch(" not in help_section, "help content is fetched rather than rendered"


def test_help_content_reaches_the_page_as_escaped_text(client):
    """Rendered by Jinja into a text node; never assembled as markup."""
    body = _text(client, "/metrics")
    entry = app_module.HELP_TOPICS["auc_pr"]
    assert entry["title"] in body
    assert entry["body"][:60] in body
    assert 'data-help-topic="auc_pr"' in body


# ---------------------------------------------------------------------------
# Copy discipline: defense-only, and nothing fabricated
# ---------------------------------------------------------------------------


def _copy_sources() -> dict[Path, str]:
    files = list(TEMPLATE_DIR.rglob("*.html"))
    files.append(Path("static/js/console.js"))
    return {path: path.read_text() for path in files}


def test_copy_uses_defense_only_vocabulary():
    """Marketing copy is held to the same rule as the code.

    Phrase-level rather than word-level: "block" appears legitimately in CSS and
    in "masked V1-V339 block", and banning the token would be noise. What may
    not appear is language claiming this system acts on a payment.
    """
    forbidden_phrases = [
        "block the payment", "block a payment", "block payments", "blocks the payment",
        "block the transaction", "block transactions", "blocked transaction",
        "cancel the payment", "cancel the transaction", "cancels the payment",
        "hold funds", "holds funds", "holding funds", "freeze the account",
        "stop the payment", "stops the payment", "stop fraud in real time",
        "decline the payment", "declines the payment", "reject the payment",
        "prevent the transaction", "auto-block", "automatically block",
    ]
    for path, content in _copy_sources().items():
        lowered = content.lower()
        for phrase in forbidden_phrases:
            assert phrase not in lowered, f"'{phrase}' in {path}"


def test_copy_makes_no_fabricated_commercial_claims():
    """No customers, testimonials, pricing, adoption figures, or partner logos.

    None of these exist, so none of them may appear. This is the same rule as
    the metrics: if it is not backed by something committed, it is not stated.
    """
    forbidden = [
        "testimonial", "our customers", "trusted by", "used by leading",
        "case study", "pricing", "per month", "/mo", "free trial",
        "enterprise plan", "book a demo", "request a quote", "roi of",
        "customers say", "rated 5", "award-winning", "industry-leading",
        "banks trust", "processing crores", "clients include",
    ]
    for path, content in _copy_sources().items():
        lowered = content.lower()
        for phrase in forbidden:
            assert phrase not in lowered, f"unsupported commercial claim '{phrase}' in {path}"


def test_landing_page_does_not_imply_live_real_time_scoring(client):
    """The live demo is ingestion, and the copy may not suggest otherwise."""
    landing = _text(client, "/")
    lowered = landing.lower()
    for phrase in ("real-time scoring", "scores live payments", "live risk scoring",
                   "scores every payment as it happens"):
        assert phrase not in lowered, f"landing copy implies {phrase!r}"

    # And it states the limitation positively somewhere on the page.
    assert "unscored" in lowered


def test_india_market_honesty_survives_on_the_site(client):
    """The dataset gap is stated on the site, not only in the README."""
    metrics = _text(client, "/metrics").lower()
    assert "upi" in metrics
    assert "synthetic" in metrics
    assert "ieee-cis" in metrics


# ---------------------------------------------------------------------------
# Cold start
# ---------------------------------------------------------------------------


def test_cold_start_ui_cannot_hang_silently():
    """A slow instance must say so, and a stalled one must fail with a retry."""
    script = Path("static/js/console.js").read_text()

    assert "AbortController" in script, "no request timeout; a stall would spin forever"
    assert "REQUEST_TIMEOUT_MS" in script
    assert "SLOW_REQUEST_MS" in script
    assert "waking-banner" in script
    # The message names the cause and the rough duration rather than just spinning.
    assert "Waking the server" in script
    assert "30 seconds" in script

    css = Path("static/css/site.css").read_text()
    assert ".spinner" in css
    assert "prefers-reduced-motion" in css, "the spinner must respect reduced motion"

    # An element with its own `display` outranks the UA's [hidden] rule, so the
    # banner stays on screen while the DOM reports it hidden. This bit twice
    # during the build; the general rule is what keeps it fixed.
    assert "[hidden] { display: none !important; }" in css, (
        "without this, hiding an element by property silently does nothing"
    )


def test_recording_a_decision_is_visibly_confirmed():
    """A decision button must say that it did something.

    The first version of this flow was silent: the card left a thirty-item list
    and a count changed on a tab the reviewer was not looking at, which is
    indistinguishable from a dead button. Recording a decision is the one
    consequential act in this interface and the least reversible, so the
    confirmation is asserted rather than left to judgement.
    """
    script = Path("static/js/console.js").read_text()

    # Confirmed in place, naming the decision and where it was written.
    assert "brief-recorded" in script
    assert "Decision recorded: " in script
    assert "append-only audit log" in script

    # Announced, with a route to the consequence.
    assert "function toast" in script
    assert "View audit trail" in script
    assert "showAuditTrail" in script
    assert 'data-queue-id' in script, "the written audit row cannot be located to flash"
    assert "is-flash" in script

    # A failure has to be as loud as a success.
    assert "Decision NOT recorded" in script

    # The refresh is deferred; without the pause the confirmation is replaced
    # within a few hundred milliseconds and the card simply vanishes again.
    assert "2600" in script, "the list refresh is not held long enough to read"

    css = Path("static/css/site.css").read_text()
    for selector in (".toast", ".brief-recorded", ".audit tbody tr.is-flash"):
        assert selector in css, f"{selector} has no styling"


def test_health_is_public_and_cheap_enough_to_ping(client):
    """The external uptime monitor pings this every few minutes."""
    res = client.get("/health")
    assert res.status_code == 200
    assert "WWW-Authenticate" not in res.headers
    assert res.get_json()["status"] == "ok"
