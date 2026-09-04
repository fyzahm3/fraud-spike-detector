# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Flask 3.1.3 + gunicorn, server-rendered. **Static assets, no build step** (user decision): CSS/JS may be split out of `app.py`'s inline template into files served by Flask, but no npm, no bundler, and no package manager may enter the dashboard's dependency chain. Deploy target is Render's free tier via `render.yaml`; `requirements-web.txt` is Flask + gunicorn only and must stay that way.

## Users

Two audiences, judges first:

- **Razorpay AI Buildathon 2026 judging panel** (primary for design decisions) — evaluating the submission in a short demo/pitch window, on a hosted instance that may be waking from sleep. They need the claim, the evidence, and the defense-only stance legible at a glance.
- **Fraud/risk analysts** (the modeled user) — clearing a queue of flagged risk briefs during a shift, deciding on each and logging that decision to an audit trail. They need scan speed, decision throughput, and audit clarity.

The surface must read as a credible working analyst tool while surviving a three-minute evaluation.

## Product Purpose

Detect *spikes* of fraudulent activity on card and digital payment rails — bursts, not isolated transactions — and route them to a human as an explainable, auditable risk brief. Per-transaction XGBoost scoring feeds rolling per-entity aggregation, which feeds an LLM-written brief, which lands in a SQLite review queue with an append-only audit log. Success is a reviewer making a faster, better-evidenced decision, with a record of why.

## Positioning

The system **flags and explains; it never acts on money.** Defense-only is enforced in code rather than asserted in policy — `ReviewQueue`'s public API is asserted to be exactly `{enqueue, list_pending, resolve, get_audit_log}`, and the source is scanned for action verbs. The second differentiator is causal, leakage-safe graph features: 12 card/email/device/address co-occurrence signals over 24h/7d windows plus 48h decayed neighbour fraud mass, every one of them computed from strictly-past rows.

## Operating Context

- The reviewer works a queue: read brief → weigh the contributing factors and confidence → log one of *Confirm True Positive*, *Dismiss False Positive*, or *Escalate*. Resolution appends to `audit_log`; nothing is deleted or overwritten.
- The hosted demo (`https://fraud-spike-review-queue.onrender.com`) serves the dashboard **alone**, without the training or scoring pipeline, reading the committed `data/demo_review_queue.db` snapshot. Basic auth is on; `/health` is exempt.
- Free-tier realities the design must tolerate: the instance sleeps and takes 30–50s to wake on first request; writes reset to the committed snapshot on every redeploy; a settling deploy returns edge `no-server` 404s.
- Locally the full pipeline runs against the ~650MB IEEE-CIS dataset (gitignored) and produces `results/pipeline_run_summary.json` at ~4,800 txns/sec.

## Capabilities and Constraints

**Capabilities:** three dashboard routes (`/`, `/api/pending`, `/api/resolve/<id>`) plus `/health`; risk briefs carrying `entity_id`, `flagged_type` (`transaction` | `spike`), `model_score`, `top_factors`, `confidence`, `estimated_fp_cost`, `recommended_action`, and an LLM-written `summary_text`; paginated pending listing via `list_pending(limit, offset)`.

**Constraints, all binding on future design work:**

- **Free-tier deploy budget.** No build step, no bundler, no npm, no CDN-hosted framework. The dashboard's import chain must never touch pandas, numpy, xgboost, sklearn, scipy, or `google.genai`.
- **No fabricated content.** Every number, brief, entity, and label rendered in the UI must trace to a committed artifact under `results/` or `data/demo_review_queue.db`. No placeholder metrics, invented entities, or lorem copy — including in mockups and comps.
- **Defense-only vocabulary.** No blocking, holding, cancelling, or payment-execution language in copy or affordances. The reviewer logs a decision; the system never touches money. Enforced by `tests/test_ui.py`.
- **The LLM explains, it does not decide.** `confidence`, `estimated_fp_cost`, and `recommended_action` are computed in Python before the model is called; only `summary_text` comes from Gemini. UI must not imply otherwise.
- **Ground-truth labels never reach the dashboard.** They live only in `results/demo_seed_provenance.json`, so the demo shows a reviewer exactly what a reviewer would see.
- The public surface of `ReviewQueue` is capped at four methods; extend signatures, never add methods.
- SQLite is deliberate and permanent for this repo. Single node, single writer, auditable file.

**Undecided / not established:** no reviewer roles, permissions, multi-user identity, or notification model exist. Queue volume in real use is unknown. There is no mobile-specific usage claim.

## Brand Commitments

- Project name: **Fraud-Spike Detector**; the deployed surface is the **Review Queue** dashboard.
- The **Razorpay / India-market framing is binding** and should stay visible in how the product presents itself: the primary dataset is US card-not-present (IEEE-CIS), the target ecosystem is UPI/IMPS/wallets, and the README states plainly what would have to change for real UPI production. That honesty is part of the pitch, not a caveat to bury.
- Voice throughout the repo is precise, measured, and unhedged: claims are stated as measured, limitations are stated as limitations. Design copy should match — no marketing inflation.
- No logo, wordmark, typeface, or palette has been committed. None is inherited from Razorpay.

## Evidence on Hand

Real, committed, and citable:

- `results/graph_metrics.{json,md}`, `results/baseline_metrics.{json,md}`, `results/phase_comparison.{json,md}` — held-out test metrics (88,581 transactions): AUC-PR 0.6732 vs 0.4681 baseline, precision 0.7106, recall 0.5855, fraud-caught-to-legit-disrupted ratio 2.029 vs 0.792.
- `results/feature_importance_graph.{json,md}`, `results/split_manifest.json` (SHA-256 per split), `results/pipeline_run_summary.json`.
- `data/demo_review_queue.db` — 30 briefs scored from the real held-out split with the real model; 10 spike events, 9 high-scoring singles, 11 borderline. Against ground truth: 9 true positives and 11 real false positives.
- `results/demo_seed_provenance.json` — full provenance for those 30 briefs.
- 70 passing tests; `pip-audit` clean as of 2026-09-04.

**Absent, and must not be invented:** customers, testimonials, press, pricing, adoption or usage numbers, SLAs, deployment claims beyond the single Render instance, and any cross-dataset (PaySim) validation — `generate_synthetic_paysim` is a placeholder generator for test mechanics only and is never a source of reported results.

## Product Principles

1. **Every displayed number traces to a committed file.** If it cannot be cited, it does not ship.
2. **The human decides.** The system's job is to make a decision well-evidenced and fast, never to pre-empt it.
3. **Show the error surface, not the highlight reel.** The demo deliberately includes real false positives, including scores above 0.99 that are legitimate, because that is what a reviewer's queue actually looks like.
4. **Auditability over convenience.** Actions append; nothing is deleted or overwritten, and the record of why must survive the session.
5. **Honesty about scope is a feature.** State what the dataset is, what it is not, and what real UPI deployment would require.

## Accessibility & Inclusion

No product-specific standard has been established. Nothing in the record contradicts targeting WCAG 2.1 AA as a baseline for future work; treat that as an open decision, not a confirmed requirement.
