# Fraud-Spike Detector with Explainable, Human-Gated Escalation

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-91%20passed-success.svg)](tests/)
[![License](https://img.shields.io/badge/defense--only-strictly%20human--gated-orange.svg)](#defense-only-policy)

An end-to-end fraud-spike detection system for card payments and digital transaction rails. Scores individual transaction risk, aggregates rolling entity velocity and graph signals to detect anomalous **spikes** (bursts of fraudulent activity), and generates auditable, human-readable risk briefs routed to a review queue. **Defense-only by design** — the system flags and explains; human reviewers retain 100% decision authority.

Built for the **Razorpay AI Buildathon 2026** (AI Risk Manager Track).

---

## Architecture

```
                          +-----------------------------------+
                          | IEEE-CIS Raw Dataset (590k txns)  |
                          +-----------------------------------+
                                            |
                                            v
                          +-----------------------------------+
                          |  Chronological 70/15/15 Time-Split|
                          |   (Leakage-Safety Boundary Snap)  |
                          +-----------------------------------+
                                            |
                                            v
                          +-----------------------------------+
                          | Causal Graph & Feature Builder    |
                          | (Card/Device/Email Co-occurrence) |
                          +-----------------------------------+
                                            |
                                            v
                          +-----------------------------------+
                          | XGBoost Classifier (Hist Method)  |
                          |  (Tuned on Val, Scored Test Once) |
                          +-----------------------------------+
                                            |
                                            v
                          +-----------------------------------+
                          |  Phase 3 Rolling SpikeScorer      |
                          | (1h/24h Window & Baseline Ratio)  |
                          +-----------------------------------+
                                            |
                                            v
                          +-----------------------------------+
                          | Phase 4 LLM RiskBrief Explainer   |
                          |   (Gemini Schema-Constrained)     |
                          +-----------------------------------+
                                            |
                                            v
                          +-----------------------------------+
                          |  SQLite Review Queue & Audit Log  |
                          |     (Human Reviewer Workflow)     |
                          +-----------------------------------+
```

---

## Measured Results (Held-Out Test Set)

Evaluated strictly once on the held-out 15% chronological test split (**88,581 transactions**, SHA-256 `c2d140a30b2b…`). All decision thresholds were selected exclusively on the validation split.

### Model Performance Comparison

| Metric | Phase 1 Baseline (Raw Features) | Phase 2 (+ Causal Graph Features) | Delta |
|---|---|---|---|
| **AUC-PR** | **0.4681** | **0.6732** | **+0.2051** |
| **AUC-ROC** | 0.8611 | 0.9364 | +0.0753 |
| **Precision** | 0.5865 | 0.7106 | +0.1242 |
| **Recall** | 0.3905 | 0.5855 | +0.1949 |
| **F1 Score** | 0.4688 | 0.6420 | +0.1732 |
| **False Positives (FP)** | 849 | 735 | -114 |
| **True Positives (TP)** | 1,204 | 1,805 | +601 |
| **Legitimate Value Disrupted** | $188,017.26 | $117,704.20 | -$70,313.06 |
| **Fraud Value Caught** | $148,963.86 | $238,781.62 | +$89,817.76 |
| **Fraud Caught : Legit Disrupted Ratio** | **0.792** | **2.029** | **+1.236** |

*Adding 12 causal entity-graph features (card/device/email co-occurrence degrees, trailing velocity, and exponentially-decayed neighbor risk) increased AUC-PR by **+0.2051** and turned a net-disruptive ratio (0.79) into a positive **2.029x return** in protected currency value.*

---

## India-Market Honesty Section

**Dataset Transparency & Structure:**
The primary dataset used in this project is IEEE-CIS Fraud Detection, which consists of real US e-commerce card-not-present (CNP) transactions. Razorpay's primary operating ecosystem in India is driven heavily by **UPI (Unified Payments Interface)**, IMPS/NEFT, and digital wallets, which differ fundamentally from US credit card rails:

1. **Entity Topology**: CNP transactions rely on `card1`–`card6`, billing address (`addr1`), and email domains. Real UPI fraud detection operates on VPAs (`user@bank`), mobile device IDs, bank handles, and account numbers.
2. **Transaction Velocity & Graph Dynamics**: Card payments experience settlement delays (1–3 days). UPI transactions feature instantaneous 24/7 settlement, enabling ultra-high-velocity multi-hop "mule account" fan-out bursts within seconds.
3. **Typologies**: CNP fraud centers around stolen card details. Indian UPI fraud primarily involves SIM-swap-enabled OTP fraud, fake collect requests, QR-code social engineering, and unauthorized wallet top-ups.

**Adaptations Required for Real UPI Production**:
- Replace card node features with multi-graph VPA-to-VPA directed transfer edges.
- Shorten rolling spike windows from 1h/24h to sub-minute streaming windows (e.g., 10s, 60s, 5m).
- Incorporate device geolocation telemetry and biometric velocity (app open to pin entry duration).

---

## Live Ingestion Proof-of-Concept

A single real payment flows from Razorpay into the review queue. It is deliberately **not scored**, and that is the point of the section.

**What was built.** Three pieces, and nothing else: a *Trigger live transaction* button on the dashboard, a backend route that creates one ₹1 order against Razorpay's **test mode** (`POST /api/live/trigger`), and a webhook endpoint that receives the resulting event (`POST /api/webhook/razorpay`), verifies its HMAC-SHA256 signature over the raw request body against `RAZORPAY_WEBHOOK_SECRET`, and enqueues the payment for human review. `src/ingest/razorpay_live.py` holds the whole feature. Its only outbound call is `POST /v1/orders`; there is no code path in the repository that can refund, capture, cancel, pay out, or subscribe, and `tests/test_live_ingestion.py::test_ingest_package_cannot_move_money` asserts it. A key id that does not begin with `rzp_test_` is refused during credential loading, before any socket is opened — this system must not be *capable* of reaching a live payment environment, not merely configured away from one.

**Why it is not scored by the trained model.** The model was trained on the IEEE-CIS feature space: roughly 430 engineered columns, including the masked `V1`–`V339` block and the causal co-occurrence features that `src/features/graph_features.py` derives from an entity's own transaction history. A Razorpay webhook payload carries an amount, a currency, a method, an order id, and a timestamp. Those two spaces do not overlap.

Defaulting the ~430 absent features to zeros and calling `predict()` *would* return a number. That number would be a property of the padding, not of the payment. This project has already shipped one metric that was an artifact of its data rather than its model — the PaySim label-leakage incident recorded under *Known Limitations* — found it, and deleted it. Producing a second one deliberately, so a demo looks more complete, is not a trade this repository makes.

So the item is stored with `flagged_type="live_demo_unscored"`, distinct from `"transaction"` and `"spike"`, and the API serves `model_score: null` alongside an explicit `scored: false`. The dashboard renders it on its own branch: a dashed **"Live ingestion · not scored"** badge, the words *"Not scored / No model score exists"* where a scored brief shows four decimal places, an *Ingestion note* rather than a *Risk brief*, and a table headed *"Observed payload fields — none is a model feature"* whose every row reads *"Not a model input"* instead of *Increases/Reduces risk*. The distinction is carried by wording and border style, not by colour alone, so it survives greyscale and colour-blindness. The confidence tag and the model-recommendation line are suppressed entirely — there is no scored judgement to be confident about. The queue's headline *Mean risk score* and *Est. false-positive cost* tiles exclude these items, so an unscored row cannot smuggle a number into an aggregate either. The brief text is a fixed constant, never LLM-generated: a model asked to explain the absence of a score may improvise around it, which is precisely the failure being avoided.

**What real-time scoring would actually require.** Deriving the model's feature space from live data — which this proof-of-concept does not attempt. Concretely: a durable per-entity store of prior payments so the 24h/7d co-occurrence and decayed-neighbour-fraud-mass features have history to read; an equivalent of the IEEE-CIS identity join (device, browser, and address signals) available at authorization time; and a replacement for the masked `V1`–`V339` block, which is proprietary engineered signal with no public definition and therefore cannot be reconstructed at all — a production system would train on its own feature space rather than port this one. Until that exists, ingestion and scoring are separate capabilities here, and the queue says so on every affected row.

**Configuration.** `RAZORPAY_KEY_ID`, `RAZORPAY_KEY_SECRET`, and `RAZORPAY_WEBHOOK_SECRET` are read from the environment only; placeholders are in `.env.example` and no value is committed or logged. Left unset, the dashboard button reports the capability as unconfigured and does nothing — the rest of the console is unaffected.

---

## Defense-Only Policy

This system is strictly **defense-only by design**:

- **Zero Blocking Code**: The codebase contains no calls, SDKs, or interfaces capable of blocking, cancelling, holding funds, or executing financial transactions.
- **Code-Level Verification**: Automated tests enforce this rather than policy alone. `tests/test_explain.py::test_no_blocking_side_effects` asserts `ReviewQueue`'s public API is exactly `{enqueue, list_pending, resolve, get_audit_log}` — so it cannot grow an action-capable method unnoticed — and source-scans `src/explain/*.py` for forbidden action terms (`block_card`, `cancel_transaction`, `hold_funds`, `execute_payment`, `chargeback`). `tests/test_ui.py::test_ui_no_blocking_payment_actions` applies the same scan to the dashboard.
- **Human Authority**: The LLM explainer generates structured risk briefs and recommendations (`"hold_for_review"`, `"monitor"`), but **human analysts retain 100% decision authority** via the append-only SQLite review queue.

---

## Quick Start & Reproducibility

### 1. Environment Setup
```bash
# Clone clean repository
git clone https://github.com/fyzahm3/fraud-spike-detector.git
cd fraud-spike-detector

# Create and activate Python 3.12+ virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and insert your GEMINI_API_KEY
```

### 2. Run Test Suite (104/104 Passing)
```bash
pytest
```
*Executes unit tests, chronological time-split leakage checks (`test_features_match_brute_force_past_only`), half-open boundary assertions, LLM schema tests, PaySim split-logic tests, review-queue concurrency and index tests, UI endpoints, dashboard security tests (XSS payload handling, required-action validation, CSRF double-submit, note bounds), deployment surface tests (health check, basic auth, read-only gating, demo-snapshot honesty), live-ingestion tests (webhook signature rejection, replay rejection, test-mode key enforcement, and the assertion that an ingested payment carries no fabricated score), and end-to-end integration tests in ~29s.*

### 3. Execute End-to-End Pipeline & Dashboard
```bash
python run_pipeline.py --variant graph
python app.py
```
*Featurizes 88,581 held-out test transactions, runs XGBoost scoring, extracts multi-transaction spike events, generates LLM risk briefs, enqueues items into `results/review_queue.db`, and outputs `results/pipeline_run_summary.json` (~4,800+ txns/sec).*

---

## Review Queue Dashboard (UI Track)

A reviewer console for fraud analysts: inspect enqueued risk briefs, read the model's contributing factors, and record a decision against each one.

```bash
python app.py --port 5050
```

Open `http://localhost:5050`. The console has two views:

| View | What it shows |
|---|---|
| **Pending review** | Briefs awaiting a decision, ordered by model risk score. Each carries the entity, flagged type, score, confidence, the risk brief, and the top contributing model features with their direction. Decisions are `Dismiss as false positive`, `Escalate`, and `Confirm fraud`, each with an inline reviewer-note field. |
| **Audit trail** | Every decision written to the append-only log — item, entity, score, decision, reviewer note, and timestamp, newest first. Rows are never edited or removed; a correction is a new row. |

### Structure

The interface is plain HTML/CSS/JS served by Flask. No framework, no build step, no npm.

```
templates/index.html        # server-rendered shell
static/css/console.css      # design system (tokens, layout, states)
static/js/console.js        # queue rendering, decisions, view switching
```

`app.py` serves it and exposes `GET /api/pending`, `GET /api/audit`, `POST /api/resolve/<id>`, and `GET /health`.

### Security properties

- **No payment-action code.** No control can block, hold, or cancel a transaction. `tests/test_ui.py::test_ui_no_blocking_payment_actions` scans `app.py`, `templates/`, and `static/` for payment-action terms — the scan follows the code, so extracting the template did not shrink its coverage.
- **Output is escaped structurally.** Queue data reaches the page through `textContent` on constructed DOM nodes; nothing is assigned to `innerHTML`. This matters because `summary_text` is LLM-generated. A test enqueues a brief whose fields carry a script payload and asserts it is never executable markup.
- **A decision is always explicit.** `POST /api/resolve` requires `action` and validates it against the three allowed values. There is no default — a resolution recorded without a human choice would be a decision nobody made, in an append-only log.
- **CSRF protected.** Double-submit token: a `SameSite=Strict` cookie issued with the page, echoed in an `X-CSRF-Token` header that a cross-origin form post cannot set.
- **Reviewer notes are bounded.** 2000 characters, control characters rejected (newline and tab allowed), non-strings rejected.

### Design

The visual direction is recorded in [`DESIGN.md`](DESIGN.md): a light, data-dense financial console taking its palette, spacing, radii, and type scale from **Blade**, Razorpay's open-source design system (MIT). Tokens were read from `packages/blade/src/tokens/global/` and converted from Blade's `hsla()` definitions — real values, not approximations.

Design language only. No Blade dependency (it is a React library; this is server-rendered Flask), and no Razorpay logo, wordmark, or symbol. This is an independent submission to Razorpay, not a product by them.

Measured on the built result: 36 distinct text-on-background pairs across both views, **none below WCAG AA**; placeholder text at 4.91:1.

---

## Live Demo

**URL:** <https://fraud-spike-review-queue.onrender.com>
**Credentials: none needed to look at anything.** The site is public and read-only; the
password applies only to the two routes that change state (recording a reviewer decision,
and creating a test-mode order). Open the link and you are looking at the product.
**Health check:** <https://fraud-spike-review-queue.onrender.com/health>

### The four surfaces

| Route | What it is |
|---|---|
| `/` | Landing page — what the system does, and the headline evidence |
| `/metrics` | Evidence — phase comparison, value ratio, split protocol, methodology, dataset study |
| `/demo` | The review queue itself, read-only for anyone who opens it |
| `/live` | Razorpay test-mode ingestion, and why the ingested item carries no score |

Four real routes in one Flask app with persistent navigation — no framework, no build step,
no tabs faked in JavaScript. Every figure rendered on `/` and `/metrics` is read at request
time from a committed artifact under `results/`;
`tests/test_site.py::test_every_displayed_metric_matches_the_committed_artifact` re-derives
each one from its source file and requires the page to contain that exact string, so a number
cannot drift from its evidence or be typed in by hand.

A small `?` beside major figures opens a written explanation — what AUC-PR is and why accuracy
misleads here, what the value ratio means, why the split is chronological, why the LLM writes
only prose, why the live transaction is unscored. That content is a static map in `app.py`,
rendered by Jinja. There is no model call behind it, deliberately: an improvised explanation
of this project's own evaluation protocol is exactly the text that must not be able to be
wrong, and a test asserts the code path cannot reach an LLM.

Measured locally under gunicorn, against the committed snapshot the hosted instance serves,
**with basic auth switched on** — every read still answers, and only the mutation is refused:

```console
$ DEMO_MODE=1 DEMO_USER=demo DEMO_PASSWORD=localtest \
    gunicorn app:app --bind 127.0.0.1:5099 &

$ curl -s http://127.0.0.1:5099/health
{"auth_enabled":true,"database":"data/demo_review_queue.db","database_reachable":true,
 "demo_mode":true,"pending_items":30,"read_only":false,"status":"ok"}

$ for p in / /metrics /demo /live /api/pending; do
>   curl -s -o /dev/null -w "$p -> %{http_code}\n" http://127.0.0.1:5099$p
> done
/ -> 200
/metrics -> 200
/demo -> 200
/live -> 200
/api/pending -> 200

$ curl -s -o /dev/null -w '%{http_code}\n' -X POST \
>   http://127.0.0.1:5099/api/resolve/1 \
>   -H 'Content-Type: application/json' -d '{"action":"escalated"}'
401
```

`pending_items` is the full queue depth; the deployed instance reports it once the current
commit is redeployed.

### What the hosted instance actually serves

The dashboard is deployed **on its own**, without the training or scoring pipeline. That is a deliberate constraint, not a shortcut: `run_pipeline.py` needs the ~650MB IEEE-CIS dataset plus the trained XGBoost artifacts, which do not fit a free hosting tier — and free tiers wipe the filesystem on every redeploy, so anything generated at boot would not survive anyway.

So the hosted app reads **`data/demo_review_queue.db`**, a small committed SQLite snapshot.

> **This snapshot is a captured demo artifact, not live data — and not synthetic data.**
> Every brief in it was produced by scoring the **real IEEE-CIS held-out test split** with the **real trained model** (`scripts/seed_demo_db.py`). Full provenance for all 30 briefs — model scores, source transaction IDs, and ground-truth labels — is committed at [`results/demo_seed_provenance.json`](results/demo_seed_provenance.json). Nothing in the demo database is fabricated; see [Known Limitations](#known-limitations) for this project's standing position on synthetic results.

### What is in the snapshot, and why

The sample is **stratified rather than top-N by score**, because a demo showing only clean true positives misrepresents the system's real behaviour to a technical audience. As seeded:

| Stratum | Count | Score range | Confidence |
|---|---|---|---|
| Spike events (multi-transaction) | 10 | 0.9993 – 1.0000 | medium |
| High-scoring single transactions | 9 | 0.9066 – 1.0000 | medium |
| Borderline single transactions | 11 | 0.7879 – 0.8741 | low |

Against the held-out ground truth, those 30 briefs contain **9 true positives and 11 real false positives** — including transactions the model scored above 0.99 that are actually legitimate. That error surface is the point: it is what a reviewer's queue genuinely looks like at a threshold of 0.7879.

Ground-truth labels are stored **only** in the provenance file, never in the database, so the dashboard shows a reviewer exactly what a reviewer would see.

Regenerate the snapshot (requires the real dataset and artifacts locally):

```bash
python scripts/seed_demo_db.py --variant graph
```

The script **refuses to run** if the dataset or artifacts are missing rather than inventing plausible-looking briefs.

### Deploy to Render

Roughly a 20-minute path, most of it waiting on the first build.

1. Push this repository to GitHub.
2. On [dashboard.render.com](https://dashboard.render.com), click **New → Blueprint**, connect the repository, and select the branch. Render reads [`render.yaml`](render.yaml) and proposes the `fraud-spike-review-queue` web service — click **Apply**.
3. Render prompts for the two variables marked `sync: false`. Set them:

   | Variable | Value | Notes |
   |---|---|---|
   | `DEMO_USER` | your choice | Basic-auth username |
   | `DEMO_PASSWORD` | a strong random string | **Never commit this.** Placeholders only in `.env.example` |

   `DEMO_MODE=1` is already set in the blueprint and is what points the instance at the committed snapshot.
4. **Expected first boot: 2–4 minutes**, nearly all of it Render provisioning rather than installing. The build installs only [`requirements-web.txt`](requirements-web.txt) — measured in a clean virtualenv, that is **9 packages, 19MB, 4.6 seconds**:

   ```
   blinker  click  Flask==3.1.3  gunicorn==23.0.0  itsdangerous
   Jinja2  MarkupSafe  packaging  Werkzeug
   ```

   The ML stack is excluded and verified absent — the dashboard boots and serves all 30 briefs in a virtualenv where `pandas`, `numpy`, `xgboost`, `sklearn`, `scipy`, and `google.genai` cannot be imported at all. That is what keeps the build inside the free tier's budget. Render polls `/health` and marks the service live once it returns 200.
5. Open the URL, enter the credentials, and paste the URL into the placeholder at the top of this section.

**Free-tier behaviour to expect during a demo:**

- The instance **sleeps when idle**; the first request after a quiet period takes **30–50 seconds** to wake. Load the URL once immediately before recording the pitch video.
- Resolutions made in the demo are real writes to real SQLite, but **reset to the committed snapshot on each redeploy**.
- **Expect `no-server` 404s while a deploy is settling.** During the first deploy on 2026-09-04, roughly 40% of requests returned HTTP 404 with `x-render-routing: no-server` — Render's edge reporting no running backend. It affected every path including `/health`, while the application answered correctly whenever a request reached it. It cleared once the deploy finished: a follow-up probe of 30 requests across `/health`, `/`, and `/api/pending` returned the expected code **30/30**, at ~160ms, with the origin reporting `x-render-origin-server: gunicorn`. So treat `no-server` as "deploy still cycling", not as an application fault — but **don't start recording until a probe comes back clean**:

  ```bash
  for i in $(seq 1 20); do curl -s -o /dev/null -w "%{http_code} " \
    https://fraud-spike-review-queue.onrender.com/health; done; echo
  ```

  Twenty `200`s means the instance is settled. If `no-server` persists well past a deploy, check the Render dashboard's **Logs** and **Events** tabs for restarts or OOM kills.

Railway works the same way via the [`Procfile`](Procfile); set the same environment variables in its dashboard. No Dockerfile is included — neither platform needs one for a Flask app, and it would only add surface area to maintain.

### Configuration reference

| Variable | Default | Effect |
|---|---|---|
| `PORT` | `5050` | Injected by the host; `--port` still wins locally |
| `HOST` | `127.0.0.1` | `0.0.0.0` to expose beyond loopback |
| `DEMO_MODE` | off | Serve `data/demo_review_queue.db` instead of `results/review_queue.db` |
| `REVIEW_DB_PATH` | unset | Explicit database path; overrides `DEMO_MODE` |
| `DEMO_USER` / `DEMO_PASSWORD` | unset | Basic auth on **state-changing requests only** — reading every page and every JSON API is public. **Auth is enabled only when both are set**; absent them mutations are open too, which keeps local development and the test suite unchanged |
| `RAZORPAY_KEY_ID` / `RAZORPAY_KEY_SECRET` | unset | Test-mode Razorpay keys for `/live`. A key id that is not `rzp_test_`-prefixed is refused before any network call |
| `RAZORPAY_WEBHOOK_SECRET` | unset | HMAC-SHA256 signing secret for `POST /api/webhook/razorpay` |
| `READ_ONLY` | off | `POST /api/resolve` returns 403. Kept separate from `DEMO_MODE` so the demo's resolve buttons work on camera |

The gate is drawn on the **HTTP method**, not on a list of public paths, and that direction is
the point: a path allowlist fails open, because adding a route and forgetting to list it leaves
it unprotected. This fails closed — a new route is public only while it is read-only, and
becomes password-protected the moment it accepts a POST.
`tests/test_ui.py::test_auth_gate_is_drawn_on_the_method_so_new_routes_fail_closed` asserts that
property against a path with no route at all.

`GET /health` therefore needs no special case; it is a GET like every other public surface. It
returns 200 with the queue state, or **503** if the database is unreachable, so a boot with a
broken snapshot never takes traffic.

### Cold start

The free instance sleeps after inactivity, and the request that wakes it can take most of a
minute. Two mitigations, in order of how much they matter:

1. **An external uptime monitor pings `/health` every 5 minutes.** This is the actual fix and it
   lives outside this repository — a warm instance never cold-starts for a visitor. `/health` is
   public and cheap, which is what makes it pingable.
2. **The page never spins silently.** Past 2.5 seconds any request shows a labelled "waking the
   server, about 30 seconds" state with a spinner; past 75 seconds it aborts and shows an error
   with a retry control rather than hanging forever.

The honest limit of the second one: once the container is fully asleep it cannot serve the page
either, so the browser shows its own blank tab until the first byte arrives. No in-app code can
change that, which is why the uptime ping is listed first rather than second.

### Run the production server locally

```bash
DEMO_MODE=1 DEMO_USER=demo DEMO_PASSWORD=localtest gunicorn app:app --bind 127.0.0.1:5099
```

### Dependency security scan

`pip-audit` was run against `requirements.txt`, `requirements-web.txt`, and the full installed environment as of 2026-09-04:

```
No known vulnerabilities found
```

No version bumps were required. Flask and `google-genai` were changed from floating (`>=`) to exact pins so the audit result describes what actually deploys.

The web dependency set was additionally **dry-run installed into a clean virtualenv** to confirm the deploy build resolves and boots without the ML stack present (see step 4 above).

---

## Repository Structure

```
.
├── README.md                     # Comprehensive project documentation
├── requirements.txt              # Pinned Python dependencies
├── pytest.ini                    # Test runner configuration
├── .gitignore                    # Secrets, data/raw, and artifact exclusions
├── .env.example                  # Template for required environment variables
├── train.py                      # Training runner with validation-only tuning
├── evaluate.py                   # Single-pass held-out test evaluation script
├── run_pipeline.py               # End-to-end integration pipeline & benchmark script
├── app.py                        # Flask review dashboard (gunicorn-served in production)
├── templates/index.html          # dashboard shell (extracted from app.py)
├── static/css/console.css        # design system — see DESIGN.md
├── static/js/console.js          # queue rendering and decision flow
├── PRODUCT.md                    # durable product context (audience, constraints, evidence)
├── DESIGN.md                     # visual direction, tokens, and recorded design decisions
├── render.yaml                   # Render blueprint for the hosted demo instance
├── Procfile                      # Process definition for Procfile-based platforms
├── requirements-web.txt          # Flask + gunicorn only; dashboard deploy dependencies
├── data/demo_review_queue.db     # Committed snapshot of REAL briefs for the live demo
├── scripts/seed_demo_db.py       # Regenerates that snapshot from a real pipeline run
├── src/
│   ├── data/                     # Loaders, checksum integrity, 70/15/15 time-split
│   ├── features/                 # Preprocessing & 12 causal graph features
│   ├── models/                   # XGBoost training, thresholding & cost metrics
│   ├── spike/                    # Phase 3 SpikeScorer (1h/24h) & SpikeEvent detection
│   ├── explain/                  # Phase 4 RiskBrief generator & SQLite ReviewQueue
│   └── ingest/                   # Razorpay test-mode live ingestion (unscored, ingest-only)
├── tests/                        # 104 unit, leakage, concurrency, deployment, security, UI, live-ingestion, and integration tests
└── results/                      # Committed metrics, manifests, and run summaries
```

---

## Known Limitations

1. **US Card Data Domain**: Primary evaluation is performed on IEEE-CIS data rather than live UPI logs.
2. **In-Memory Streaming State**: The causal graph builder and `SpikeScorer` maintain entity state in local Python `deque`s. Production at scale would deploy these algorithms on Apache Flink or Kafka Streams.
3. **Static Windowing**: Rolling aggregation windows (1h and 24h) are fixed. Dynamic windowing based on per-merchant volatility profiles is a planned extension.
4. **LLM API Network Latency**: External LLM calls add latency per brief generation; batch runs utilize clean fallback templates when API access is unconfigured.
5. **No Cross-Dataset Validation**: Validation on a second payment rail (real mobile-money P2P data such as PaySim) was scoped but not completed within the build window. The repository therefore makes no cross-dataset transferability claim. `src/data/paysim_loader.py` retains only a placeholder generator used to exercise split/loader mechanics in tests — it produces fabricated rows, not real PaySim data, and is never used to produce reported metrics.

---

## Deployment Roadmap (deliberately not built)

Scoped out for the submission window; listed so the omissions are explicit rather than accidental:

- **CDN / static asset caching** — the dashboard is one self-contained HTML response; a CDN would add configuration without measurable benefit at demo scale.
- **Autoscaling & multi-instance** — SQLite is single-writer by design here (see `CLAUDE.md`); horizontal scaling would require migrating the queue to a client/server database, which is a deliberate non-goal for this repository.
- **Managed Postgres** — same reasoning: the auditable single-file queue is the point.
- **CI/CD pipeline** — tests run locally via `pytest`; a GitHub Actions workflow is the obvious next step.
- **Observability stack** — gunicorn access logs to stdout are the whole story today; structured logging, metrics, and alerting would come before any real reviewer traffic.
