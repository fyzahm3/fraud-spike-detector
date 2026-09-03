# Fraud-Spike Detector with Explainable, Human-Gated Escalation

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-62%20passed-success.svg)](tests/)
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

### 2. Run Test Suite (70/70 Passing)
```bash
pytest
```
*Executes unit tests, chronological time-split leakage checks (`test_features_match_brute_force_past_only`), half-open boundary assertions, LLM schema tests, PaySim split-logic tests, review-queue concurrency and index tests, UI endpoints, deployment surface tests (health check, basic auth, read-only gating, demo-snapshot honesty), and end-to-end integration tests in ~35s.*

### 3. Execute End-to-End Pipeline & Dashboard
```bash
python run_pipeline.py --variant graph
python app.py
```
*Featurizes 88,581 held-out test transactions, runs XGBoost scoring, extracts multi-transaction spike events, generates LLM risk briefs, enqueues items into `results/review_queue.db`, and outputs `results/pipeline_run_summary.json` (~4,800+ txns/sec).*

---

## Review Queue Dashboard (UI Track)

A lightweight human-in-the-loop dashboard (`app.py`) is provided for security analysts to inspect enqueued risk briefs, view LLM explanations and contributing features, and log resolution audit decisions:

```bash
python app.py --port 5050
```

- Open `http://localhost:5050` in a browser.
- Review pending transaction & spike briefs.
- Log decisions (`Confirm True Positive`, `Dismiss False Positive`, `Escalate`) — all actions are append-only written to SQLite audit log with **zero transaction-blocking side effects**.

---

## Live Demo

**URL:** <https://fraud-spike-review-queue.onrender.com>
**Credentials:** shared privately with the judging panel (HTTP basic auth).
**Health check (unauthenticated):** <https://fraud-spike-review-queue.onrender.com/health>

```console
$ curl -s https://fraud-spike-review-queue.onrender.com/health
{"auth_enabled":true,"database":"data/demo_review_queue.db","database_reachable":true,
 "demo_mode":true,"pending_items":1,"read_only":false,"status":"ok"}
```

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
| `DEMO_USER` / `DEMO_PASSWORD` | unset | Basic auth on every route except `/health`. **Auth is enabled only when both are set** — absent them the app runs open, which is what keeps local development and the test suite unchanged |
| `READ_ONLY` | off | `POST /api/resolve` returns 403. Kept separate from `DEMO_MODE` so the demo's resolve buttons work on camera |

`GET /health` is unauthenticated by design — the platform's readiness probe cannot send credentials, and a 401 there would fail the deploy. It returns 200 with the queue state, or **503** if the database is unreachable, so a boot with a broken snapshot never takes traffic.

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
│   └── explain/                  # Phase 4 RiskBrief generator & SQLite ReviewQueue
├── tests/                        # 70 unit, leakage, concurrency, deployment, and integration tests
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
