# Fraud-Spike Detector with Explainable, Human-Gated Escalation

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-48%20passed-success.svg)](tests/)
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
- **Code-Level Verification**: Automated AST test suite (`tests/test_explain.py::test_no_blocking_side_effects`) scans source files to verify no forbidden transaction-modification imports exist.
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

### PaySim Cross-Dataset Validation (Mobile Money P2P Transfers)

To evaluate model transferability to peer-to-peer mobile transfer topologies, we trained and evaluated our baseline pipeline on **PaySim** simulated mobile money logs:

| Dataset / Rail | Primary Entity | Transactions | Held-Out AUC-PR | Precision | Recall | F1 Score |
|---|---|---|---|---|---|---|
| **IEEE-CIS (Phase 2)** | CNP Credit Cards | 88,581 | **0.6732** | 0.7106 | 0.5855 | 0.6420 |
| **PaySim (P2P Mobile)** | Mobile Accounts | 300 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |

*PaySim balance-difference features (`oldbalanceOrg - newbalanceOrig`) and recipient entity type transfer effectively to peer-to-peer transfer topologies.*

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

### 2. Run Test Suite (48/48 Passing)
```bash
pytest
```
*Executes unit tests, chronological time-split leakage checks (`test_features_match_brute_force_past_only`), half-open boundary assertions, LLM schema tests, PaySim time-split tests, UI endpoints, and end-to-end integration tests in ~30s.*

### 3. Execute End-to-End Pipeline & Dashboard
```bash
python run_pipeline.py --variant graph
python app.py
```
*Featurizes 88,581 held-out test transactions, runs XGBoost scoring, extracts multi-transaction spike events, generates LLM risk briefs, enqueues items into `results/review_queue.db`, and outputs `results/pipeline_run_summary.json` (~4,800+ txns/sec).*

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
├── src/
│   ├── data/                     # Loaders, checksum integrity, 70/15/15 time-split
│   ├── features/                 # Preprocessing & 12 causal graph features
│   ├── models/                   # XGBoost training, thresholding & cost metrics
│   ├── spike/                    # Phase 3 SpikeScorer (1h/24h) & SpikeEvent detection
│   └── explain/                  # Phase 4 RiskBrief generator & SQLite ReviewQueue
├── tests/                        # 43 unit, leakage, and integration tests
└── results/                      # Committed metrics, manifests, and run summaries
```

---

## Known Limitations

1. **US Card Data Domain**: Primary evaluation is performed on IEEE-CIS data rather than live UPI logs.
2. **In-Memory Streaming State**: The causal graph builder and `SpikeScorer` maintain entity state in local Python `deque`s. Production at scale would deploy these algorithms on Apache Flink or Kafka Streams.
3. **Static Windowing**: Rolling aggregation windows (1h and 24h) are fixed. Dynamic windowing based on per-merchant volatility profiles is a planned extension.
4. **LLM API Network Latency**: External LLM calls add latency per brief generation; batch runs utilize clean fallback templates when API access is unconfigured.
