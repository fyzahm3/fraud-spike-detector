# Phase 1 (baseline) vs Phase 2 (+graph features) — held-out test

Both models scored the identical held-out test set (sha256 `c2d140a30b2b…`), threshold selected on validation only in both cases.

| metric | Phase 1 baseline | Phase 2 graph | delta |
|---|---|---|---|
| AUC-PR | 0.4681 | 0.6732 | +0.2051 |
| AUC-ROC | 0.8611 | 0.9364 | +0.0753 |
| Precision | 0.5865 | 0.7106 | +0.1242 |
| Recall | 0.3905 | 0.5855 | +0.1949 |
| F1 | 0.4688 | 0.6420 | +0.1732 |
| False positives | 849 | 735 | -114 |
| True positives | 1204 | 1805 | +601 |
| Legitimate value disrupted ($$) | 188,017 | 117,704 | -70,313 |
| Fraud value caught ($$) | 148,964 | 238,782 | +89,818 |
| Caught : disrupted ratio | 0.7923 | 2.0287 | +1.2364 |

## Reading this honestly

- The graph variant adds 12 causal entity-graph features (trailing
  window velocities/degrees + exponentially-decayed neighbor fraud
  rates) computed strictly from pre-transaction history
  (see src/features/graph_features.py and the brute-force leakage test).
- A positive delta means real signal; a ~zero or negative delta would
  mean the raw features already carried that information.
