# Baseline model — held-out test metrics

Threshold **0.787852** selected on the validation split only (max F1 on a quantile grid); held-out test was scored once.

| split | precision | recall | F1 | AUC-ROC | AUC-PR | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| validation (tuning) | 0.7585 | 0.6144 | 0.6789 | 0.9499 | 0.7165 | 1869 | 595 | 1173 | 84944 |
| **held-out test** | 0.7106 | 0.5855 | 0.6420 | 0.9364 | 0.6732 | 1805 | 735 | 1278 | 84763 |

## False-positive cost (held-out test)

- Flagged transactions that were legitimate: **735**, disrupting **117,704.20** in transaction value
- Fraud correctly caught: **1805**, protecting **238,781.62**
- Fraud missed (false negatives): **1278**, value **230,826.90**
- Ratio fraud-value-caught : legitimate-value-disrupted = **2.029**

_Defense-only reminder: these flags are advisory. Nothing is blocked, cancelled, or held automatically._
