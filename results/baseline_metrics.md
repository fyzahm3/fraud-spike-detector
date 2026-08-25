# Baseline model — held-out test metrics

Threshold **0.707069** selected on the validation split only (max F1 on a quantile grid); held-out test was scored once.

| split | precision | recall | F1 | AUC-ROC | AUC-PR | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| validation (tuning) | 0.6675 | 0.4645 | 0.5478 | 0.9002 | 0.5623 | 1413 | 704 | 1629 | 84835 |
| **held-out test** | 0.5865 | 0.3905 | 0.4688 | 0.8611 | 0.4681 | 1204 | 849 | 1879 | 84649 |

## False-positive cost (held-out test)

- Flagged transactions that were legitimate: **849**, disrupting **188,017.26** in transaction value
- Fraud correctly caught: **1204**, protecting **148,963.86**
- Fraud missed (false negatives): **1879**, value **320,644.66**
- Ratio fraud-value-caught : legitimate-value-disrupted = **0.792**

_Defense-only reminder: these flags are advisory. Nothing is blocked, cancelled, or held automatically._
