# PaySim Cross-Dataset Metrics (Mobile Money P2P Transfers)

Evaluated on held-out test split of PaySim simulated mobile-money transfer logs.

| split | precision | recall | F1 | AUC-ROC | AUC-PR | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| validation | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 10 | 0 | 0 | 290 |
| **held-out test** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 12 | 0 | 0 | 288 |

## Cross-Dataset Comparison Note
PaySim represents mobile transfer topologies (CASH-IN, CASH-OUT, TRANSFER). Model balance-difference features transfer effectively to P2P rails.
