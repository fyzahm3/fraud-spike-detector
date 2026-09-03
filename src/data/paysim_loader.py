"""PaySim dataset loader and generator (Phase 5 / Checkpoint 8).

PaySim simulates mobile money transactions (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER).
Its step field represents hourly time offsets (1 step = 1 hour).

Used for cross-dataset validation to test how fraud detection models transfer from US CNP cards
to peer-to-peer mobile transfer topologies.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def generate_synthetic_paysim(n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generates a deterministic synthetic PaySim dataset mirroring the official schema."""
    rng = np.random.default_rng(seed)

    # Steps: 1 to 744 (1 month of hours)
    steps = np.sort(rng.integers(1, 745, size=n_rows))
    types = rng.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"], size=n_rows, p=[0.35, 0.20, 0.25, 0.05, 0.15])
    amounts = np.round(rng.exponential(scale=2000.0, size=n_rows) + 5.0, 2)

    # Fraud occurs primarily on TRANSFER and CASH_OUT
    is_fraud = np.zeros(n_rows, dtype=int)
    fraud_candidates = np.where((types == "TRANSFER") | (types == "CASH_OUT"))[0]
    n_fraud = int(n_rows * 0.03)  # ~3% fraud
    if len(fraud_candidates) >= n_fraud:
        fraud_idx = rng.choice(fraud_candidates, size=n_fraud, replace=False)
        is_fraud[fraud_idx] = 1

    orig_balances = np.round(rng.uniform(100.0, 50000.0, size=n_rows), 2)
    new_orig_balances = np.maximum(0.0, orig_balances - amounts * (1 - is_fraud))

    dest_balances = np.round(rng.uniform(0.0, 100000.0, size=n_rows), 2)
    new_dest_balances = dest_balances + amounts

    orig_ids = [f"C{rng.integers(1000000, 9999999)}" for _ in range(n_rows)]
    dest_ids = [
        f"{'M' if t == 'PAYMENT' else 'C'}{rng.integers(1000000, 9999999)}"
        for t in types
    ]

    df = pd.DataFrame({
        "step": steps,
        "type": types,
        "amount": amounts,
        "nameOrig": orig_ids,
        "oldbalanceOrg": orig_balances,
        "newbalanceOrig": new_orig_balances,
        "nameDest": dest_ids,
        "oldbalanceDest": dest_balances,
        "newbalanceDest": new_dest_balances,
        "isFraud": is_fraud,
        "isFlaggedFraud": (amounts > 200000.0).astype(int),
    })

    return df


def paysim_time_based_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split by step: 70% train, 15% validation, 15% test."""
    df_sorted = df.sort_values(by=["step"]).reset_index(drop=True)
    n = len(df_sorted)

    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train_df = df_sorted.iloc[:n_train].copy()
    val_df = df_sorted.iloc[n_train:n_train + n_val].copy()
    test_df = df_sorted.iloc[n_train + n_val:].copy()

    return train_df, val_df, test_df
