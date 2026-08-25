"""Generates deterministic synthetic fixture CSVs under tests/fixtures/.

These mimic the IEEE-CIS schema (same required columns, plausible value
ranges, some missing values) so the test suite runs in seconds without the
650 MB real dataset. A deliberate fraud "spike" (two cards with clustered
fraudulent transactions) is baked in for later spike-detection tests.

Re-run to regenerate:  python scripts/make_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 600
OUT_DIR = Path("tests/fixtures")

EMAILS = ["gmail.com", "outlook.com", "yahoo.com", "protonmail.com", None]
CARDS = ["visa", "mastercard", "american express", "discover", None]


def make_transactions(rng: np.random.Generator) -> pd.DataFrame:
    # One row per hour over ~60 days; shuffled on disk so tests prove that the
    # splitter sorts correctly rather than relying on pre-sorted input.
    dt = np.arange(3600, 3600 * (N_ROWS + 1), step=3600)
    rng.shuffle(dt)

    df = pd.DataFrame({
        "TransactionID": np.arange(1, N_ROWS + 1),
        "TransactionDT": dt,
        "isFraud": (rng.random(N_ROWS) < 0.02).astype(int),
        "TransactionAmt": np.round(rng.gamma(2.0, 40.0, N_ROWS) + 0.5, 2),
        "card1": rng.choice(np.arange(1000, 1040), N_ROWS),
        "card2": rng.choice([100.0, 111.0, 200.0, np.nan], N_ROWS),
        "card3": rng.choice([150.0, 185.0], N_ROWS),
        "card4": rng.choice(CARDS, N_ROWS),
        "card5": rng.choice([126.0, 132.0, 166.0, np.nan], N_ROWS),
        "card6": rng.choice(["debit", "credit", "charge card"], N_ROWS),
        "addr1": rng.choice([100.0, 200.0, 300.0, np.nan], N_ROWS),
        "addr2": rng.choice([87.0, 60.0], N_ROWS),
        "dist1": rng.choice([0.0, 5.0, 23.0, np.nan], N_ROWS),
        "dist2": rng.choice([1.0, np.nan, np.nan, np.nan], N_ROWS),
        "P_emaildomain": rng.choice(EMAILS, N_ROWS),
        "R_emaildomain": rng.choice(EMAILS, N_ROWS),
        "C1": rng.integers(1, 50, N_ROWS).astype(float),
        "C2": rng.integers(1, 30, N_ROWS).astype(float),
        "D1": rng.integers(1, 120, N_ROWS).astype(float),
        "M1": rng.choice(["T", "F"], N_ROWS),
        "M2": rng.choice(["T", "F", np.nan], N_ROWS),
        "M3": rng.choice(["T", "F", np.nan], N_ROWS),
        "V1": np.round(rng.normal(0, 1, N_ROWS), 3),
        "V2": np.round(rng.normal(0, 1, N_ROWS), 3),
        "V3": np.round(rng.normal(0, 1, N_ROWS), 3),
    })

    # Inject a fraud spike: two cards each get a tight burst of fraudulent
    # transactions inside a one-hour window near the end of the timeline.
    burst_start = int(df["TransactionDT"].max()) - 3600 * 24
    for card_id, offset in ((1007, 0), (1021, 1800)):
        idx = rng.choice(N_ROWS - 15, 12, replace=False)
        df.loc[idx, "card1"] = card_id
        df.loc[idx, "TransactionDT"] = burst_start + offset + np.arange(12) * 150
        df.loc[idx, "isFraud"] = 1
        df.loc[idx, "TransactionAmt"] = np.round(
            rng.uniform(300, 900, 12), 2)

    return df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def make_identity(transactions: pd.DataFrame,
                  rng: np.random.Generator) -> pd.DataFrame:
    ids = transactions["TransactionID"].to_numpy()
    n = len(ids) // 2
    chosen = rng.choice(ids, n, replace=False)
    return pd.DataFrame({
        "TransactionID": np.sort(chosen),
        "id_01": np.round(rng.uniform(0, 30, n), 1),
        "DeviceType": rng.choice(["desktop", "mobile", np.nan], n),
        "DeviceInfo": rng.choice(["Windows", "iOS Device", "Linux", np.nan, np.nan], n),
    })


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    tx = make_transactions(rng)
    ident = make_identity(tx, rng)
    tx.to_csv(OUT_DIR / "train_transaction.csv", index=False)
    ident.to_csv(OUT_DIR / "train_identity.csv", index=False)
    print(f"Wrote {len(tx)} transaction rows and {len(ident)} identity rows "
          f"to {OUT_DIR}/ (seed={SEED})")


if __name__ == "__main__":
    main()
