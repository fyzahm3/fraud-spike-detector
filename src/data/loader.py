"""Loaders for the IEEE-CIS Fraud Detection dataset.

The dataset is expected under data/raw/ (gitignored) as:
  train_transaction.csv, train_identity.csv, test_transaction.csv, test_identity.csv

See docs/DATA_SETUP.md for how to obtain the files, or run:
  bash scripts/download_data.sh
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path("data/raw")

TRANSACTION_TRAIN_FILE = "train_transaction.csv"
IDENTITY_TRAIN_FILE = "train_identity.csv"
TRANSACTION_TEST_FILE = "test_transaction.csv"
IDENTITY_TEST_FILE = "test_identity.csv"

_MISSING_DATA_MSG = (
    "Dataset files not found in '{path}'.\n"
    "Run 'bash scripts/download_data.sh' (requires Kaggle credentials), or\n"
    "manually download the CSVs from "
    "https://www.kaggle.com/competitions/ieee-fraud-detection/data\n"
    "and place them in that directory. See docs/DATA_SETUP.md."
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(_MISSING_DATA_MSG.format(path=path))
    return pd.read_csv(path)


def load_transactions(data_dir: Path | str = DEFAULT_DATA_DIR,
                      split: str = "train") -> pd.DataFrame:
    """Load the transaction table ('train' or 'test' split)."""
    fname = TRANSACTION_TRAIN_FILE if split == "train" else TRANSACTION_TEST_FILE
    return _read_csv(Path(data_dir) / fname)


def load_identity(data_dir: Path | str = DEFAULT_DATA_DIR,
                  split: str = "train") -> pd.DataFrame:
    """Load the identity table ('train' or 'test' split)."""
    fname = IDENTITY_TRAIN_FILE if split == "train" else IDENTITY_TEST_FILE
    path = Path(data_dir) / fname
    return _read_csv(path) if path.exists() else pd.DataFrame()


def load_dataset(data_dir: Path | str = DEFAULT_DATA_DIR,
                 split: str = "train") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: returns (transactions, identity) for a split."""
    return (
        load_transactions(data_dir, split),
        load_identity(data_dir, split),
    )
