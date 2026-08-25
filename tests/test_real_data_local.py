"""Local integration check against the REAL dataset in data/raw/.

Skipped automatically on machines/CI clones where data/raw/ is absent
(it is gitignored), so `pytest` stays green on a fresh checkout.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import load_dataset
from src.data.split import build_manifest, time_based_split
from src.data.validate import validate

RAW = Path("data/raw")
pytestmark = pytest.mark.skipif(
    not (RAW / "train_transaction.csv").exists(),
    reason="real dataset not downloaded (run scripts/download_data.sh)",
)


def test_real_data_validates_and_splits_cleanly():
    tx, ident = load_dataset(RAW, "train")
    report = validate(tx, ident)
    assert report["passed"], [c for c in report["checks"] if not c["passed"]]
    # Documented dataset scale (~590k rows).
    assert len(tx) > 500_000

    train, val, test = time_based_split(tx)
    assert train["TransactionDT"].max() < val["TransactionDT"].min()
    assert val["TransactionDT"].max() < test["TransactionDT"].min()
    assert set(train["TransactionID"]).isdisjoint(set(val["TransactionID"]))
    assert set(val["TransactionID"]).isdisjoint(set(test["TransactionID"]))
    assert set(train["TransactionID"]) | set(val["TransactionID"]) \
        | set(test["TransactionID"]) == set(tx["TransactionID"])

    manifest = build_manifest(train, val, test, source_path=str(RAW))
    assert all(s["n_rows"] > 0 for s in manifest["splits"].values())
