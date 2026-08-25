from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path("tests/fixtures")
TRANSACTION_CSV = FIXTURES / "train_transaction.csv"
IDENTITY_CSV = FIXTURES / "train_identity.csv"


@pytest.fixture()
def fixture_transactions() -> pd.DataFrame:
    return pd.read_csv(TRANSACTION_CSV)


@pytest.fixture()
def fixture_identity() -> pd.DataFrame:
    return pd.read_csv(IDENTITY_CSV)
