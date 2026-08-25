"""Stable content checksums for DataFrames.

Used as the leakage guard: the held-out test split is checksummed at split
time and re-checked at evaluation time, proving it was not modified anywhere
in between (e.g. accidentally used for training or feature fitting).
"""

from __future__ import annotations

import hashlib

import pandas as pd
from pandas.util import hash_pandas_object


def checksum_dataframe(df: pd.DataFrame) -> str:
    """Order-sensitive SHA-256 of a DataFrame's contents.

    Row order matters on purpose: our time-based split fixes an exact ordering,
    so any reordering/shuffling of the held-out set must be detected, not
    hidden. Deterministic for a given pandas version, column order and dtypes.
    """
    hashes = hash_pandas_object(df, index=True).to_numpy().astype("uint64")
    return hashlib.sha256(hashes.tobytes()).hexdigest()
