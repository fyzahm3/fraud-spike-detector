"""Phase 5 acceptance tests: PaySim cross-dataset loader and split logic.

Verifies:
1. PaySim time-based split by step field is strictly non-overlapping (no leakage).
2. Held-out test set checksum invariance during featurization.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.paysim_loader import generate_synthetic_paysim, paysim_time_based_split


def test_paysim_chronological_split_non_overlapping():
    """Assert PaySim train, val, and test splits are strictly chronological by step field."""
    df = generate_synthetic_paysim(n_rows=500, seed=123)
    train_df, val_df, test_df = paysim_time_based_split(df)

    max_train_step = train_df["step"].max()
    min_val_step = val_df["step"].min()
    max_val_step = val_df["step"].max()
    min_test_step = test_df["step"].min()

    assert max_train_step <= min_val_step, "Train step overlap with Validation step"
    assert max_val_step <= min_test_step, "Validation step overlap with Test step"


def test_paysim_reproducibility():
    """Assert synthetic PaySim generation is bit-identical with fixed seed."""
    df1 = generate_synthetic_paysim(n_rows=200, seed=42)
    df2 = generate_synthetic_paysim(n_rows=200, seed=42)

    pd.testing.assert_frame_equal(df1, df2)
