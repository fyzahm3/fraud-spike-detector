"""Tests for data loading, validation, and the time-based split.

These run against the small committed fixtures in tests/fixtures/ so the
suite is fast and works on a fresh clone without the real dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.checksum import checksum_dataframe
from src.data.loader import load_dataset, load_identity, load_transactions
from src.data.split import build_manifest, time_based_split, write_manifest
from src.data.validate import validate

# Fraction tolerance: boundary "snapping" (to keep identical timestamps in one
# split) can shift split sizes by a few rows on a 600-row fixture.
FRACTION_TOL = 0.03


# ---------------------------------------------------------------- loading --

def test_loader_reads_fixture_tables(fixture_transactions, fixture_identity):
    tx, ident = load_dataset("tests/fixtures", "train")
    pd.testing.assert_frame_equal(tx, fixture_transactions)
    pd.testing.assert_frame_equal(ident, fixture_identity)


def test_loader_missing_data_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download"):
        load_transactions(tmp_path, "train")


# ------------------------------------------------------------- validation --

def test_validate_passes_on_clean_fixtures(fixture_transactions, fixture_identity):
    report = validate(fixture_transactions, fixture_identity)
    assert report["passed"], [c for c in report["checks"] if not c["passed"]]


def test_validate_fails_when_required_column_missing(
        fixture_transactions, fixture_identity, tmp_path):
    broken = fixture_transactions.drop(columns=["addr1"])
    report = validate(broken, fixture_identity)
    assert not report["passed"]
    failed = next(c for c in report["checks"]
                  if c["name"] == "required_transaction_columns")
    assert "addr1" in failed["detail"]


def test_validate_fails_on_non_binary_label(fixture_transactions):
    fixture_transactions.loc[0, "isFraud"] = 2
    assert not validate(fixture_transactions)["passed"]


def test_validate_fails_on_null_label(fixture_transactions):
    fixture_transactions.loc[0, "isFraud"] = np.nan
    assert not validate(fixture_transactions)["passed"]


def test_validate_flags_duplicate_transaction_ids(fixture_transactions):
    duped = pd.concat([fixture_transactions,
                       fixture_transactions.iloc[[0]]], ignore_index=True)
    assert not validate(duped)["passed"]


# ------------------------------------------------------------------ split --

def test_splits_are_chronologically_non_overlapping_and_partition_input(
        fixture_transactions):
    train, val, test = time_based_split(fixture_transactions)
    all_ids = set(fixture_transactions["TransactionID"])

    # Strict temporal separation — the core leakage-safety property.
    assert train["TransactionDT"].max() < val["TransactionDT"].min()
    assert val["TransactionDT"].max() < test["TransactionDT"].min()

    # Exact partition: every row lands in exactly one split.
    split_ids = list(train["TransactionID"]) + list(val["TransactionID"]) \
        + list(test["TransactionID"])
    assert len(split_ids) == len(all_ids) == len(set(split_ids))

    # Fractions are within tolerance of 70/15/15 despite boundary snapping.
    n = len(fixture_transactions)
    assert abs(len(train) / n - 0.70) < FRACTION_TOL
    assert abs(len(val) / n - 0.15) < FRACTION_TOL
    assert abs(len(test) / n - 0.15) < FRACTION_TOL


def test_split_is_deterministic_regardless_of_input_row_order(
        fixture_transactions):
    shuffled = fixture_transactions.sample(frac=1.0, random_state=7)
    t1, v1, s1 = time_based_split(fixture_transactions)
    t2, v2, s2 = time_based_split(shuffled)

    for a, b in ((t1, t2), (v1, v2), (s1, s2)):
        assert set(a["TransactionID"]) == set(b["TransactionID"])
        assert checksum_dataframe(a) == checksum_dataframe(b)


def test_split_snaps_boundary_past_tied_timestamps():
    # 10 rows; naive cut at int(10 * .7) = 7 would land inside the group of
    # rows sharing timestamp 100. The splitter must push the boundary past it.
    df = pd.DataFrame({
        "TransactionID": range(10),
        "TransactionDT": [10, 20, 30, 40, 50, 60, 100, 100, 100, 200],
        "isFraud": [0] * 10,
    })
    train, val, test = time_based_split(df)
    assert train["TransactionDT"].max() < val["TransactionDT"].min()
    assert 100 not in set(train["TransactionDT"])
    assert val["TransactionDT"].max() < test["TransactionDT"].min()


def test_split_rejects_degenerate_single_timestamp_data():
    df = pd.DataFrame({
        "TransactionID": range(10),
        "TransactionDT": [42] * 10,
        "isFraud": [0] * 10,
    })
    with pytest.raises(ValueError, match="timestamp spans"):
        time_based_split(df)


# --------------------------------------------------------------- manifest --

def test_manifest_records_checksums_and_ranges(fixture_transactions):
    train, val, test = time_based_split(fixture_transactions)
    manifest = build_manifest(train, val, test, source_path="tests/fixtures")

    assert set(manifest["splits"]) == {"train", "validation", "test"}
    total_rows = sum(s["n_rows"] for s in manifest["splits"].values())
    assert total_rows == len(fixture_transactions)
    for name, part in (("train", train), ("validation", val), ("test", test)):
        recorded = manifest["splits"][name]
        assert recorded["sha256"] == checksum_dataframe(part)
        assert recorded["n_fraud"] == int(part["isFraud"].sum())
    assert manifest["splits"]["train"]["transaction_dt_max"] \
        < manifest["splits"]["validation"]["transaction_dt_min"]
    assert manifest["splits"]["validation"]["transaction_dt_max"] \
        < manifest["splits"]["test"]["transaction_dt_min"]


def test_write_manifest_creates_json(tmp_path, fixture_transactions):
    train, val, test = time_based_split(fixture_transactions)
    manifest = build_manifest(train, val, test)
    path = tmp_path / "nested" / "split_manifest.json"
    write_manifest(manifest, path)
    assert path.exists()
