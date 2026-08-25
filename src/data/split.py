"""Time-based train/validation/test split.

Fraud patterns drift over time; a random split lets the model peek at future
behavior of the same cards/merchants and inflates metrics. We therefore split
strictly by TransactionDT:
    earliest 70% -> train, next 15% -> validation, most recent 15% -> test.

Boundaries are "snapped" past any rows sharing a boundary timestamp, so every
timestamp value belongs entirely to one split and max(train.dt) is strictly
less than min(val.dt), etc. This property is asserted by tests and recorded in
a manifest with per-split SHA-256 checksums (see src/data/checksum.py) — the
held-out set's checksum must be identical when it is finally evaluated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .checksum import checksum_dataframe
from .loader import load_transactions

DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_VAL_FRAC = 0.15


def time_based_split(df: pd.DataFrame,
                     train_frac: float = DEFAULT_TRAIN_FRAC,
                     val_frac: float = DEFAULT_VAL_FRAC
                     ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically. Returns (train, val, test)."""
    if not {"TransactionDT", "TransactionID"} <= set(df.columns):
        raise ValueError("df must contain TransactionDT and TransactionID")
    # The held-out test share is the remainder: test_frac = 1 - train - val.
    if not (0 < train_frac < 1 and val_frac >= 0
            and train_frac + val_frac <= 1.0 + 1e-9):
        raise ValueError("require 0 < train_frac < 1 and val_frac >= 0 "
                         "with train_frac + val_frac <= 1")

    # Stable total order: primary key TransactionDT, tie-break on TransactionID
    # so the split is fully deterministic regardless of input row order.
    ordered = df.sort_values(["TransactionDT", "TransactionID"],
                             kind="mergesort").reset_index(drop=True)
    dt = ordered["TransactionDT"].to_numpy()
    n = len(ordered)

    def snap_to_group(i: int) -> int:
        # Move cut position i to the nearest edge of the block of rows sharing
        # its timestamp value, so no timestamp straddles two splits.
        lo = i
        while lo > 0 and dt[lo - 1] == dt[i]:
            lo -= 1
        hi = i
        while hi < n and dt[hi] == dt[i]:
            hi += 1
        return lo if (i - lo) <= (hi - i) else hi

    i_train = snap_to_group(int(n * train_frac))
    i_val = snap_to_group(int(n * (train_frac + val_frac)))

    if not (0 < i_train <= i_val < n):
        raise ValueError(
            "a single timestamp spans more rows than the split allows; "
            "cannot produce strictly separated splits.")

    train = ordered.iloc[:i_train].copy()
    val = ordered.iloc[i_train:i_val].copy()
    test = ordered.iloc[i_val:].copy()
    return train, val, test


def build_manifest(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                   source_path: str | None = None) -> dict:
    """Checksummed summary of each split — the leakage-audit artifact."""
    manifest: dict = {
        "split_rule": "chronological by TransactionDT: first 70% train, "
                      "next 15% validation, last 15% held-out test; boundaries "
                      "snapped past tied timestamps",
        "splits": {},
    }
    if source_path:
        manifest["source"] = str(source_path)
    for name, part in (("train", train), ("validation", val), ("test", test)):
        manifest["splits"][name] = {
            "n_rows": int(len(part)),
            "n_fraud": int(part["isFraud"].sum()) if "isFraud" in part else None,
            "transaction_dt_min": int(part["TransactionDT"].min()),
            "transaction_dt_max": int(part["TransactionDT"].max()),
            "sha256": checksum_dataframe(part),
        }
    return manifest


def write_manifest(manifest: dict, path: Path | str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Time-based split of IEEE-CIS transactions + leakage manifest.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--manifest-out", default="results/split_manifest.json")
    args = parser.parse_args(argv)

    df = load_transactions(args.data_dir, "train")
    train, val, test = time_based_split(df)
    manifest = build_manifest(train, val, test, source_path=args.data_dir)
    write_manifest(manifest, args.manifest_out)

    print(f"Split of {len(df):,} transactions written to {args.manifest_out}")
    for name, info in manifest["splits"].items():
        print(f"  {name:<11}: {info['n_rows']:>7,} rows | fraud={info['n_fraud']:>5,} "
              f"| DT [{info['transaction_dt_min']}, {info['transaction_dt_max']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
