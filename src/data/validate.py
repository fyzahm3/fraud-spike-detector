"""Data quality validation for the IEEE-CIS Fraud Detection dataset.

Run as a CLI:
    python -m src.data.validate --data-dir data/raw
or import `validate()` / `print_report()` from tests and pipelines.

Exit code is non-zero if any required check fails, so this can gate CI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .loader import load_dataset

# Core columns the whole pipeline depends on. The dataset has hundreds of
# optional masked features (V1-V339 etc.) which are reported but not required.
REQUIRED_TRANSACTION_COLUMNS = [
    "TransactionID", "TransactionDT", "isFraud", "TransactionAmt",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1", "dist2",
    "P_emaildomain", "R_emaildomain",
]
KNOWN_OPTIONAL_PREFIXES = ("C", "D", "V", "M", "id_")

REQUIRED_IDENTITY_COLUMNS = ["TransactionID", "DeviceType", "DeviceInfo"]


def validate(transactions: pd.DataFrame,
             identity: pd.DataFrame | None = None) -> dict:
    """Run all checks; returns a report dict with an overall pass/fail."""
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    # 1. Expected columns present.
    missing = [c for c in REQUIRED_TRANSACTION_COLUMNS if c not in transactions.columns]
    check("required_transaction_columns",
          not missing,
          f"missing={missing}" if missing else f"all {len(REQUIRED_TRANSACTION_COLUMNS)} present")

    # 2. Label is binary and never null (silent NaN labels would poison metrics).
    label_ok = False
    if "isFraud" in transactions.columns:
        labels = transactions["isFraud"]
        n_null = int(labels.isna().sum())
        uniq = set(pd.unique(labels.dropna()).tolist())
        label_ok = (n_null == 0 and uniq <= {0, 1})
        check("label_binary_no_nulls", label_ok,
              f"unique_values={sorted(uniq)}, nulls={n_null}")
    else:
        check("label_binary_no_nulls", False, "column missing")

    # 3. No fully-null columns.
    if len(transactions):
        null_cols = [c for c in transactions.columns if transactions[c].notna().sum() == 0]
        check("no_fully_null_columns", not null_cols,
              f"fully_null_columns={null_cols[:10]}{'...' if len(null_cols) > 10 else ''}")
    else:
        check("no_fully_null_columns", False, "empty dataframe")

    # 4. TransactionDT numeric and non-negative (time-based split depends on it).
    dt_ok = False
    if "TransactionDT" in transactions.columns:
        dt = pd.to_numeric(transactions["TransactionDT"], errors="coerce")
        dt_ok = dt.notna().all() and (dt >= 0).all()
        check("transaction_dt_numeric_nonneg", dt_ok,
              f"min={dt.min()}, max={dt.max()}, non_numeric_or_null={int(dt.isna().sum())}")
    else:
        check("transaction_dt_numeric_nonneg", False, "column missing")

    # 5. TransactionID unique in both tables (join key integrity).
    tid_ok = "TransactionID" in transactions.columns and transactions["TransactionID"].is_unique
    check("transaction_id_unique", tid_ok,
          f"duplicates={int(transactions['TransactionID'].duplicated().sum()) if 'TransactionID' in transactions.columns else 'n/a'}")

    report = {
        "n_transaction_rows": int(len(transactions)),
        "n_transaction_cols": int(transactions.shape[1]),
        "fraud_rate": (float(transactions["isFraud"].mean())
                       if "isFraud" in transactions.columns and len(transactions) else None),
        "checks": checks,
    }

    if identity is not None and len(identity):
        miss_id = [c for c in REQUIRED_IDENTITY_COLUMNS if c not in identity.columns]
        check("required_identity_columns", not miss_id,
              f"missing={miss_id}" if miss_id else f"all {len(REQUIRED_IDENTITY_COLUMNS)} present")
        id_subset = identity["TransactionID"].isin(transactions["TransactionID"]).all() \
            if "TransactionID" in identity.columns else False
        check("identity_ids_subset_of_transactions", id_subset,
              f"{len(identity)} identity rows, orphan ids="
              f"{int((~identity['TransactionID'].isin(transactions['TransactionID'])).sum()) if 'TransactionID' in identity.columns else 'n/a'}")
        report["n_identity_rows"] = int(len(identity))

    report["passed"] = all(c["passed"] for c in checks)
    return report


def print_report(report: dict) -> None:
    print("=" * 64)
    print("DATA QUALITY REPORT")
    print("=" * 64)
    print(f"transactions : {report['n_transaction_rows']:,} rows x {report['n_transaction_cols']} cols")
    if report.get("n_identity_rows") is not None:
        print(f"identity     : {report['n_identity_rows']:,} rows")
    if report.get("fraud_rate") is not None:
        print(f"fraud rate   : {report['fraud_rate']:.4%}")
    print("-" * 64)
    for c in report["checks"]:
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"[{mark}] {c['name']}: {c['detail']}")
    print("-" * 64)
    print(f"OVERALL: {'PASS' if report['passed'] else 'FAIL'}")
    print("=" * 64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate IEEE-CIS raw data.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--split", default="train", choices=["train", "test"],
                        help="which split's files to validate")
    args = parser.parse_args(argv)

    try:
        transactions, identity = load_dataset(args.data_dir, args.split)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 2

    report = validate(transactions, identity)
    print_report(report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
