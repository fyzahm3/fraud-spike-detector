"""Commit a set of REAL held-out transactions the hosted site can score live.

The website should not merely report what the model scored somewhere else; it
should run the model. The model itself is deployed (artifacts/), but the ~650MB
IEEE-CIS dataset and the sequential graph-feature pipeline that produces a
feature row cannot be — so the feature vectors are computed here, once, from the
real held-out test split, and committed.

What is committed is the model's input, not its output. The score shown on the
site is computed at request time by the real XGBoost booster; the reference
score recorded here exists only so a test can prove the hosted answer matches
the one produced by the full local pipeline.

Two refusals, both deliberate:

- No dataset or no artifacts => MissingSourceDataError. This never falls back to
  generated rows; `generate_synthetic_paysim` output is not a valid source.
- If a reconstructed vector does not reproduce the prediction the original
  DataFrame produced, the script aborts. A committed vector that scores
  differently from the row it claims to be would put a wrong number on the site
  under the label of a real transaction.

Usage:
    python scripts/make_score_samples.py --variant graph
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_dataset
from src.data.split import time_based_split, verify_against_manifest
from src.features.pipeline import align_to_training, build_model_frames

OUT_PATH = Path("data/score_samples.json")

# A spread across the model's behaviour, not a highlight reel: its most
# confident catches, the borderline calls a reviewer actually has to make, real
# false positives, and fraud it missed. A demo that only shows clean hits
# misrepresents the system to the people evaluating it.
N_TOP_FRAUD = 4
N_BORDERLINE = 4
N_FALSE_POSITIVE = 3
N_FALSE_NEGATIVE = 3
N_CLEAR_LEGIT = 3


class MissingSourceDataError(RuntimeError):
    """The real dataset or trained artifacts are unavailable. Never substituted."""


def _require_real_sources(data_dir: Path, artifacts_dir: Path, variant: str) -> None:
    missing = [
        str(data_dir / rel) for rel in ("train_transaction.csv", "train_identity.csv")
        if not (data_dir / rel).exists()
    ]
    missing += [
        str(artifacts_dir / rel) for rel in (f"{variant}_model.json", f"{variant}_meta.json")
        if not (artifacts_dir / rel).exists()
    ]
    if missing:
        raise MissingSourceDataError(
            "Cannot build score samples: the real source data is unavailable.\n"
            "Missing:\n  " + "\n  ".join(missing) + "\n\n"
            "This script will NOT fabricate transactions. Fetch the IEEE-CIS dataset\n"
            "into data/raw/ and run `python train.py`, then re-run."
        )


def _to_numeric_matrix(X: pd.DataFrame) -> np.ndarray:
    """Render the frame as the float matrix XGBoost scores.

    Categorical columns become their integer codes, which is what XGBoost's
    categorical support consumes; pandas uses -1 for missing, and XGBoost's
    missing marker is NaN, so that sentinel is translated rather than passed
    through as a real category.
    """
    out = np.empty((len(X), X.shape[1]), dtype=np.float32)
    for j, col in enumerate(X.columns):
        series = X[col]
        if isinstance(series.dtype, pd.CategoricalDtype):
            codes = series.cat.codes.to_numpy(dtype=np.float32)
            codes[codes < 0] = np.nan
            out[:, j] = codes
        else:
            out[:, j] = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32)
    return out


def _select(proba: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, list[int]]:
    flagged = proba >= threshold
    fraud = y == 1

    order = np.argsort(proba)[::-1]
    top_fraud = [i for i in order if fraud[i]][:N_TOP_FRAUD]
    false_positive = [i for i in order if flagged[i] and not fraud[i]][:N_FALSE_POSITIVE]
    false_negative = [i for i in order if fraud[i] and not flagged[i]][:N_FALSE_NEGATIVE]

    borderline_order = np.argsort(np.abs(proba - threshold))
    borderline = [int(i) for i in borderline_order[:N_BORDERLINE]]

    clear_legit = [i for i in np.argsort(proba) if not fraud[i]][:N_CLEAR_LEGIT]

    return {
        "top_fraud": [int(i) for i in top_fraud],
        "borderline": borderline,
        "false_positive": [int(i) for i in false_positive],
        "false_negative": [int(i) for i in false_negative],
        "clear_legitimate": [int(i) for i in clear_legit],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="graph", choices=["graph", "baseline"])
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--manifest", type=Path, default=Path("results/split_manifest.json"))
    args = parser.parse_args(argv)

    _require_real_sources(args.data_dir, args.artifacts_dir, args.variant)

    with open(args.artifacts_dir / f"{args.variant}_meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    threshold = float(meta["threshold"])
    feature_names = meta["feature_names"]

    print("Loading dataset and verifying split integrity ...")
    transactions, identity = load_dataset(args.data_dir, "train")
    train_tx, val_tx, test_tx = time_based_split(transactions)
    id_train = identity[identity["TransactionID"].isin(train_tx["TransactionID"])]
    id_val = identity[identity["TransactionID"].isin(val_tx["TransactionID"])]
    id_test = identity[identity["TransactionID"].isin(test_tx["TransactionID"])]

    # Same refusal as evaluate.py and seed_demo_db.py: a drifted split
    # invalidates every number downstream of it, this page included.
    verify_against_manifest(
        args.manifest, {"train": train_tx, "validation": val_tx, "test": test_tx}
    )

    print("Building features (sequential, causal) ...")
    frames = build_model_frames(
        train_tx, id_train, val_tx, id_val, test_tx, id_test,
        use_graph=(args.variant == "graph"),
    )
    X_test_raw, y_test = frames["test"]
    X_test = align_to_training(X_test_raw, feature_names)
    y = y_test.to_numpy(dtype=int)

    booster = xgb.Booster()
    booster.load_model(str(args.artifacts_dir / f"{args.variant}_model.json"))

    print("Scoring the held-out split with the real model ...")
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    proba = booster.predict(dtest)

    strata = _select(proba, y, threshold)
    chosen: list[tuple[str, int]] = []
    seen: set[int] = set()
    for name, idxs in strata.items():
        for i in idxs:
            if i not in seen:
                seen.add(i)
                chosen.append((name, i))

    matrix = _to_numeric_matrix(X_test)

    # The guardrail: rebuilding a row from the committed floats must reproduce
    # the prediction the DataFrame produced. Otherwise the committed vector is
    # not the transaction it claims to be.
    rows = matrix[[i for _, i in chosen]]
    d_reconstructed = xgb.DMatrix(
        rows, feature_names=booster.feature_names,
        feature_types=booster.feature_types, enable_categorical=True,
    )
    reconstructed = booster.predict(d_reconstructed)
    original = np.array([proba[i] for _, i in chosen], dtype=np.float64)
    drift = np.abs(reconstructed - original).max()
    print(f"  max |reconstructed - original| = {drift:.3e}")
    if drift > 1e-6:
        raise MissingSourceDataError(
            f"Reconstructed feature vectors drifted from the pipeline's own scoring "
            f"(max {drift:.3e}). Refusing to commit rows that do not score as themselves."
        )

    tx_ids = test_tx["TransactionID"].to_numpy()
    samples = []
    for rank, (stratum, i) in enumerate(chosen, start=1):
        vector = matrix[i]
        samples.append({
            "id": f"txn-{rank:02d}",
            "transaction_id": int(tx_ids[i]),
            "stratum": stratum,
            "amount": float(X_test_raw.iloc[i].get("TransactionAmt", float("nan"))),
            "label": int(y[i]),
            "reference_score": float(proba[i]),
            # NaN is not valid JSON; null round-trips back to NaN on load.
            "features": [None if np.isnan(v) else float(v) for v in vector],
        })

    payload = {
        "variant": args.variant,
        "threshold": threshold,
        "feature_names": feature_names,
        "source": "IEEE-CIS Fraud Detection (data/raw), held-out test split",
        "note": (
            "Real transactions from the held-out test split with the feature vectors "
            "the pipeline produced for them. reference_score is what the real model "
            "returned locally and exists so a test can prove the hosted score matches; "
            "the site computes its own score at request time."
        ),
        "samples": samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)

    print(f"\nwrote {args.out}  ({args.out.stat().st_size/1e6:.2f} MB, {len(samples)} samples)")
    for stratum in strata:
        n = sum(1 for s in samples if s["stratum"] == stratum)
        print(f"  {stratum:<18} {n}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MissingSourceDataError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
