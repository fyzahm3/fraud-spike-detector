#!/usr/bin/env python3
"""Seed the committed demo review-queue database (deployment support).

Produces `data/demo_review_queue.db`: a small, committed SQLite snapshot of
real risk briefs so the hosted dashboard has something to show. The hosted
instance cannot run `run_pipeline.py` — that needs the ~650MB IEEE-CIS dataset
and the trained artifacts, neither of which fit a free hosting tier, whose
filesystem is wiped on every redeploy anyway.

HONESTY CONTRACT (see README "No fabricated results"):
  Every brief in the seeded database comes from scoring the REAL held-out test
  split with the REAL trained model. This script refuses to run if the dataset
  or artifacts are missing rather than inventing plausible-looking briefs — a
  previous revision of this project shipped synthetic numbers as real results
  and we are not repeating that. `generate_synthetic_paysim` output is never a
  valid source here.

The sample is stratified, not just the top-N by score, because a demo showing
only clean true positives misrepresents the system. It deliberately includes:
  - high-scoring spike events (the system at its most confident),
  - near-threshold single transactions (low confidence — the borderline band),
  - flagged rows whose ground-truth label is isFraud == 0, i.e. real false
    positives the model actually produced on held-out data.

Ground-truth labels are recorded in `results/demo_seed_provenance.json`, NOT in
the briefs themselves: a reviewer looking at the dashboard sees exactly what a
real reviewer would see, with no label leaking into the UI.

Usage:
    python scripts/seed_demo_db.py [--variant graph|baseline] [--out data/demo_review_queue.db]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

# Invoked as `python scripts/seed_demo_db.py` from the repo root, so the root
# is not on sys.path the way it is for `python -m` or pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_dataset
from src.data.split import time_based_split, verify_against_manifest
from src.explain.queue import ReviewQueue
from src.explain.risk_brief import ContributingFactor, RiskBrief, generate_risk_brief
from src.features.pipeline import align_to_training, build_model_frames
from src.models.train import load_artifacts
from src.spike.spike_scorer import detect_spike_events

# Target queue composition. Kept small so the database is comfortably
# committable (tens of KB) while still showing every branch of the UI.
N_TRANSACTION_BRIEFS = 20
N_SPIKE_BRIEFS = 10
N_TOP_SCORE = 6           # the model at its most confident
N_BORDERLINE = 8          # nearest the threshold: the low-confidence band
N_KNOWN_FALSE_POSITIVE = 6  # flagged, but isFraud == 0 on held-out data


class MissingSourceDataError(RuntimeError):
    """Raised when the real dataset or artifacts are unavailable.

    Deliberately fatal: the alternative is fabricating briefs, which this
    project has an explicit prohibition against.
    """


def _require_real_sources(data_dir: Path, artifacts_dir: Path, variant: str) -> None:
    missing = []
    for rel in ("train_transaction.csv", "train_identity.csv"):
        if not (data_dir / rel).exists():
            missing.append(str(data_dir / rel))
    for rel in (f"{variant}_model.json", f"{variant}_meta.json"):
        if not (artifacts_dir / rel).exists():
            missing.append(str(artifacts_dir / rel))
    if missing:
        raise MissingSourceDataError(
            "Cannot seed the demo database: the real source data is unavailable.\n"
            "Missing:\n  " + "\n  ".join(missing) + "\n\n"
            "This script will NOT fabricate briefs to work around this. Fetch the\n"
            "IEEE-CIS dataset into data/raw/ and run `python train.py` to produce the\n"
            "artifacts, then re-run this script."
        )


def _top_factors_for_row(row_feats: pd.Series, feature_names: list[str], k: int = 3) -> list[ContributingFactor]:
    """The k largest-magnitude numeric features for one scored row."""
    num_vals = row_feats.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    top_idx = np.argsort(np.abs(num_vals))[::-1][:k]
    return [
        ContributingFactor(
            feature=feature_names[i],
            value=float(num_vals[i]),
            direction="increases_risk" if num_vals[i] > 0 else "decreases_risk",
        )
        for i in top_idx
    ]


def select_stratified_indices(
    proba: np.ndarray,
    y_true: np.ndarray | None,
    threshold: float,
) -> tuple[list[int], dict[str, list[int]]]:
    """Pick a representative sample of flagged rows, not just the top scorers.

    Returns (ordered unique indices, {stratum_name: indices}) so the caller can
    record in the provenance file why each row is in the demo.
    """
    flagged = np.where(proba >= threshold)[0]
    if flagged.size == 0:
        raise MissingSourceDataError(
            f"No transactions scored at or above the threshold {threshold:.4f}; nothing to seed."
        )

    by_score_desc = flagged[np.argsort(proba[flagged])[::-1]]
    top_score = list(by_score_desc[:N_TOP_SCORE])

    # Closest to the decision boundary — these land in the low-confidence band
    # and are the calls a human reviewer actually has to make.
    by_distance = flagged[np.argsort(np.abs(proba[flagged] - threshold))]
    borderline = [int(i) for i in by_distance[:N_BORDERLINE]]

    # Real false positives: flagged by the model, labelled legitimate. Spread
    # across the score range rather than clustered, so the demo shows the model
    # being confidently wrong as well as marginally wrong.
    known_fp: list[int] = []
    if y_true is not None:
        fp_pool = flagged[y_true[flagged] == 0]
        if fp_pool.size:
            fp_sorted = fp_pool[np.argsort(proba[fp_pool])[::-1]]
            step = max(1, len(fp_sorted) // N_KNOWN_FALSE_POSITIVE)
            known_fp = [int(i) for i in fp_sorted[::step][:N_KNOWN_FALSE_POSITIVE]]

    strata = {
        "top_score": [int(i) for i in top_score],
        "borderline": borderline,
        "known_false_positive": known_fp,
    }

    # Interleave the strata so the committed queue is not three visually
    # distinct blocks, then cap at the target size.
    ordered: list[int] = []
    for group in zip(*[iter_padded(v, N_TRANSACTION_BRIEFS) for v in strata.values()]):
        for idx in group:
            if idx is not None and idx not in ordered:
                ordered.append(idx)
    selected = ordered[:N_TRANSACTION_BRIEFS]

    # Guarantee the error surface survives the cap — a demo without a real
    # false positive is the exact thing this stratification exists to prevent.
    if known_fp and not any(i in selected for i in known_fp):
        selected[-1] = known_fp[0]

    return selected, strata


def iter_padded(values: list[int], length: int) -> list[int | None]:
    return list(values) + [None] * max(0, length - len(values))


def _template_summary(entity_id: str, flagged_type: str, score: float, detail: str) -> str:
    """Deterministic brief text used when GEMINI_API_KEY is absent.

    run_pipeline.py does the same rather than silently degrading the API path;
    the provenance file records which generator produced the committed rows.
    """
    return (
        f"Entity {entity_id} flagged as a {flagged_type} with model risk score "
        f"{score:.4f}. {detail} Generated deterministically without the LLM "
        f"explainer (no GEMINI_API_KEY at seed time)."
    )


def _confidence_for(score: float, threshold: float) -> str:
    """Mirrors generate_risk_brief's Python-side confidence bands exactly."""
    dist = abs(score - threshold)
    if dist < 0.1:
        return "low"
    if dist < 0.25:
        return "medium"
    return "high"


def _recommended_action_for(score: float, threshold: float, confidence: str) -> str:
    if score >= threshold:
        return "hold_for_review" if confidence in ("high", "medium") else "monitor"
    return "dismiss_low_priority"


def seed(
    data_dir: str = "data/raw",
    artifacts_dir: str = "artifacts",
    manifest: str = "results/split_manifest.json",
    out_db: str = "data/demo_review_queue.db",
    results_dir: str = "results",
    variant: str = "graph",
) -> dict:
    data_path, art_path, out_path = Path(data_dir), Path(artifacts_dir), Path(out_db)
    _require_real_sources(data_path, art_path, variant)

    print("Loading dataset and verifying split integrity ...")
    transactions, identity = load_dataset(data_dir, "train")
    train_tx, val_tx, test_tx = time_based_split(transactions)
    id_train = identity[identity["TransactionID"].isin(train_tx["TransactionID"])]
    id_val = identity[identity["TransactionID"].isin(val_tx["TransactionID"])]
    id_test = identity[identity["TransactionID"].isin(test_tx["TransactionID"])]

    # Same refusal as evaluate.py / run_pipeline.py: a drifted split invalidates
    # every number downstream of it, including anything shown in the demo.
    verify_against_manifest(manifest, {"train": train_tx, "validation": val_tx, "test": test_tx})

    model, meta = load_artifacts(artifacts_dir, name=f"{variant}_model")
    threshold = float(meta["threshold"])

    print(f"Featurizing and scoring {len(test_tx):,} held-out transactions ({variant}) ...")
    t0 = time.perf_counter()
    frames = build_model_frames(
        train_tx, id_train, val_tx, id_val, test_tx, id_test, use_graph=(variant == "graph")
    )
    X_test_raw, y_test = frames["test"]
    X_test = align_to_training(X_test_raw, meta["feature_names"])
    proba = model.predict_proba(X_test)[:, 1]
    print(f"Scored in {time.perf_counter() - t0:.1f}s")

    y_true = None if y_test is None else np.asarray(y_test, dtype=int)
    feature_names = list(X_test.columns)

    selected, strata = select_stratified_indices(proba, y_true, threshold)
    spikes = detect_spike_events(test_tx, proba, threshold=threshold, entity_col="card1")
    print(f"{len(selected)} transaction briefs selected; {len(spikes)} spike events detected")

    # Start from a clean file so the committed snapshot is a pure function of
    # this run rather than an accumulation across runs.
    for sidecar in (out_path, Path(f"{out_path}-wal"), Path(f"{out_path}-shm")):
        sidecar.unlink(missing_ok=True)

    queue = ReviewQueue(db_path=out_path)
    api_key_set = bool(os.getenv("GEMINI_API_KEY"))
    provenance_rows: list[dict] = []

    for idx in selected:
        idx = int(idx)
        row = test_tx.iloc[idx]
        score = float(proba[idx])
        entity_id = str(row.get("card1", "unknown"))
        amount = float(row.get("TransactionAmt", 50.0))
        factors = _top_factors_for_row(X_test.iloc[idx], feature_names)

        labels = [name for name, members in strata.items() if idx in members] or ["flagged"]

        if api_key_set:
            row_dict = row.to_dict()
            row_dict["model_score"] = score
            row_dict["flagged_type"] = "transaction"
            brief = generate_risk_brief(row_dict, factors, cost_estimate=amount, risk_threshold=threshold)
        else:
            confidence = _confidence_for(score, threshold)
            detail = (
                "Score sits close to the decision threshold, so this is a borderline call."
                if confidence == "low"
                else "Score is well clear of the decision threshold."
            )
            brief = RiskBrief(
                entity_id=entity_id,
                flagged_type="transaction",
                model_score=score,
                top_factors=factors,
                confidence=confidence,
                estimated_fp_cost=amount,
                recommended_action=_recommended_action_for(score, threshold, confidence),
                summary_text=_template_summary(entity_id, "transaction", score, detail),
            )

        qid = queue.enqueue(brief)
        provenance_rows.append({
            "queue_id": qid,
            "flagged_type": "transaction",
            "entity_id": entity_id,
            "test_split_row_index": idx,
            "transaction_id": int(row["TransactionID"]),
            "model_score": round(score, 6),
            "strata": labels,
            # The honest bit: what the held-out label actually says. Kept out of
            # the queue row so the dashboard shows no ground truth.
            "ground_truth_is_fraud": None if y_true is None else int(y_true[idx]),
        })

    # Spikes: highest aggregate score first, which is also how the dashboard
    # sorts, so the demo opens on the system's strongest signal.
    for spike in sorted(spikes, key=lambda s: s.aggregate_risk_score, reverse=True)[:N_SPIKE_BRIEFS]:
        factors = [
            ContributingFactor("spike_baseline_ratio_24h", round(float(spike.baseline_ratio), 2), "increases_risk"),
            ContributingFactor("spike_transaction_count", len(spike.transaction_ids), "increases_risk"),
        ]
        score = float(spike.aggregate_risk_score)
        entity_id = str(spike.entity_id)

        if api_key_set:
            brief = generate_risk_brief(spike, factors, cost_estimate=1000.0, risk_threshold=threshold)
        else:
            confidence = _confidence_for(score, threshold)
            brief = RiskBrief(
                entity_id=entity_id,
                flagged_type="spike",
                model_score=score,
                top_factors=factors,
                confidence=confidence,
                estimated_fp_cost=1000.0,
                recommended_action=_recommended_action_for(score, threshold, confidence),
                summary_text=_template_summary(
                    entity_id, "spike", score,
                    f"{len(spike.transaction_ids)} flagged transactions clustered within the "
                    f"detection window at {spike.baseline_ratio:.1f}x this entity's own 24h baseline.",
                ),
            )

        qid = queue.enqueue(brief)
        spike_labels = None
        if y_true is not None:
            member = test_tx["TransactionID"].isin(spike.transaction_ids)
            spike_labels = int(y_true[member.to_numpy()].sum())
        provenance_rows.append({
            "queue_id": qid,
            "flagged_type": "spike",
            "entity_id": entity_id,
            "transaction_ids": [int(t) for t in spike.transaction_ids],
            "model_score": round(score, 6),
            "strata": ["spike_event"],
            "ground_truth_fraud_count_in_spike": spike_labels,
        })

    n_fp = sum(1 for r in provenance_rows if r.get("ground_truth_is_fraud") == 0)
    n_tp = sum(1 for r in provenance_rows if r.get("ground_truth_is_fraud") == 1)

    provenance = {
        "description": (
            "Provenance for the committed demo review-queue snapshot. Every brief was "
            "produced by scoring the real IEEE-CIS held-out test split with the real "
            "trained model. Ground-truth labels live here and not in the database, so "
            "the dashboard shows a reviewer exactly what a reviewer would see."
        ),
        "source_dataset": "IEEE-CIS Fraud Detection (data/raw), held-out test split",
        "split_manifest": manifest,
        "variant": variant,
        "risk_threshold": threshold,
        "test_rows_scored": int(len(test_tx)),
        "total_flagged_transactions": int((proba >= threshold).sum()),
        "total_spike_events_detected": len(spikes),
        "seeded_brief_count": len(provenance_rows),
        "seeded_true_positives": n_tp,
        "seeded_false_positives": n_fp,
        "summary_generator": "gemini" if api_key_set else "deterministic_template",
        "database_path": str(out_path),
        "rows": provenance_rows,
    }

    prov_path = Path(results_dir) / "demo_seed_provenance.json"
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    prov_path.write_text(json.dumps(provenance, indent=2))

    print(f"\nSeeded {len(provenance_rows)} briefs into {out_path} "
          f"({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"  true positives: {n_tp} | false positives: {n_fp} (real held-out labels)")
    print(f"Provenance written to {prov_path}")
    return provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="graph", choices=["baseline", "graph"])
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--manifest", default="results/split_manifest.json")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="data/demo_review_queue.db")
    args = parser.parse_args(argv)

    try:
        seed(
            data_dir=args.data_dir,
            artifacts_dir=args.artifacts_dir,
            manifest=args.manifest,
            out_db=args.out,
            results_dir=args.results_dir,
            variant=args.variant,
        )
    except MissingSourceDataError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
