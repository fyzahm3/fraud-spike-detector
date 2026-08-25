"""Baseline XGBoost training with validation-only hyperparameter selection.

Leakage discipline enforced structurally:
  - this module's functions accept ONLY (train, val) frames — the test split
    is not a parameter anywhere, so it cannot influence model or threshold;
  - the decision threshold is tuned on validation only;
  - all seeds are fixed and histogram building is deterministic, so two runs
    with the same seed produce identical numbers.
"""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

import numpy as np
import xgboost as xgb

from ..features.preprocess import (
    build_features,
    fit_categorical_vocab,
    join_identity,
)

SEED = 42
N_THREADS = 4  # fixed for bit-reproducible runs across invocations

# Small search space; sampled deterministically. Values are conventional GBDT
# ranges — no exotic tuning that couldn't be defended in review.
SEARCH_SPACE = {
    "learning_rate": [0.05, 0.1],
    "max_depth": [6, 8],
    "min_child_weight": [5.0, 50.0],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.7, 0.9],
}
N_CONFIGS = 10
MAX_ROUNDS = 1200
EARLY_STOPPING_ROUNDS = 80


def sample_configs(n: int, seed: int = SEED) -> list[dict]:
    keys = list(SEARCH_SPACE)
    combos = [dict(zip(keys, values))
              for values in itertools.product(*SEARCH_SPACE.values())]
    return random.Random(seed).sample(combos, n)


def base_params(scale_pos_weight: float) -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        # tree_method=hist is deterministic given fixed seed + n_threads
        # (verified bit-identical reruns in tests/test_baseline.py).
        "tree_method": "hist",
        "n_jobs": N_THREADS,
        "seed": SEED,
        "scale_pos_weight": scale_pos_weight,
    }


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """neg/pos from TRAIN counts only — never from val/test."""
    pos = int((y_train == 1).sum())
    neg = int(len(y_train) - pos)
    if pos == 0:
        raise ValueError("training split contains no positives")
    return neg / pos


def train_baseline(train_transactions, train_identity,
                   val_transactions, val_identity,
                   max_rounds: int = MAX_ROUNDS,
                   verbose: bool = True) -> tuple[xgb.XGBClassifier, dict]:
    """Fit on train, select config+threshold on validation, return both."""
    # Join first so the categorical vocabulary covers merged identity fields.
    train_joined = join_identity(train_transactions, train_identity)
    val_joined = join_identity(val_transactions, val_identity)

    # Vocabulary fitted on TRAIN alone; unseen validation/test categories
    # become NaN and flow through XGBoost's learned default direction.
    vocab = fit_categorical_vocab(train_joined)
    X_tr, y_tr = build_features(train_joined)
    X_va, y_va = build_features(val_joined, vocab=vocab)

    spw = compute_scale_pos_weight(y_tr.to_numpy())
    params = base_params(spw)
    cat_cols = [c for c in X_tr.columns
                if str(X_tr[c].dtype) in ("category",)]

    best_model, best_cfg, best_aucpr = None, None, -1.0
    for cfg in sample_configs(N_CONFIGS):
        run_params = {**params, **cfg}
        model = xgb.XGBClassifier(
            n_estimators=max_rounds, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            enable_categorical=True, **run_params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        score = model.best_score
        if verbose:
            print(f"config={cfg} -> val AUC-PR={score:.5f} "
                  f"(best_iter={model.best_iteration})")
        if score > best_aucpr:
            best_model, best_cfg, best_aucpr = model, cfg, score

    val_proba = best_model.predict_proba(X_va)[:, 1]
    from .metrics import classification_metrics, select_threshold_on_validation
    threshold = select_threshold_on_validation(val_proba, y_va.to_numpy())
    val_metrics = classification_metrics(y_va.to_numpy(), val_proba, threshold)

    meta = {
        "model": "xgboost.XGBClassifier (hist)",
        "seed": SEED,
        "n_jobs": N_THREADS,
        "scale_pos_weight": spw,
        "best_config": best_cfg,
        "val_aucpr_at_best_iter": best_aucpr,
        "best_iteration": int(best_model.best_iteration),
        "threshold": threshold,
        "validation_metrics": val_metrics,
        "n_train_rows": int(len(X_tr)),
        "n_val_rows": int(len(X_va)),
        "n_features": int(X_tr.shape[1]),
        "feature_names": list(map(str, X_tr.columns)),
        "xgboost_version": xgb.__version__,
    }
    return best_model, meta


def save_artifacts(model: xgb.XGBClassifier, meta: dict,
                   out_dir: Path | str = "artifacts") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(out_dir / "baseline_model.json")
    (out_dir / "baseline_meta.json").write_text(json.dumps(meta, indent=2))


def load_artifacts(in_dir: Path | str = "artifacts"
                   ) -> tuple[xgb.XGBClassifier, dict]:
    in_dir = Path(in_dir)
    model_path, meta_path = in_dir / "baseline_model.json", in_dir / "baseline_meta.json"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"trained artifacts not found under '{in_dir}' — run train.py first")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    meta = json.loads(meta_path.read_text())
    return model, meta
