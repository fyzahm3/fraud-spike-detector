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

from ..features.pipeline import build_model_frames

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


def fit_and_select(X_tr, y_tr, X_va, y_va,
                   max_rounds: int = MAX_ROUNDS,
                   verbose: bool = True) -> tuple[xgb.XGBClassifier, dict]:
    """Search configs + threshold on validation. Operates on ready matrices."""
    y_tr = np.asarray(y_tr)
    y_va = np.asarray(y_va)
    spw = compute_scale_pos_weight(y_tr)
    params = base_params(spw)

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
    threshold = select_threshold_on_validation(val_proba, y_va)
    val_metrics = classification_metrics(y_va, val_proba, threshold)

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


def train_baseline(train_transactions, train_identity,
                   val_transactions, val_identity,
                   max_rounds: int = MAX_ROUNDS,
                   verbose: bool = True,
                   use_graph: bool = False) -> tuple[xgb.XGBClassifier, dict]:
    """Featurize splits causally, then fit+select on validation only."""
    frames = build_model_frames(
        train_transactions, train_identity,
        val_transactions, val_identity,
        use_graph=use_graph)
    X_tr, y_tr = frames["train"]
    X_va, y_va = frames["validation"]

    model, meta = fit_and_select(X_tr, y_tr, X_va, y_va,
                                 max_rounds=max_rounds, verbose=verbose)
    meta["variant"] = "graph" if use_graph else "baseline"
    return model, meta


def save_artifacts(model: xgb.XGBClassifier, meta: dict,
                   out_dir: Path | str = "artifacts",
                   name: str = "baseline_model") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(out_dir / f"{name}.json")
    meta["artifact"] = name
    (out_dir / f"{name.replace('_model', '_meta')}.json").write_text(
        json.dumps(meta, indent=2))


def load_artifacts(in_dir: Path | str = "artifacts",
                   name: str = "baseline_model"
                   ) -> tuple[xgb.XGBClassifier, dict]:
    in_dir = Path(in_dir)
    model_path, meta_path = (in_dir / f"{name}.json",
                             in_dir / f"{name.replace('_model', '_meta')}.json")
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"trained artifacts '{name}.*' not found under '{in_dir}' — "
            "run train.py first")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    meta = json.loads(meta_path.read_text())
    return model, meta
