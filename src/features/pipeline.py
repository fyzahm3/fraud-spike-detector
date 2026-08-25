"""End-to-end featurization: raw split frames -> model matrices.

The graph features are causal, so they must be computed over the FULL
chronological stream (train -> validation -> test concatenated), then sliced
back per split. A validation/test row legitimately sees all history that
predates it — exactly what production would have — while never seeing
anything from its own or later rows (proven by tests/test_graph_features.py).

Test-set LABELS never enter any computation here: GraphFeatureBuilder uses
labels only to accumulate past fraud mass, and for the test slice those
accumulations come exclusively from earlier rows.
"""

from __future__ import annotations

import pandas as pd

from .graph_features import GraphFeatureBuilder
from .preprocess import build_features, fit_categorical_vocab, join_identity


def build_model_frames(train_tx, train_id,
                       val_tx, val_id,
                       test_tx=None, test_id=None,
                       use_graph: bool = True,
                       ) -> dict[str, tuple[pd.DataFrame, pd.Series | None]]:
    """Returns {'train': (X, y), 'validation': (X, y)[, 'test': (X, y)]}.

    Splits are featurized SEQUENTIALLY through one shared stateful
    GraphFeatureBuilder — train first, then validation, then test — so each
    row's graph features reflect exactly the history that predates it and
    peak memory stays at one split instead of the whole stream.
    """
    tx_frames = {"train": train_tx, "validation": val_tx, "test": test_tx}
    id_frames = {"train": train_id, "validation": val_id, "test": test_id}

    builder = GraphFeatureBuilder() if use_graph else None
    vocab = None
    result: dict[str, tuple[pd.DataFrame, pd.Series | None]] = {}
    for name in ("train", "validation", "test"):
        if tx_frames[name] is None:
            continue
        joined = join_identity(tx_frames[name], id_frames[name]).reset_index(drop=True)
        if builder is not None:
            graph = builder.process(joined)
            if len(graph):
                joined = pd.concat([joined, graph], axis=1)
        if name == "train":
            # Category vocabulary from TRAIN alone; unseen values downstream
            # become NaN rather than expanding the vocabulary.
            vocab = fit_categorical_vocab(joined)
        X, y = build_features(joined, vocab=vocab)
        result[name] = (X, y)
    return result


def align_to_training(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Reindex a feature frame to the exact training column order.

    Missing columns become NaN (e.g. identity fields absent at scoring time);
    extra columns are dropped. Loud failure would be better than silent NaNs
    for structural mismatches, but scoring-time sparsity is legitimate.
    """
    return X.reindex(columns=feature_names)
