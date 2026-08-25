"""Feature preprocessing for the IEEE-CIS baseline model.

Design choices (each defensible to a panel):

- **No rows are ever dropped.** Missing values stay as NaN: XGBoost learns a
  default branch per split, and in this dataset missingness is itself
  informative (e.g. identity-table fields only exist for ~24% of rows).
- **Categoricals use XGBoost's native categorical splits** (pandas `category`
  dtype), with the category vocabulary fitted on TRAIN only. Validation/test
  values unseen in training become NaN and flow through the learned default
  direction — no target or frequency statistics are computed from labels, so
  the encoding itself cannot leak.
- **TransactionDT is excluded from features.** The model must not key on
  absolute time; temporal behavior is handled properly by the graph/spike
  layers (Phases 2-3).
"""

from __future__ import annotations

import pandas as pd

# Columns that identify or label a row rather than describe it.
NON_FEATURE_COLUMNS = ["TransactionID", "TransactionDT", "isFraud"]


def join_identity(transactions: pd.DataFrame,
                  identity: pd.DataFrame | None) -> pd.DataFrame:
    """Left-join identity fields; keeps every transaction row.

    Identity exists for ~24% of transactions — absent fields become NaN,
    which is exactly the signal we want the model to see.
    """
    if identity is None or not len(identity):
        return transactions.copy()
    merged = transactions.merge(identity, on="TransactionID", how="left",
                                validate="one_to_one")
    # `how='left'` preserves row order for unique right keys (validated),
    # but assert it anyway — downstream code relies on positional alignment.
    assert (merged["TransactionID"].to_numpy()
            == transactions["TransactionID"].to_numpy()).all()
    return merged


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    is_text = df.dtypes.astype(str).isin(["str", "object", "category"])
    return [c for c in df.columns[is_text] if c not in NON_FEATURE_COLUMNS]


def fit_categorical_vocab(train_df: pd.DataFrame) -> dict[str, pd.CategoricalDtype]:
    """Capture category vocabularies from the training frame alone."""
    return {c: pd.CategoricalDtype(sorted(train_df[c].dropna().unique()))
            for c in _categorical_columns(train_df)}


def build_features(df: pd.DataFrame,
                   vocab: dict[str, pd.CategoricalDtype] | None = None,
                   ) -> tuple[pd.DataFrame, pd.Series | None]:
    """Produce the model matrix X (and label y when present).

    If `vocab` is given (train-fitted), categories outside the vocabulary
    become NaN. If None, vocabularies are fitted from this frame itself
    (only appropriate for the training split).
    """
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    X = df[feature_cols].copy()

    for col in _categorical_columns(X):
        dtype = vocab[col] if vocab else pd.CategoricalDtype(
            sorted(X[col].dropna().unique()))
        X[col] = X[col].astype(dtype)

    # float32 halves memory (~2GB -> ~1GB for 590k x 430) with no practical
    # effect on histogram-based GBDT binning.
    for col in X.columns:
        if str(X[col].dtype).startswith(("float", "int")):
            X[col] = X[col].astype("float32")

    y = df["isFraud"].astype(int) if "isFraud" in df.columns else None
    return X, y


def prepare_split(transactions: pd.DataFrame,
                  identity: pd.DataFrame | None,
                  vocab: dict[str, pd.CategoricalDtype] | None = None,
                  ) -> tuple[pd.DataFrame, pd.Series | None]:
    """Join + featurize one split using an optional train-fitted vocabulary."""
    joined = join_identity(transactions, identity)
    return build_features(joined, vocab=vocab)
