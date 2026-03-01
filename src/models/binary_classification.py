"""Walk-forward helpers for binary classification and feature importance."""

import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import evaluate_classification
from src.config import DEF_SPLITS


def run_binary_cls_with_feature_importance(
    data: pd.DataFrame,
    target_col: str,
    model,
    ticker: str,
    n_splits: int = DEF_SPLITS
):
    """Run walk-forward binary classification and return fold-level metrics."""

    X = data.drop(columns=[target_col])
    y = data[target_col]

    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        X_train_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X.columns)
        X_val_df = pd.DataFrame(X_val_scaled, index=X_val.index, columns=X.columns)

        model_fold = clone(model)
        model_fold.fit(X_train_df, y_train)

        preds = model_fold.predict(X_val_df)
        preds_series = pd.Series(preds, index=X_val.index)
        preds_series = (preds_series > 0.5).astype(int)

        fold_metrics = evaluate_classification(y_val, preds_series)

        all_results.append({
            "Ticker": ticker,
            "Fold": fold,
            **fold_metrics
        })

    return pd.DataFrame(all_results)


def run_binary_cls_embedded_importance(
    data: pd.DataFrame,
    target_col: str,
    model,
    ticker: str,
    n_splits: int = DEF_SPLITS
) -> pd.DataFrame:
    """Compute per-fold embedded feature importance for a binary model."""
    data = data.dropna()
    X = data.drop(columns=[target_col])
    y = data[target_col]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_importances = []

    for fold, (train_idx, _) in enumerate(tscv.split(X)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        X_train_df = pd.DataFrame(
            X_train_scaled,
            index=X_train.index,
            columns=X.columns,
        )

        model_fold = clone(model)
        model_fold.fit(X_train_df, y_train)

        if hasattr(model_fold, "coef_"):
            importance = np.abs(model_fold.coef_).ravel()

        elif hasattr(model_fold, "feature_importances_"):
            importance = model_fold.feature_importances_

        else:
            continue

        fold_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importance,
            "Fold": fold,
            "Ticker": ticker,
        })

        all_importances.append(fold_df)

    if not all_importances:
        return pd.DataFrame(columns=["Feature", "Importance", "Fold", "Ticker"])

    return pd.concat(all_importances, ignore_index=True)
