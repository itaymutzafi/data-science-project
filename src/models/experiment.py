"""Experiment module.

Provides utilities to compare multiple models on a common dataset.
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_regression
from src.features.preprocessing import TimeSeriesScaler
from src.models import baselines
from src.models.lstm import LSTMRegressor

RESULTS_DIR = Path("results")


def _init_model(model_spec: Any) -> Any:
    """Return a fresh model instance from a class, factory, or estimator."""
    if callable(model_spec) and not hasattr(model_spec, "fit"):
        return model_spec()
    try:
        return copy.deepcopy(model_spec)
    except Exception:
        return model_spec


def _train_test_split(df: pd.DataFrame, train_frac: float, val_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-ordered split into train/val/test."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end] if val_frac > 0 else pd.DataFrame()
    test = df.iloc[val_end:]
    return train, val, test


def run_comparison(
    df: pd.DataFrame,
    models_config: Optional[Dict[str, Any]] = None,
    target_col: str = "Target",
    feature_cols: Optional[list] = None,
    save_path: Path | str = RESULTS_DIR / "experiment_results.json",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series], pd.Series]:
    """
    Run model comparison on a prepared dataset.

    Args:
        df: Input DataFrame with features and target.
        models_config: Mapping of model name -> estimator or factory. If None, defaults are used.
        target_col: Target column name.
        feature_cols: Optional list of feature columns; defaults to all numeric except target.
        save_path: Where to persist metrics (JSON).
        train_frac: Fraction of data for training split.
        val_frac: Fraction of data for validation split.

    Returns:
        metrics_df: DataFrame of metrics per model.
        predictions: Dict of model name -> prediction series on test set.
        y_test: Ground-truth target for the test set.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    df_sorted = df.sort_index()
    df_sorted = df_sorted.dropna(subset=[target_col])

    if feature_cols is None:
        numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != target_col]
    if "Ticker" in feature_cols:
        feature_cols.remove("Ticker")

    if not feature_cols:
        raise ValueError("No feature columns available for modeling.")

    train_df, val_df, test_df = _train_test_split(df_sorted, train_frac=train_frac, val_frac=val_frac)

    scaler = TimeSeriesScaler()
    scaler.fit(train_df, columns=feature_cols)
    X_train = scaler.transform(train_df, columns=feature_cols)[feature_cols]
    X_test = scaler.transform(test_df, columns=feature_cols)[feature_cols]
    y_train = train_df[target_col]
    y_test = test_df[target_col]

    default_models = {
        "NaiveBaseline": baselines.NaiveBaseline(strategy="zero"),
        "RandomBaseline": baselines.RandomBaseline(seed=42),
        "CAPMBaseline": baselines.CAPMBaseline(),
        "LSTM": LSTMRegressor(
            input_size=len(feature_cols),
            hidden_size=32,
            num_layers=1,
            dropout=0.1,
            seq_length=20,
            num_epochs=5,
            batch_size=32,
        ),
    }
    models = models_config or default_models

    metrics_store: Dict[str, Dict[str, float]] = {}
    predictions: Dict[str, pd.Series] = {}

    for name, model_spec in models.items():
        model = _init_model(model_spec)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        # Some estimators (e.g., sklearn) return numpy arrays. Convert to a
        # pandas Series aligned with the test index so we can safely call
        # `dropna()` and align by index with `y_test` below.
        if not hasattr(preds, "dropna"):
            preds = pd.Series(np.asarray(preds).ravel(), index=X_test.index if hasattr(X_test, "index") else None)
        preds = preds.dropna()
        aligned_idx = y_test.index.intersection(preds.index)
        if aligned_idx.empty:
            continue
        y_true_aligned = y_test.loc[aligned_idx]
        y_pred_aligned = preds.loc[aligned_idx]
        metrics_store[name] = evaluate_regression(y_true_aligned, y_pred_aligned)
        predictions[name] = y_pred_aligned

    metrics_df = pd.DataFrame(metrics_store).T

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = Path(save_path)
    metrics_df.to_json(save_path, orient="index")

    return metrics_df, predictions, y_test


def train_lstm_with_loss(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[list] = None,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    lstm_params: Optional[Dict[str, Any]] = None,
) -> Tuple[LSTMRegressor, pd.DataFrame, pd.Series, pd.Series]:
    """
    Train an LSTM with scaling and return fitted model plus validation split for analysis.

    Returns:
        model: Fitted LSTMRegressor
        X_val: Scaled validation features
        y_val: Validation targets
        losses: Training losses per epoch
    """
    df_sorted = df.sort_index().dropna(subset=[target_col])

    if feature_cols is None:
        numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != target_col]
    if "Ticker" in feature_cols:
        feature_cols.remove("Ticker")

    train_df, val_df, _ = _train_test_split(df_sorted, train_frac=train_frac, val_frac=val_frac)

    scaler = TimeSeriesScaler()
    scaler.fit(train_df, columns=feature_cols)
    X_train = scaler.transform(train_df, columns=feature_cols)[feature_cols]
    y_train = train_df[target_col]
    X_val = scaler.transform(val_df, columns=feature_cols)[feature_cols]
    y_val = val_df[target_col]

    params = lstm_params or {}
    model = LSTMRegressor(
        input_size=len(feature_cols),
        hidden_size=params.get("hidden_size", 32),
        num_layers=params.get("num_layers", 1),
        dropout=params.get("dropout", 0.1),
        seq_length=params.get("seq_length", 20),
        num_epochs=params.get("num_epochs", 8),
        batch_size=params.get("batch_size", 32),
    )
    model.fit(X_train, y_train)
    return model, X_val, y_val, model.training_losses


def train_lstm_model(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[list] = None,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    lstm_params: Optional[Dict[str, Any]] = None,
) -> Tuple[LSTMRegressor, pd.DataFrame, pd.Series, pd.Series]:
    """
    Train an LSTM with scaling and return the fitted model, validation split, and training losses.
    """
    return train_lstm_with_loss(
        df=df,
        target_col=target_col,
        feature_cols=feature_cols,
        train_frac=train_frac,
        val_frac=val_frac,
        lstm_params=lstm_params,
    )
