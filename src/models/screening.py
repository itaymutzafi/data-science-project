"""Walk-forward screening with per-split scaling and pluggable model definitions."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import evaluate_regression
from src.models import baselines
from src.models.advanced import LSTMRegressor
from src.config import DEF_SPLITS


@dataclass
class ScreeningArtifacts:
    """Container for screening outputs."""

    metrics: pd.DataFrame
    best_model_artifact: object
    last_X_val: pd.DataFrame
    last_y_val: pd.Series
    feature_cols: List[str]
    scaler: Optional[StandardScaler]


def _validate_input(df: pd.DataFrame, target_col: str) -> None:
    """
    Validate that required columns exist before screening.

    Args:
        df: Full dataset containing tickers, features, and target.
        target_col: Name of the target column.

    Raises:
        ValueError: If required columns or expected feature hints are missing.
    """
    required = ["Ticker", target_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    candidate_features = [
        "Log_Return",
        "Sentiment_Score",
        "sentiment_mean_lag1",
        "news_count_lag1",
        "market_sentiment_lag1",
        "sentiment_trend_lag1",
        "RSI",
        "MACD",
        "ATR",
    ]
    if not any(col in df.columns for col in candidate_features):
        raise ValueError(
            "No usable feature columns found (expected technical/sentiment features like "
            "'Log_Return', 'RSI', 'Sentiment_Score')."
        )


def _prepare_feature_matrix(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Extract feature matrix and names from a ticker-specific slice.

    Args:
        df: Single-ticker DataFrame.
        target_col: Name of the target column.

    Returns:
        Tuple of (X, feature_cols).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in {target_col} and c.lower() != "ticker"]
    if not feature_cols:
        raise ValueError("No numeric feature columns available after excluding target/ticker.")
    return df[feature_cols], feature_cols


def get_model_candidates(input_size: int) -> Dict[str, object]:
    """
    Define the model zoo to be evaluated.

    To add a new model, extend the returned dictionary with a new key/value pair.
    """
    from .registry import get_model
    
    models: Dict[str, object] = {
        "NaiveBaseline": get_model("NaiveBaseline", strategy="zero"),
        "Ridge": get_model("Ridge", alpha=1.0),
        "RandomForest": get_model("RandomForest", n_estimators=200, random_state=42, n_jobs=-1, max_depth=None),
        "LSTM": get_model(
            "LSTM",
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            seq_length=20,
            num_epochs=10,
            batch_size=32,
        ),
    }
    return models


def _train_models(X_train: pd.DataFrame, y_train: pd.Series, feature_cols: List[str], preset: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """
    Fit the model zoo on the provided training split.

    Args:
        X_train: Scaled training features.
        y_train: Training targets.
        feature_cols: Names of feature columns (used for LSTM input size).
        preset: Optional pre-instantiated models to fit.

    Returns:
        Mapping of model name to fitted estimator.
    """
    candidates = preset if preset is not None else get_model_candidates(input_size=len(feature_cols))
    fitted: Dict[str, object] = {}
    for name, model in candidates.items():
        model_copy = copy.deepcopy(model)
        model_copy.fit(X_train, y_train)
        fitted[name] = model_copy
    return fitted


def _evaluate_models(
    models: Dict[str, object],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    ticker: str,
    fold_idx: int,
) -> List[Dict[str, float]]:
    """
    Evaluate trained models on the validation split.

    Args:
        models: Mapping of model name to estimator.
        X_val: Scaled validation features.
        y_val: Validation targets.
        ticker: Current ticker symbol.
        fold_idx: Fold index from TimeSeriesSplit.

    Returns:
        List of metric records per model.
    """
    records: List[Dict[str, float]] = []
    for name, model in models.items():
        preds = model.predict(X_val)
        preds_series = (
            pd.Series(np.asarray(preds).ravel(), index=X_val.index)
            if not hasattr(preds, "index")
            else preds
        )
        aligned_idx = y_val.index.intersection(preds_series.index)
        if aligned_idx.empty:
            continue
        metrics = evaluate_regression(
            y_val.loc[aligned_idx],
            preds_series.loc[aligned_idx],
            n_features=X_val.shape[1],
        )
        record = {
            "Ticker": ticker,
            "Model": name,
            "Fold": fold_idx,
        }
        record.update({k: float(v) for k, v in metrics.items()})
        records.append(record)
    return records


def run_screening(
    df: pd.DataFrame | Dict[str, pd.DataFrame],
    tickers: Iterable[str],
    target_col: str = "Target",
    n_splits: int = DEF_SPLITS,
    models: Optional[Dict[str, object]] = None,
    perform_scaling: bool = True,
) -> ScreeningArtifacts:
    """
    Execute walk-forward validation with per-split scaling and multiple models.

    Args:
        df: Full dataset (DataFrame or Dict of DataFrames).
        tickers: Iterable of ticker symbols to evaluate.
        target_col: Name of the target column.
        n_splits: Number of folds for TimeSeriesSplit.
        models: Optional pre-instantiated model dictionary.
        perform_scaling: Whether to fit/transform StandardScaler inside the loop.
                         Set to False if input is already scaled.

    Returns:
        ScreeningArtifacts with aggregated metrics and the final validation artifacts.
    """
    # Validation logic update for Dict
    if isinstance(df, pd.DataFrame):
        _validate_input(df, target_col=target_col)
    
    records: List[Dict[str, float]] = []
    best_model_artifact: object | None = None
    last_X_val: pd.DataFrame | None = None
    last_y_val: pd.Series | None = None
    last_scaler: StandardScaler | None = None
    last_feature_cols: List[str] = []

    for ticker in tickers:
        # Support Dict or DataFrame input
        if isinstance(df, dict):
            if ticker not in df: continue
            df_t = df[ticker].dropna(subset=[target_col]).sort_index()
        else:
            df_t = df[df["Ticker"] == ticker].dropna(subset=[target_col]).sort_index()
            
        if df_t.empty:
            continue

        X_all, feature_cols = _prepare_feature_matrix(df_t, target_col=target_col)
        y_all = df_t[target_col]

        splitter = TimeSeriesSplit(n_splits=n_splits)

        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_all)):
            X_train_raw, X_val_raw = X_all.iloc[train_idx], X_all.iloc[val_idx]
            y_train, y_val = y_all.iloc[train_idx], y_all.iloc[val_idx]

            if X_train_raw.empty or X_val_raw.empty:
                continue

            if perform_scaling:
                scaler = StandardScaler()
                X_train_scaled = pd.DataFrame(
                    scaler.fit_transform(X_train_raw),
                    index=X_train_raw.index,
                    columns=feature_cols,
                )
                X_val_scaled = pd.DataFrame(
                    scaler.transform(X_val_raw),
                    index=X_val_raw.index,
                    columns=feature_cols,
                )
            else:
                scaler = None
                X_train_scaled = X_train_raw
                X_val_scaled = X_val_raw

            trained_models = _train_models(X_train_scaled, y_train, feature_cols, preset=models)
            fold_records = _evaluate_models(
                trained_models,
                X_val_scaled,
                y_val,
                ticker,
                fold_idx,
            )
            records.extend(fold_records)

            # Persist the last fold artifacts for downstream explainability
            best_model_artifact = trained_models.get("LSTM")
            last_X_val = X_val_scaled
            last_y_val = y_val
            last_scaler = scaler
            last_feature_cols = feature_cols

    metrics_df = pd.DataFrame(records)
    if metrics_df.empty:
        raise ValueError("No metrics were produced. Check that the input data has sufficient rows per ticker.")

    leaderboard = (
        metrics_df.groupby(["Ticker", "Model"])[["MSE", "Directional Accuracy", "Strategy Sharpe"]]
        .mean()
        .reset_index()
        .sort_values("Strategy Sharpe", ascending=False)
    )

    artifacts = ScreeningArtifacts(
        metrics=leaderboard,
        best_model_artifact=best_model_artifact,
        last_X_val=last_X_val if last_X_val is not None else pd.DataFrame(),
        last_y_val=last_y_val if last_y_val is not None else pd.Series(dtype=float),
        feature_cols=last_feature_cols,
        scaler=last_scaler,
    )
    return artifacts


def run_walk_forward_screening(
    df: pd.DataFrame | Dict[str, pd.DataFrame],
    tickers: Iterable[str],
    target_col: str = "Target",
    n_splits: int = DEF_SPLITS,
    models: Optional[Dict[str, object]] = None,
    perform_scaling: bool = True,
) -> ScreeningArtifacts:
    """
    Backwards-compatible alias for run_screening.
    """
    return run_screening(
        df=df, 
        tickers=tickers, 
        target_col=target_col, 
        n_splits=n_splits, 
        models=models,
        perform_scaling=perform_scaling
    )


__all__ = ["run_screening", "run_walk_forward_screening", "ScreeningArtifacts"]
