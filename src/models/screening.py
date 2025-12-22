"""Model screening utilities for multi-asset experiments."""

from __future__ import annotations

import copy
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.svm import SVR

from src.features.preprocessing import TimeSeriesScaler
from src.evaluation.metrics import evaluate_regression
from src.models import baselines
from src.models.lstm import LSTMRegressor


TECH_COL_KEYS = [
    "RSI",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "BB_Upper",
    "BB_Lower",
    "BB_Width",
    "ATR",
    "OBV",
    "Log_Return",
]

SENTIMENT_COL_PREFIXES = [
    "sentiment_mean",
    "sentiment_trend",
    "sentiment_momentum",
    "sentiment_volatility",
    "market_sentiment",
    "news_count",
]

def _init_model(model_spec):
    """Return a fresh model instance."""
    try:
        return copy.deepcopy(model_spec)
    except Exception:
        return model_spec

def _select_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Return feature subsets for screening."""
    tech_cols = [c for c in df.columns if any(c.startswith(k) for k in TECH_COL_KEYS)]
    sentiment_cols = [c for c in df.columns if any(c.startswith(p) for p in SENTIMENT_COL_PREFIXES)]
    sets: Dict[str, List[str]] = {"Tech": tech_cols}
    if sentiment_cols:
        sets["Tech+Sentiment"] = sorted(set(tech_cols + sentiment_cols))
    return sets


def _default_models(input_size: int) -> Dict[str, object]:
    """Construct default model zoo for screening."""
    models = {
        "NaiveBaseline": baselines.NaiveBaseline(strategy="zero"),
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "SVR": SVR(kernel="rbf", C=10.0, gamma="scale"),
        "LSTM": LSTMRegressor(
            input_size=input_size,
            hidden_size=32,
            num_layers=1,
            dropout=0.1,
            seq_length=20,
            num_epochs=6,
            batch_size=32,
        ),
    }

    # Optional: XGBoost / LightGBM if available
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    except Exception:
        pass

    return models


def _expanding_window_splits(df: pd.DataFrame, n_splits: int = 3, min_train_size: int | None = None):
    """Yield expanding-window train/test splits to prevent leakage."""
    n = len(df)
    if n_splits < 1 or n < 2:
        return
    test_size = max(1, n // (n_splits + 1))
    start_train = min_train_size or test_size
    for i in range(n_splits):
        train_end = start_train + i * test_size
        test_end = train_end + test_size
        if test_end > n:
            break
        train = df.iloc[:train_end]
        test = df.iloc[train_end:test_end]
        if len(train) == 0 or len(test) == 0:
            continue
        yield train, test


def run_model_screening(
    df: pd.DataFrame,
    tickers: Iterable[str],
    target_col: str = "Target",
    n_splits: int = 3,
    min_train_size: int | None = None,
) -> pd.DataFrame:
    """
    Screen multiple model families across feature subsets and tickers using walk-forward validation.

    Returns a tidy DataFrame with metrics per (Ticker, Feature_Set, Model, Fold).
    """
    records: List[Dict[str, float]] = []
    df_sorted = df.sort_index()

    for ticker in tickers:
        df_t = df_sorted[df_sorted["Ticker"] == ticker].dropna(subset=[target_col])
        feature_sets = _select_feature_sets(df_t)
        if not feature_sets:
            continue

        for subset_name, feature_cols in feature_sets.items():
            if not feature_cols:
                continue
            models_config = _default_models(input_size=len(feature_cols))
            for fold_idx, (train_df, test_df) in enumerate(_expanding_window_splits(df_t, n_splits=n_splits, min_train_size=min_train_size)):
                scaler = TimeSeriesScaler()
                scaler.fit(train_df, columns=feature_cols)
                X_train = scaler.transform(train_df, columns=feature_cols)[feature_cols]
                X_test = scaler.transform(test_df, columns=feature_cols)[feature_cols]
                y_train = train_df[target_col]
                y_test = test_df[target_col]

                for model_name, model_spec in models_config.items():
                    model = _init_model(model_spec)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    if not hasattr(preds, "index"):
                        preds = pd.Series(np.asarray(preds).ravel(), index=X_test.index)
                    preds = preds.dropna()
                    aligned_idx = y_test.index.intersection(preds.index)
                    if aligned_idx.empty:
                        continue
                    metrics = evaluate_regression(y_test.loc[aligned_idx], preds.loc[aligned_idx])
                    rec = {
                        "Ticker": ticker,
                        "Feature_Set": subset_name,
                        "Model": model_name,
                        "Fold": fold_idx,
                    }
                    rec.update({k: float(v) for k, v in metrics.items()})
                    records.append(rec)

    return pd.DataFrame(records)


def run_screening(df: pd.DataFrame, tickers: Iterable[str], target_col: str = "Target") -> pd.DataFrame:
    """Convenience wrapper for default walk-forward screening."""
    return run_model_screening(df=df, tickers=tickers, target_col=target_col, n_splits=3, min_train_size=None)


def run_feature_combinations_experiment(
    df: pd.DataFrame,
    tickers: Iterable[str],
    target_col: str = "Target",
    n_splits: int = 3,
    min_train_size: int | None = None,
) -> pd.DataFrame:
    """
    Train multiple models across feature combinations (Technical, Sentiment, Combined) with walk-forward validation.

    This is intentionally heavier and logs progress for transparency.
    """
    records: List[Dict[str, float]] = []
    df_sorted = df.sort_index().copy()

    # Ensure lag features exist (up to 5)
    if "Log_Return" in df_sorted.columns:
        for k in range(1, 6):
            col = f"Log_Return_Lag_{k}"
            if col not in df_sorted.columns:
                df_sorted[col] = df_sorted["Log_Return"].shift(k)

    feature_sets = {
        "Technical_Only": [
            "RSI",
            "MACD",
            "MACD_Signal",
            "MACD_Hist",
            "BB_Upper",
            "BB_Lower",
            "BB_Width",
            "ATR",
            "OBV",
            "Log_Return_Lag_1",
            "Log_Return_Lag_2",
            "Log_Return_Lag_3",
            "Log_Return_Lag_4",
            "Log_Return_Lag_5",
        ],
        "Sentiment_Only": [
            "Sentiment_Score",
            "sentiment_mean",
            "sentiment_mean_lag1",
        ],
        "Combined": [],  # populated dynamically as union of the above
    }
    feature_sets["Combined"] = sorted(set(feature_sets["Technical_Only"] + feature_sets["Sentiment_Only"]))

    model_names = ["Ridge", "RandomForest", "LSTM"]

    for ticker in tickers:
        df_t = df_sorted[df_sorted["Ticker"] == ticker].copy()
        if df_t.empty or target_col not in df_t.columns:
            continue
        for f_name, candidate_cols in feature_sets.items():
            feature_cols = [c for c in candidate_cols if c in df_t.columns]
            if not feature_cols:
                continue
            df_feat = df_t.dropna(subset=feature_cols + [target_col]).sort_index()
            if df_feat.empty:
                continue
            for model_name in model_names:
                print(f"Training {model_name} on {ticker} using {f_name}...")
                for fold_idx, (train_df, test_df) in enumerate(
                    _expanding_window_splits(df_feat, n_splits=n_splits, min_train_size=min_train_size)
                ):
                    scaler = TimeSeriesScaler()
                    scaler.fit(train_df, columns=feature_cols)
                    X_train = scaler.transform(train_df, columns=feature_cols)[feature_cols]
                    X_test = scaler.transform(test_df, columns=feature_cols)[feature_cols]
                    y_train = train_df[target_col]
                    y_test = test_df[target_col]

                    if model_name == "Ridge":
                        model = Ridge(alpha=1.0)
                    elif model_name == "RandomForest":
                        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
                    elif model_name == "LSTM":
                        model = LSTMRegressor(
                            input_size=len(feature_cols),
                            hidden_size=32,
                            num_layers=1,
                            dropout=0.1,
                            seq_length=20,
                            num_epochs=8,
                            batch_size=32,
                        )
                    else:
                        model = LinearRegression()

                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    if not hasattr(preds, "index"):
                        preds = pd.Series(np.asarray(preds).ravel(), index=X_test.index)
                    preds = preds.dropna()
                    aligned_idx = y_test.index.intersection(preds.index)
                    if aligned_idx.empty:
                        continue
                    metrics = evaluate_regression(y_test.loc[aligned_idx], preds.loc[aligned_idx])
                    rec = {
                        "Ticker": ticker,
                        "Feature_Set": f_name,
                        "Model": model_name,
                        "Fold": fold_idx,
                    }
                    rec.update({k: float(v) for k, v in metrics.items()})
                    records.append(rec)
    return pd.DataFrame(records)


def display_leaderboard(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results and display sorted leaderboard plus sentiment impact plot."""
    if results_df.empty:
        print("No results to display.")
        return results_df
    grouped = (
        results_df.groupby(["Ticker", "Feature_Set", "Model"])[["MSE", "Directional Accuracy", "Strategy Sharpe"]]
        .mean()
        .reset_index()
    )
    leaderboard = grouped.sort_values("Strategy Sharpe", ascending=False)
    print("Top models by Sharpe:")
    print(leaderboard.head(10))

    try:
        from src.evaluation import plots as eval_plots

        # Plot leaderboard heatmap aggregated over models
        leader_pivot = (
            leaderboard.groupby("Model")[["MSE", "Directional Accuracy", "Strategy Sharpe"]]
            .mean()
            .sort_values("Directional Accuracy", ascending=False)
        )
        eval_plots.plot_model_leaderboard(leader_pivot)

        # Sentiment impact: compare Technical vs Combined directional accuracy per ticker (best model each set)
        impact = (
            leaderboard[leaderboard["Feature_Set"].isin(["Technical_Only", "Combined"])]
            .sort_values("Directional Accuracy", ascending=False)
            .groupby(["Ticker", "Feature_Set"])["Directional Accuracy"]
            .first()
            .unstack()
        )
        if not impact.empty:
            impact = impact.rename(columns={"Technical_Only": "Technical", "Combined": "Combined"})
            eval_plots.plot_accuracy_comparison(impact)
    except Exception as e:
        print(f"Could not render plots: {e}")

    return leaderboard
