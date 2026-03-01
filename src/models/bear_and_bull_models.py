# === Regime-Conditional Training: Bull-Only Robustness Check ===
# Logic:
#   1. Compute target on the FULL (contiguous) time series so shift(-horizon) is correct.
#   2. Filter to Regime_Bull==1 rows AFTER target creation.
#   3. Drop constant / regime columns that are uninformative in the bull-only subset.
#   4. Run walk-forward CV on the filtered bull-only data with a given feature set per ticker.

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone

from src.evaluation.metrics import evaluate_classification
from src.features import experiment_create_target_variable
from src.utils.feature_names import canonicalize_feature_columns
from src.models.registry import get_model
from src import config

# Default classifiers to evaluate
DEFAULT_CLASSIFIERS = ["LogisticRegression", "RandomForestClassifier", "XGBClassifier"]

# Columns that are constant or meaningless inside the bull-only subset
_BULL_ONLY_DROP_COLS = {"Regime_Bull", "Regime_Strength"}

_BULL_RESULTS_FILENAME = "bull_only_cv_results.parquet"


def _get_bull_cache_path() -> Path:
    cache_dir = Path(config.BULL_RESULTS_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / _BULL_RESULTS_FILENAME


def _load_bull_cache() -> pd.DataFrame:
    cache_path = _get_bull_cache_path()
    if not cache_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(cache_path)
        if "Features" in df.columns:
            df["Features"] = df["Features"].apply(json.loads)
        return df
    except Exception as exc:
        print(f"Warning: failed to load bull results cache: {exc}")
        return pd.DataFrame()


def _save_bull_cache(df: pd.DataFrame) -> None:
    cache_path = _get_bull_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_df = df.copy()
    if "Features" in save_df.columns:
        save_df["Features"] = save_df["Features"].apply(json.dumps)
    save_df.to_parquet(cache_path, index=False)


def _prepare_bull_data(
    df: pd.DataFrame,
    target_horizon: int = 1,
    target_type: str = "binary",
    ) -> tuple:
    """
    Create target on the full series, then filter to bull days only.

    Returns
    -------
    bull_df : pd.DataFrame
        Bull-only rows with the target column attached.
    target_col : str
        Name of the generated target column.
    """
    df = canonicalize_feature_columns(df.copy())

    # 1. Create target on FULL contiguous data
    df, target_col = experiment_create_target_variable(
        df,
        horizon=target_horizon,
        target_type=target_type,
    )

    # 2. Filter to bull days
    if "Regime_Bull" not in df.columns:
        return pd.DataFrame(), target_col

    bull_df = df[df["Regime_Bull"] == 1].copy()

    # 3. Drop constant / regime columns
    cols_to_drop = [c for c in _BULL_ONLY_DROP_COLS if c in bull_df.columns]
    if cols_to_drop:
        bull_df = bull_df.drop(columns=cols_to_drop)

    bull_df = bull_df.dropna()
    return bull_df, target_col


def run_bull_only_cv(
    feature_data: Dict[str, pd.DataFrame],
    feature_strategies: Dict[str, Dict[str, List[str]]],
    *,
    target_horizon: int = 1,
    model_names: Optional[List[str]] = None,
    n_splits: int = 5,
    min_fold_size: int = 50,
) -> pd.DataFrame:
    """Train and validate only on bull days across multiple feature strategies and models.

    Parameters
    ----------
    feature_data : dict
        {ticker: full DataFrame} — must still contain Regime_Bull and Close.
    feature_strategies : dict
        {strategy_name: {ticker: [feature_col, ...]}} — one or more named
        feature-set strategies to evaluate (e.g. from Section 6.2).
    target_horizon : int
        Prediction horizon in trading days.
    model_names : list[str] or None
        Registry names of classifiers to evaluate.
        Default: ["LogisticRegression", "RandomForestClassifier", "XGBClassifier"].
    n_splits : int
        Maximum number of TimeSeriesSplit folds.
    min_fold_size : int
        Minimum validation samples per fold; folds smaller than this are skipped.

    Returns
    -------
    pd.DataFrame with columns: Strategy, Model, Ticker, Fold, N_Train, N_Val,
    Features, Accuracy, Precision, Recall, F1.
    """
    if model_names is None:
        model_names = DEFAULT_CLASSIFIERS

    all_results: List[Dict] = []

    for strategy_name, feature_sets in feature_strategies.items():
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy_name}")
        print(f"{'='*60}")

        for ticker, df in feature_data.items():
            if ticker not in feature_sets:
                print(f"  [{ticker}] No feature set for this strategy — skipping.")
                continue

            # --- Correct target creation + bull filtering ---
            bull_df, target_col = _prepare_bull_data(
                df,
                target_horizon=target_horizon,
                target_type="binary",
            )

            if bull_df.empty:
                print(f"  [{ticker}] No bull days after filter (or Regime_Bull missing).")
                continue

            # --- Resolve feature columns ---
            requested_features = feature_sets[ticker]
            available_features = [f for f in requested_features if f in bull_df.columns]
            missing = set(requested_features) - set(available_features)
            if missing:
                print(f"  [{ticker}] Warning: features not found in bull data: {missing}")

            if not available_features:
                print(f"  [{ticker}] No usable features after filtering.")
                continue

            cols = available_features + [target_col]
            data = bull_df[cols].dropna()

            if len(data) < min_fold_size * 2:
                print(f"  [{ticker}] Too few bull samples ({len(data)}) for CV.")
                continue

            X = data[available_features]
            y = data[target_col]

            n_splits_use = min(n_splits, max(2, len(data) // min_fold_size))
            tscv = TimeSeriesSplit(n_splits=n_splits_use)

            # --- Loop over models ---
            for model_name in model_names:
                model = get_model(model_name)
                fold_results: List[Dict] = []

                for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    if len(X_val) < min_fold_size:
                        continue

                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)

                    model_fold = clone(model)
                    model_fold.fit(X_train_scaled, y_train)
                    preds = model_fold.predict(X_val_scaled)

                    preds_series = pd.Series(
                        (preds > 0.5).astype(int) if np.issubdtype(preds.dtype, np.floating) else preds.astype(int),
                        index=X_val.index,
                    )

                    metrics = evaluate_classification(y_val, preds_series)
                    fold_results.append({
                        "Strategy": strategy_name,
                        "Model": model_name,
                        "Ticker": ticker,
                        "Fold": fold,
                        "N_Train": len(X_train),
                        "N_Val": len(X_val),
                        "Features": available_features,
                        **metrics,
                    })

                if fold_results:
                    res_df = pd.DataFrame(fold_results)
                    mean_acc = res_df["Accuracy"].mean()
                    all_results.extend(fold_results)
                    fold_accs = res_df["Accuracy"].tolist()
                    fold_str = ", ".join(f"{a:.4f}" for a in fold_accs)
                    print(
                        f"  [{ticker}] {model_name}: {len(fold_results)} folds | "
                        f"Acc: [{fold_str}] | Mean: {mean_acc:.4f}"
                    )

    if not all_results:
        return pd.DataFrame()
    return pd.DataFrame(all_results)


def summarize_bull_only(results: pd.DataFrame) -> tuple:
    """Build summary tables from bull-only CV results.

    Returns
    -------
    summary : pd.DataFrame
        Mean accuracy per Strategy x Model x Ticker.
    best_per_ticker : pd.DataFrame
        Single best Strategy+Model row per Ticker.
    """
    summary = (
        results
        .groupby(["Strategy", "Model", "Ticker"])["Accuracy"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "MeanAcc", "std": "StdAcc", "count": "Folds"})
        .round(4)
    )

    ticker_mean = (
        results
        .groupby(["Ticker", "Strategy", "Model"])
        .agg(
            MeanAcc=("Accuracy", "mean"),
            StdAcc=("Accuracy", "std"),
            Folds=("Accuracy", "count"),
            Precision=("Precision", "mean"),
            Recall=("Recall", "mean"),
            F1=("F1", "mean"),
        )
        .round(4)
    )
    best_per_ticker = (
        ticker_mean
        .sort_values("MeanAcc", ascending=False)
        .groupby("Ticker")
        .head(1)
        .reset_index()
        .sort_values("Ticker")
    )

    return summary, best_per_ticker

 
def run_bull_only(feature_data: Dict[str, pd.DataFrame], feature_strategies, *, force_refresh: bool = False) -> tuple:
    """Run bull-only CV with all Section 6.2 strategies and default classifiers.

    Parameters
    ----------
    feature_data : dict
        {ticker: full DataFrame} — must still contain Regime_Bull and Close.
    force_refresh : bool
        If False (default), returns cached results when available.
        If True, recomputes from scratch and updates the cache.

    Returns
    -------
    results : pd.DataFrame
        Raw per-fold results (empty DataFrame if nothing produced).
    summary : pd.DataFrame or None
        Mean accuracy per Strategy x Model x Ticker.
    best_per_ticker : pd.DataFrame or None
        Best Strategy+Model row per Ticker.
    """
    if not force_refresh:
        cached = _load_bull_cache()
        if not cached.empty:
            print("Successfully loaded cached bull-only CV results.")
            summary, best_per_ticker = summarize_bull_only(cached)
            return cached, summary, best_per_ticker

    results = run_bull_only_cv(
        feature_data,
        feature_strategies=feature_strategies,
        target_horizon=1,
        n_splits=config.SPLITS,
    )

    if results.empty:
        print("No results produced.")
        return results, None, None

    _save_bull_cache(results)
    summary, best_per_ticker = summarize_bull_only(results)
    return results, summary, best_per_ticker
