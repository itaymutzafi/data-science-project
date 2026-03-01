"""Model-zoo orchestration helpers for notebook-facing experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional
import pandas as pd

from src import config as project_config
from src.evaluation import CLS_METRICS, REG_METRICS, collect_top_results
from src.features import (
    audit_features_vs_sets,
    feature_audit_to_frame,
    feature_sets_to_frame,
    generate_diverse_combinations,
    preprocess_day_feature,
    preprocess_month_feature,
)
from src.utils.feature_names import canonicalize_feature_columns
from .experiment import ExperimentConfig, ExperimentRunner

DEFAULT_SEARCH_BUDGET = 1.0

CONTINUOUS_MODEL_ZOO = [
    "Ridge",
    "RandomForest",
    "XGB_Aggressive",
    "LSTM",
    "NaiveBaseline",
    "MarketBenchmark",
    "RandomBaseline",
    "SVR",
    "XGBRegressor",
    "RandomForest_Deep",
    "XGB_Conservative",
]

DISCRETE_MODEL_ZOO = [
    "LogisticRegression",
    "RandomForestClassifier",
    "XGBClassifier",
    "ClassificationBaselineRandom",
    "ClassificationBaselineZero",
    "ClassificationBaselineOne",
    "ClassificationBaselineMajor",
]


@dataclass
class ModelZooInputs:
    """Prepared inputs for model-zoo runs."""

    feature_audit_summary: pd.DataFrame
    feature_sets: Dict[str, Dict[int, List[str]]]
    feature_set_preview: pd.DataFrame


@dataclass
class ModelZooRun:
    """Materialized outputs from a model-zoo branch."""

    results: pd.DataFrame
    top_tables: Dict[str, pd.DataFrame]
    model_summary: pd.DataFrame


def _prepare_model_zoo_data(
    feature_data: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Build modeling-ready frames used consistently by all model-zoo branches.

    The notebook may pass feature_data with raw Day/Month columns. For
    model-zoo runs we always apply cyclic encoding and a final NA drop so
    sampled feature sets and experiment training operate on the same schema.
    """
    prepared: Dict[str, pd.DataFrame] = {}
    for ticker, df_orig in feature_data.items():
        df_tmp = canonicalize_feature_columns(df_orig.copy())
        df_tmp = preprocess_day_feature(preprocess_month_feature(df_tmp))
        prepared[ticker] = df_tmp.dropna()
    return prepared


def _subset_feature_sets_by_budget(
    feature_sets: Dict[str, Dict[int, List[str]]],
    search_budget: float,
) -> Dict[str, Dict[int, List[str]]]:
    """Select a deterministic subset of feature sets according to search budget."""
    if not (0 < search_budget <= 1):
        raise ValueError("search_budget must be in the interval (0, 1].")

    if search_budget == 1:
        return feature_sets

    subset: Dict[str, Dict[int, List[str]]] = {}
    for ticker, sets_by_ticker in feature_sets.items():
        ordered_ids = sorted(sets_by_ticker.keys())
        keep_n = max(1, int(round(len(ordered_ids) * search_budget)))
        keep_ids = ordered_ids[:keep_n]
        subset[ticker] = {sid: sets_by_ticker[sid] for sid in keep_ids}

    return subset


def prepare_model_zoo_inputs(
    feature_data: Dict[str, pd.DataFrame],
    *,
    n_feature_sets: int = 20,
    random_state: Optional[int] = None,
) -> ModelZooInputs:
    """Build feature audit and sampled feature-set space for model-zoo experiments."""
    prepared_data = _prepare_model_zoo_data(feature_data)
    feature_audit = audit_features_vs_sets(prepared_data)
    feature_audit_summary = feature_audit_to_frame(feature_audit)
    feature_sets = generate_diverse_combinations(
        prepared_data,
        n=n_feature_sets,
        random_state=random_state,
        verbose=False,
    )
    feature_set_preview = feature_sets_to_frame(feature_sets)
    return ModelZooInputs(
        feature_audit_summary=feature_audit_summary,
        feature_sets=feature_sets,
        feature_set_preview=feature_set_preview,
    )


def build_model_zoo_config(
    feature_sets: Dict[str, Dict[int, List[str]]],
    *,
    target_type: str,
    target_horizon: int,
    models: List[str],
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    n_splits: Optional[int] = None,
    show_progress: bool = False,
    log_missing_features: bool = False,
) -> ExperimentConfig:
    """Create a standardized experiment config for model-zoo runs."""
    resolved_tickers = tickers or list(feature_sets.keys()) or project_config.TICKERS
    return ExperimentConfig(
        tickers=resolved_tickers,
        feature_sets=feature_sets,
        models=models,
        target_type=target_type,
        target_horizon=target_horizon,
        start_date=start_date or project_config.START_DATE,
        end_date=end_date or project_config.END_DATE,
        n_splits=n_splits or project_config.SPLITS,
        show_progress=show_progress,
        log_missing_features=log_missing_features,
    )


def _summarize_model_performance(
    results: pd.DataFrame,
    metrics_map: Mapping[str, bool],
) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=list(metrics_map.keys()))

    metric_cols = list(metrics_map.keys())
    summary = results.groupby("Model")[metric_cols].mean()
    primary_metric = metric_cols[0]
    summary = summary.sort_values(primary_metric, ascending=metrics_map[primary_metric])
    return summary.round(4)


def run_model_zoo(
    feature_data: Dict[str, pd.DataFrame],
    config: ExperimentConfig,
    metrics_map: Mapping[str, bool],
    *,
    save_path: Optional[str] = None,
    top_n: int = 12,
    search_budget: float = DEFAULT_SEARCH_BUDGET,
) -> ModelZooRun:
    """Run one model-zoo branch and return compact artifacts for reporting.

    Args:
        search_budget: Fraction of sampled feature-sets to evaluate per ticker.
            Must be in (0, 1]. Lower values reduce runtime linearly.
    """
    prepared_data = _prepare_model_zoo_data(feature_data)
    feature_sets_subset = _subset_feature_sets_by_budget(config.feature_sets, search_budget)
    scoped_config = replace(config, feature_sets=feature_sets_subset)

    runner = ExperimentRunner(prepared_data, scoped_config)
    runner.run()
    results = runner.get_results_df()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(path, index=False)

    top_tables = collect_top_results(dict(metrics_map), results, top_n=top_n)
    model_summary = _summarize_model_performance(results, metrics_map)
    return ModelZooRun(results=results, top_tables=top_tables, model_summary=model_summary)


def run_continuous_model_zoo(
    feature_data: Dict[str, pd.DataFrame],
    feature_sets: Dict[str, Dict[int, List[str]]],
    *,
    models: Optional[List[str]] = None,
    target_horizon: int = 2,
    save_path: Optional[str] = None,
    top_n: int = 12,
    show_progress: bool = False,
    log_missing_features: bool = False,
    search_budget: float = DEFAULT_SEARCH_BUDGET,
) -> ModelZooRun:
    """Run the continuous-target model-zoo branch."""
    config = build_model_zoo_config(
        feature_sets,
        target_type="continuous",
        target_horizon=target_horizon,
        models=models or CONTINUOUS_MODEL_ZOO,
        show_progress=show_progress,
        log_missing_features=log_missing_features,
    )
    return run_model_zoo(
        feature_data,
        config,
        REG_METRICS,
        save_path=save_path,
        top_n=top_n,
        search_budget=search_budget,
    )


def run_discrete_model_zoo(
    feature_data: Dict[str, pd.DataFrame],
    feature_sets: Dict[str, Dict[int, List[str]]],
    *,
    models: Optional[List[str]] = None,
    target_horizon: int = 1,
    save_path: Optional[str] = None,
    top_n: int = 12,
    show_progress: bool = False,
    log_missing_features: bool = False,
    search_budget: float = DEFAULT_SEARCH_BUDGET,
) -> ModelZooRun:
    """Run the discrete-target model-zoo branch."""
    config = build_model_zoo_config(
        feature_sets,
        target_type="binary",
        target_horizon=target_horizon,
        models=models or DISCRETE_MODEL_ZOO,
        show_progress=show_progress,
        log_missing_features=log_missing_features,
    )
    return run_model_zoo(
        feature_data,
        config,
        CLS_METRICS,
        save_path=save_path,
        top_n=top_n,
        search_budget=search_budget,
    )


def get_top_metric_table(
    top_tables: Dict[str, pd.DataFrame],
    metric_name: str,
) -> pd.DataFrame:
    """Safe metric-table accessor for notebook display cells."""
    table = top_tables.get(metric_name)
    if table is not None and not table.empty:
        return table
    return pd.DataFrame({"Notice": [f"No rows available for {metric_name}."]})
