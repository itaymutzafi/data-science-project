"""Public API for the evaluation package."""

from .analysis import check_stationarity, run_stationarity_analysis
from .metrics import evaluate_regression, CLS_METRICS, REG_METRICS
from .plots import (
    infer_calendar_index,
    plot_stationarity_check,
    plot_walk_forward_validation,
    plot_target_alignment,
    plot_return_distributions,
    plot_correlation_heatmap,
)
from .comparisons import collect_top_results, get_top_results, plot_metrics_by_featureset

__all__ = [
    "check_stationarity",
    "run_stationarity_analysis",
    "evaluate_regression",
    "CLS_METRICS",
    "REG_METRICS",
    "infer_calendar_index",
    "plot_stationarity_check",
    "plot_walk_forward_validation",
    "plot_target_alignment",
    "plot_return_distributions",
    "plot_correlation_heatmap",
    "collect_top_results",
    "get_top_results",
    "plot_metrics_by_featureset"
]
