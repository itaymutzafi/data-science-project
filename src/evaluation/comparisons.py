"""
Visualization module for experiment result comparisons.
"""

from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import COMPANY_COLORS

# Set global style defaults for consistency
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

MODEL_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]
BASELINES = {"NaiveBaseline", "MarketBenchmark", "RandomBaseline", "CAPMBaseline", "ClassificationBaselineMajor", "ClassificationBaselineOne", "ClassificationBaselineZero", "ClassificationBaselineRandom"}


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        f"{c[0]}_{c[1]}" if isinstance(c, tuple) else c
        for c in df.columns
    ]
    df.columns = [col.rstrip("_") for col in df.columns]
    return df


def collect_top_results(
    metrics: Dict[str, bool],
    df: pd.DataFrame,
    *,
    top_n: int = 20,
) -> Dict[str, pd.DataFrame]:
    """Return best per-model feature-set summaries for each metric without printing."""
    if df.empty:
        return {}

    metrics_names = list(metrics.keys())
    required_cols = {"Model", "FeatureSet", "Ticker", *metrics_names}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for top-result summary: {sorted(missing)}")

    best_dfs = {}
    agg = df.groupby(["Model", "FeatureSet", "Ticker"])[metrics_names].agg(["mean", "std"])

    for metric, lower_is_better in metrics.items():
        series = agg[(metric, "mean")].groupby(level=0)
        best_indices = (series.idxmin() if lower_is_better else series.idxmax())
        best_df = (
            agg.loc[best_indices]
            .reset_index()
            .sort_values((metric, "mean"), ascending=lower_is_better)
        )
        best_df = _flatten_columns(best_df).round(4).head(top_n).reset_index(drop=True)
        best_dfs[metric] = best_df

    return best_dfs


def get_top_results(metrics: Dict[str, bool], df: pd.DataFrame):
    best_dfs = collect_top_results(metrics, df, top_n=50)
    if not best_dfs:
        print("No results to display (DataFrame is empty).")
        return

    for metric, df_metric in best_dfs.items():
        print(f"\n=== Top {metric} ===")

        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.expand_frame_repr", False)
        print(df_metric.to_string(index=False))


def plot_metric_by_featureset_scatter(results_df: pd.DataFrame, metric: str, agg_fn: str = "mean",  # or "median"
) -> None:
    if results_df.empty:
        return

    required_cols = {"Ticker", "Model", "FeatureSet", "Fold", metric}
    if not required_cols.issubset(results_df.columns):
        raise ValueError(f"Missing columns: {required_cols - set(results_df.columns)}")

    # --- 1. Aggregate over folds ---
    agg_df = (
        results_df
        .groupby(["Ticker", "Model", "FeatureSet"], as_index=False)
        .agg({metric: agg_fn})
    )

    # --- 2. Marker mapping per model ---
    models = sorted(agg_df["Model"].unique())
    marker_map = {
        model: MODEL_MARKERS[i % len(MODEL_MARKERS)]
        for i, model in enumerate(models)
    }

    plt.figure(figsize=(12, max(6, len(agg_df["FeatureSet"].unique()) * 0.5)))

    # --- 3. Plot points ---
    for model, model_df in agg_df.groupby("Model"):
        for ticker, ticker_df in model_df.groupby("Ticker"):
            color = COMPANY_COLORS.get(ticker, "gray")

            plt.scatter(
                ticker_df[metric],
                ticker_df["FeatureSet"],
                marker=marker_map[model],
                color=color,
                alpha=0.8,
                s=120,
                edgecolors="black",
                linewidths=0.6,
            )

    # --- 4. Axis formatting ---
    plt.xlabel(metric, fontsize=14, weight="bold")
    plt.ylabel("Feature Set", fontsize=14, weight="bold")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # --- 5. Legends (INSIDE the plot) ---
    # Ticker legend (colors)
    ticker_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            label=ticker,
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=10,
        )
        for ticker, color in COMPANY_COLORS.items()
        if ticker in agg_df["Ticker"].unique()
    ]

    # Model legend (markers)
    model_handles = [
        plt.Line2D(
            [0], [0],
            marker=marker_map[model],
            linestyle="None",
            label=model,
            color="black",
            markersize=10,
        )
        for model in models
    ]

    legend1 = plt.legend(
        handles=ticker_handles,
        title="Ticker",
        title_fontsize=13,
        fontsize=11,
        loc="upper left",
        frameon=True,
        framealpha=0.9,
    )
    plt.gca().add_artist(legend1)

    plt.legend(
        handles=model_handles,
        title="Model",
        title_fontsize=13,
        fontsize=11,
        loc="lower left",
        frameon=True,
        framealpha=0.9,
    )

    plt.title(f"{metric} by Feature Set, Ticker and Model", fontsize=16, weight="bold")
    plt.tight_layout()
    # --- Force y-axis to show all FeatureSet values ---
    feature_sets = sorted(agg_df["FeatureSet"].unique())

    plt.yticks(
        ticks=feature_sets,
        labels=feature_sets,
        fontsize=12
    )

    # Display inline without persisting to disk
    plt.show()
    plt.close()  # Free memory immediately


def plot_metrics_by_featureset(metrics: List[str], df: pd.DataFrame):
    if df.empty:
        print("No results to plot.")
        return

    try:
        plt.close('all')
        for metric in metrics:
            filtered_cls = df[~df["Model"].isin(BASELINES)].copy()
            if filtered_cls.empty:
                continue
            plot_metric_by_featureset_scatter(filtered_cls, metric)
            # Clear memory after each plot
            plt.close()
    except Exception as e:
        print(f"Plotting skipped due to error: {e}")
        # Ensure we don't leave lingering plots
        plt.close('all')
