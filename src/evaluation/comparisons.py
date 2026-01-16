"""
Visualization module for experiment result comparisons.

This module provides functions to generate academic-quality plots using Seaborn and Matplotlib.
It supports Leaderboards, Risk-Return Scatter plots, and Performance Heatmaps.
"""

import os
from typing import Optional, List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import COMPANY_COLORS

# Set global style defaults for consistency
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

MODEL_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]
BASELINES = {"NaiveBaseline", "MarketBenchmark", "RandomBaseline", "CAPMBaseline", "ClassificationBaselineMajor", "ClassificationBaselineOne", "ClassificationBaselineZero", "ClassificationBaselineRandom"}


def plot_leaderboard(
    results_df: pd.DataFrame, 
    metric: str, 
    title: str, 
    filepath: Optional[str] = None
) -> None:
    """Plots a bar chart of the given metric for each Model/FeatureSet combination.

    Args:
        results_df (pd.DataFrame): Experiment results containing 'Model', 'FeatureSet', and metric.
        metric (str): The column name of the metric to plot (e.g., 'Strategy Sharpe').
        title (str): Title of the plot.
        filepath (Optional[str]): Path to save the figure (inclusive of filename).
    """
    if results_df.empty:
        return

    plt.figure(figsize=(12, 6))
    
    # Aggregating across folds/tickers if raw results provided
    agg_df = results_df.groupby(['Model', 'FeatureSet'])[metric].mean().reset_index()
    agg_df = agg_df.sort_values(metric, ascending=False)
    
    # Select top 20 for readability if too many, but user wants to see more.
    # Let's increase limit or make it optional.
    if len(agg_df) > 30:
        print(f"Truncating leaderboard to top 30 (total {len(agg_df)}) for readability.")
        agg_df = agg_df.head(30)

    chart = sns.barplot(
        data=agg_df, 
        x='Model', 
        y=metric, 
        hue='FeatureSet', 
        palette='viridis'
    )
    
    plt.title(title, fontsize=16, weight='bold')
    plt.ylabel(metric)
    plt.xlabel("Model")
    plt.legend(title='Feature Set', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filepath}")
        plt.close()
    else:
        plt.show()


def plot_risk_return_scatter(
    results_df: pd.DataFrame, 
    filepath: Optional[str] = None
) -> None:
    """Scatter plot of Directional Accuracy vs Strategy Sharpe (Regression).

    Args:
        results_df (pd.DataFrame): Experiment results.
        filepath (Optional[str]): Path to save the figure.
    """
    if results_df.empty:
        return
        
    df = results_df[results_df['TargetType'] == 'continuous'].copy()
    if df.empty:
        return
    
    # Aggregate
    agg_cols = ['Strategy Sharpe', 'Directional Accuracy']
    agg_df = df.groupby(['Model', 'FeatureSet'])[agg_cols].mean().reset_index()
    
    plt.figure(figsize=(10, 8))
    
    # Logic to handle too many feature sets for 'style' mapping
    unique_fsets = agg_df['FeatureSet'].nunique()
    use_style = 'FeatureSet' if unique_fsets <= 10 else None
    
    scatter = sns.scatterplot(
        data=agg_df,
        x='Directional Accuracy',
        y='Strategy Sharpe',
        hue='Model',
        style=use_style,
        s=150,
        palette='deep',
        alpha=0.8
    )
    
    # Annotate Top Performers (Top 3 by Sharpe)
    top_performers = agg_df.sort_values('Strategy Sharpe', ascending=False).head(3)
    for _, row in top_performers.iterrows():
        plt.text(
            row['Directional Accuracy'] + 0.002, 
            row['Strategy Sharpe'] + 0.02, 
            f"{row['Model']}", 
            fontsize=9,
            weight='bold'
        )
        
    plt.title("Risk-Return Tradeoff: Sharpe vs Directional Accuracy", fontsize=16, weight='bold')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    plt.axvline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1) # 50% accuracy baseline
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved scatter plot to {filepath}")
        plt.close()
    else:
        plt.show()


def plot_model_performance_heatmap(
    results_df: pd.DataFrame, 
    metric: str, 
    filepath: Optional[str] = None
) -> None:
    """Plots a heatmap of Model vs FeatureSet for a specific metric.

    Args:
        results_df (pd.DataFrame): Experiment results.
        metric (str): Metric to visualize.
        filepath (Optional[str]): Path to save the figure.
    """
    if results_df.empty:
        return

    # Pivot Data: Rows=Model, Cols=FeatureSet, Values=Metric
    pivot_df = results_df.groupby(['Model', 'FeatureSet'])[metric].mean().unstack()
    
    if pivot_df.empty:
        return

    plt.figure(figsize=(14, 10))
    sns.set(font_scale=1.1)
    
    # Center map around 0 for Sharpe/R2, or 0.5 for Accuracy/AUC
    center_val = 0.5 if metric in ['Accuracy', 'AUC', 'Directional Accuracy', 'F1'] else 0
    
    sns.heatmap(
        pivot_df, 
        annot=True, 
        fmt=".3f", 
        cmap="RdYlGn", 
        center=center_val,
        linewidths=.5,
        cbar_kws={'label': metric}
    )
    
    plt.title(f"Heatmap: {metric} by Model & Feature Set", fontsize=18, weight='bold')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {filepath}")
        plt.close()
    else:
        plt.show()


def get_top_results(metrics: Dict[str, bool], df: pd.DataFrame):
    best_dfs = {}
    metrics_names = list(metrics.keys())
    agg = df.groupby(["Model", "FeatureSet", "Ticker"])[metrics_names].agg(["mean", "std"])
    
    for metric, lower_is_better in metrics.items():
        series = agg[(metric, "mean")].groupby(level=0)
        best_indices = (series.idxmin() if lower_is_better else series.idxmax())
        best_df = (
            agg.loc[best_indices]
            .reset_index()
            .sort_values((metric, "mean"), ascending=lower_is_better)
        )
        best_dfs[metric] = best_df

    for metric, df_metric in best_dfs.items():
        print(f"\n=== Top {metric} ===")

        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.expand_frame_repr", False)

        df_metric.columns = [
            f"{c[0]}_{c[1]}" if isinstance(c, tuple) else c
            for c in df_metric.columns
        ]
        df_metric.columns = [col.rstrip("_") for col in df_metric.columns]

        df_metric = df_metric.round(4)
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

    plt.show()


def plot_metrics_by_featureset(metrics: List[str], df: pd.DataFrame):
    for metric in metrics:
        filtered_cls = df[~df["Model"].isin(BASELINES)].copy()
        plot_metric_by_featureset_scatter(filtered_cls, metric)
