"""
Visualization module for experiment result comparisons.

This module provides functions to generate academic-quality plots using Seaborn and Matplotlib.
It supports Leaderboards, Risk-Return Scatter plots, and Performance Heatmaps.
"""

import os
from typing import Optional, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set global style defaults for consistency
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


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
    
    scatter = sns.scatterplot(
        data=agg_df,
        x='Directional Accuracy',
        y='Strategy Sharpe',
        hue='Model',
        style='FeatureSet',
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
