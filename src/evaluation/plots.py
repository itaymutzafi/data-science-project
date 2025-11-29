"""Plotting utilities for consistent academic visualization.

This module defines the global style settings for matplotlib/seaborn
to ensure all figures in the report have a uniform, professional appearance.
"""

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as patches
import seaborn as sns

def set_style():
    """Sets the global plotting style for the project."""
    # Use a clean, academic style
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # Custom overrides for better readability in reports
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 1.5,
        "grid.alpha": 0.3,
        "figure.dpi": 150  # High resolution for export
    })

def save_fig(fig, filename: str, folder: str = "results"):
    """Saves a figure to the results folder with consistent settings."""
    # Implementation placeholder for future use
    # fig.savefig(f"{folder}/{filename}", bbox_inches='tight')
    pass

def plot_walk_forward_validation(n_splits=5, total_samples=100):
    """Visualizes the Strict Walk-Forward Validation (Expanding Window) scheme."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Simulate expanding window indices
    step = total_samples // (n_splits + 1)
    
    for i in range(n_splits):
        train_end = (i + 1) * step
        test_end = train_end + step
        
        # Train bar
        ax.broken_barh([(0, train_end)], (i - 0.4, 0.8), facecolors='blue', label='Train' if i == 0 else "")
        # Test bar
        ax.broken_barh([(train_end, step)], (i - 0.4, 0.8), facecolors='orange', label='Test' if i == 0 else "")
        
    ax.set_yticks(range(n_splits))
    ax.set_yticklabels([f'Fold {i+1}' for i in range(n_splits)])
    ax.set_xlabel('Time Index')
    ax.set_title('Strict Walk-Forward Validation (Expanding Window)')
    ax.legend(loc='lower right')
    plt.grid(True, axis='x', alpha=0.3)
    plt.show()

def plot_price_vs_returns(df: pd.DataFrame, target_col: str) -> None:
    """Visualizes the contrast between raw prices and log-returns.

    Generates a dual-axis plot showing the non-stationary price series
    versus the stationary log-returns series.

    Args:
        df (pd.DataFrame): Dataframe containing 'Close' and target_col.
        target_col (str): Name of the log-returns column.
    """
    fig, axes = plt.subplots(2, 1, sharex=True)
    
    # Plot 1: Raw Price
    axes[0].plot(df.index, df['Close'], label='Close Price', color='#1f77b4')
    axes[0].set_title("Raw Price ($P_t$): Non-Stationary", fontweight='bold')
    axes[0].legend(loc='upper left')
    
    # Plot 2: Log Returns
    axes[1].plot(df.index, df[target_col], label='Log Returns', color='#ff7f0e', alpha=0.8)
    axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[1].set_title("Log Returns ($Y_t$): Stationary", fontweight='bold')
    axes[1].legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()
