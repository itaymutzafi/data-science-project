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

def plot_autocorrelation(series: pd.Series, lags: int = 40):
    """
    Plots the Autocorrelation Function (ACF) to analyze serial correlation.
    
    Args:
        series (pd.Series): Time series data (e.g., log returns).
        lags (int): Number of lags to plot.
    """
    from statsmodels.graphics.tsaplots import plot_acf
    
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_acf(series.dropna(), lags=lags, ax=ax, alpha=0.05)
    ax.set_title(f"Autocorrelation (ACF) - {series.name}", fontweight='bold')
    ax.set_xlabel("Lags")
    ax.set_ylabel("Correlation")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_context_comparison(target_series: pd.Series, aux_data: pd.DataFrame, target_name: str):
    """
    Visualizes the target stock against two context groups:
    1. Macro Context (Nasdaq 100)
    2. Segment Context (NVIDIA)
    Includes a separate panel for Volatility (VIX).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Align dates
    # Ensure timezone-naive for consistent comparison
    if target_series.index.tz is not None:
        target_series = target_series.copy()
        target_series.index = target_series.index.tz_localize(None)
    
    # Ensure index is DatetimeIndex
    if not isinstance(aux_data.index, pd.DatetimeIndex):
        try:
            aux_data.index = pd.to_datetime(aux_data.index)
        except Exception:
             pass # Let it fail downstream or handle gracefully, but at least we tried.

    if aux_data.index.tz is not None:
        aux_data = aux_data.copy()
        aux_data.index = aux_data.index.tz_localize(None)

    common_idx = target_series.index.intersection(aux_data.index)
    
    if common_idx.empty:
        raise ValueError(
            f"No overlapping dates found between target ({target_series.index.min().date()} - {target_series.index.max().date()}) "
            f"and auxiliary data ({aux_data.index.min().date()} - {aux_data.index.max().date()})."
        )

    target = target_series.loc[common_idx]
    aux = aux_data.loc[common_idx]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # --- Plot 1: Normalized Performance (Base=100) ---
    ax1 = axes[0]
    
    # Normalize to 100 for valid visual comparison
    norm_target = (target / target.iloc[0]) * 100
    norm_macro = (aux['Nasdaq_100'] / aux['Nasdaq_100'].iloc[0]) * 100
    norm_segment = (aux['NVIDIA_Segment_Leader'] / aux['NVIDIA_Segment_Leader'].iloc[0]) * 100
    
    ax1.plot(norm_target, label=f'{target_name} (Target)', linewidth=2.5, color='#1f77b4')
    ax1.plot(norm_macro, label='Nasdaq 100 (Macro Baseline)', linestyle='--', alpha=0.8, color='black')
    ax1.plot(norm_segment, label='NVIDIA (Segment Peer)', linestyle=':', alpha=0.8, color='green')
    
    ax1.set_title(f'Contextual Analysis: {target_name} vs. Market & Segment', fontsize=12)
    ax1.set_ylabel('Normalized Price (Base=100)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Market Sentiment (VIX) ---
    ax2 = axes[1]
    color_vix = 'tab:red'
    ax2.plot(aux['VIX_Index'], color=color_vix, label='VIX (Volatility Index)', linewidth=1.5)
    ax2.set_ylabel('VIX Index', color=color_vix)
    ax2.tick_params(axis='y', labelcolor=color_vix)
    ax2.fill_between(aux.index, aux['VIX_Index'], alpha=0.1, color=color_vix)
    ax2.set_title('Market Risk Sentiment (VIX)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_sentiment_vs_price(df: pd.DataFrame, sentiment_col: str = 'sentiment_mean_lag1', days: int = None):
    """
    Visualizes stock price vs sentiment score for the last N days.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock price ('Close') and sentiment.
        sentiment_col (str): Column name for sentiment score.
        days (int): Number of last days to plot. If None, plots all available data.
    """
    if sentiment_col not in df.columns:
        print(f"Column '{sentiment_col}' not found. Skipping plot.")
        return

    fig, ax1 = plt.subplots(figsize=(14, 7))

    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Stock Price', color=color)
    
    # Slice last N days if specified
    if days:
        plot_data = df.iloc[-days:]
        title_suffix = f"(Last {days} Days)"
    else:
        plot_data = df
        title_suffix = "(Full Period)"
    
    ax1.plot(plot_data.index, plot_data['Close'], color=color, label='Close Price')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:orange'
    ax2.set_ylabel('Sentiment Score (Lagged)', color=color)
    ax2.plot(plot_data.index, plot_data[sentiment_col], color=color, linestyle='--', label='Sentiment (Lag 1)', alpha=0.6)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f"AAPL Stock Price vs. FinBERT Sentiment {title_suffix}")
    fig.tight_layout()
    plt.show()
