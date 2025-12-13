"""Plotting utilities for consistent academic visualization.

This module defines the global style settings for matplotlib/seaborn
to ensure all figures in the report have a uniform, professional appearance.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as patches
import seaborn as sns
from src.config import COMPANY_COLORS, TICKER_TO_COMPANY_MAP, TICKER

def set_style():
    """Sets the global plotting style for the project."""
    # Use a clean, academic style
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # Custom overrides for better readability in reports
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "axes.titlesize": 16,     # Larger font size as requested
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2.0,
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
    axes[0].plot(df.index, df['Close'], label='Close Price', color=COMPANY_COLORS.get(TICKER, '#1f77b4'))
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
    
    ax1.plot(norm_target, label=f'{target_name} (Target)', linewidth=2.5, color=COMPANY_COLORS.get(target_name, '#1f77b4'))
    ax1.plot(norm_macro, label='Nasdaq 100 (Macro Baseline)', linestyle='--', alpha=0.8, color='black')
    ax1.plot(norm_segment, label='NVIDIA (Segment Peer)', linestyle=':', alpha=0.8, color=COMPANY_COLORS.get('NVDA', 'green'))
    
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

def plot_sentiment_trends(daily_sentiment_df: pd.DataFrame):
    """
    Plots a 4-split grid (2x2) showing both Raw Sentiment (daily means) and 
    Smoothed (7-day MA) Trends for all major companies.
    
    Args:
        daily_sentiment_df (pd.DataFrame): DataFrame with 'date', 'company', 'sentiment_mean', 'sentiment_ma_7d'.
    """
    if daily_sentiment_df.empty:
        print("No sentiment data available to plot.")
        return

    df = daily_sentiment_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Ensure MA exists
    if 'sentiment_ma_7d' not in df.columns:
        df['sentiment_ma_7d'] = df.groupby('company')['sentiment_mean'].transform(lambda x: x.rolling(7, min_periods=1).mean())

    companies = ['Apple', 'Amazon', 'Google', 'Microsoft'] # Fixed order
    available_companies = [c for c in companies if c in df['company'].unique()]
    
    if not available_companies:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    
    # Fixed mapping ensures consistent placement regardless of missing data
    # (0,0)=Apple, (0,1)=Amazon, (1,0)=Google, (1,1)=Microsoft
    grid_map = {
        'Apple': 0, 
        'Amazon': 1, 
        'Google': 2, 
        'Microsoft': 3
    }
    
    for company in companies: # Iterate fixed list
        if company not in grid_map: continue
        
        idx = grid_map[company]
        ax = axes_flat[idx]
        
        subset = df[df['company'] == company].sort_values('date')
        
        if subset.empty:
            ax.text(0.5, 0.5, "No Data Available", ha='center', va='center')
            ax.set_title(company, fontweight='bold')
            continue

        color = COMPANY_COLORS.get(company, 'blue')
        
        # 1. Raw Daily Sentiment (Scatter) - Only plot days with actual news
        raw_days = subset[subset['news_count'] > 0]
        if not raw_days.empty:
            ax.scatter(raw_days['date'], raw_days['sentiment_mean'], 
                       color=color, alpha=0.3, s=15, label='Daily Raw Sentiment')
        
        # 2. Smoothed Trend (Line)
        # Handle sparse data: drop NaNs for plotting line to avoid gaps
        trend_data = subset.dropna(subset=['sentiment_ma_7d'])
        if not trend_data.empty:
            ax.plot(trend_data['date'], trend_data['sentiment_ma_7d'], 
                    color=color, linewidth=2.5, label='7-Day Moving Avg')
        
        ax.set_title(f"{company}", fontweight='bold', fontsize=12)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        # Axis formatting
        ax.tick_params(axis='x', rotation=45)
        
        # Legend only on first plot
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    fig.suptitle("Sentiment Trends: Raw Signals vs. Smoothed Trends", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) 
    plt.show()

def plot_sentiment_decay_verification():
    """
    Visualizes the effect of Exponential Decay on a synthetic signal
    to verify the logic of filling missing dates.
    """
    import numpy as np
    import pandas as pd
    from src.features.sentiment_analysis import process_sentiment_timeseries
    
    # Create a synthetic signal with gaps
    demo_dates = pd.date_range("2024-01-01", periods=15)
    demo_df = pd.DataFrame({'date': demo_dates, 'company': 'Demo', 'sentiment_mean': np.nan, 'news_count': 0})
    
    # Set some events
    demo_df.loc[0, 'sentiment_mean'] = 1.0; demo_df.loc[0, 'news_count'] = 5
    demo_df.loc[5, 'sentiment_mean'] = -0.8; demo_df.loc[5, 'news_count'] = 3
    
    # Apply processing
    processed_demo = process_sentiment_timeseries(demo_df)
    
    plt.figure(figsize=(10, 5))
    plt.plot(processed_demo['date'], processed_demo['sentiment_mean'], marker='o', label='Decayed Sentiment', color='purple')
    
    # Plot original events
    events = demo_df.dropna(subset=['sentiment_mean'])
    plt.scatter(events['date'], events['sentiment_mean'], color='red', s=100, label='News Events', zorder=5)
    
    plt.title("Verification: Exponential Decay of Sentiment Signal ($S_t = S_{t-1} \\times 0.85$)", fontweight='bold')
    plt.ylabel("Sentiment Score")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_sentiment_signal_quality(daily_sentiment_df: pd.DataFrame, company: str = 'Apple', ax=None):
    """
    Visualizes the "Signal Continuity" aspect (Raw vs Decayed) for a single company.
    Designed to be part of a larger grid.
    
    Args:
        daily_sentiment_df (pd.DataFrame): DataFrame with sentiment features.
        company (str): Company to visualize.
        ax (matplotlib.axes.Axes): Axis to plot on.
    """
    if daily_sentiment_df.empty:
        return
        
    subset = daily_sentiment_df[daily_sentiment_df['company'] == company].copy().sort_values('date')
    if subset.empty:
        return
        
    # Last 180 days
    subset = subset.iloc[-180:] 
    
    # Color Setup
    main_color = COMPANY_COLORS.get(company, '#34A853')
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        
    # "Raw" points are where news_count > 0
    raw_points = subset[subset['news_count'] > 0]
    
    # 1. Plot Continuous (Decayed) Signal Line
    ax.plot(subset['date'], subset['sentiment_mean'], color=main_color, linestyle='-', alpha=0.6, label='Decayed Signal')
    
    # 2. Plot Raw Points
    ax.scatter(raw_points['date'], raw_points['sentiment_mean'], color=main_color, s=25, label='Raw News Days', zorder=5)
    
    ax.set_title(f"{company}", fontweight='bold')
    ax.grid(True, alpha=0.3)
    # No legend per subplot to save space, or minimal


def plot_day_sentiment_breakdown(daily_news_df: pd.DataFrame, date: str, company: str, daily_score: float):
    """
    Visualizes the raw news sentiment distribution for a single day to demystify the aggregation.
    
    Args:
        daily_news_df (pd.DataFrame): Subset of news for that day/company with 'sentiment' scores.
        date (str): The date being analyzed.
        company (str): The company name.
        daily_score (float): The final aggregated score for that day.
    """
    if daily_news_df.empty:
        print(f"No news found for {company} on {date}")
        return

    # Extract scores
    scores = daily_news_df['sentiment_score'].values
    headlines = daily_news_df['headline'].values
    
    # Create Layout
    fig = plt.figure(figsize=(14, 7))
    grid = plt.GridSpec(1, 2, width_ratios=[1.2, 1])
    
    # Left: Distribution
    ax1 = fig.add_subplot(grid[0])
    sns.histplot(scores, kde=True, ax=ax1, color='skyblue', bins=10)
    ax1.axvline(daily_score, color='red', linestyle='--', linewidth=2, label=f'Daily Mean: {daily_score:.2f}')
    ax1.set_title(f"Sentiment Distribution: {company} on {date}", fontweight='bold')
    ax1.set_xlabel("Sentiment Score (-1 to 1)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Top/Bottom Headlines
    ax2 = fig.add_subplot(grid[1])
    ax2.axis('off')
    
    # Sort samples
    # Deduplicate headlines to avoid repetitive "Key Headlines"
    daily_news_df = daily_news_df.drop_duplicates(subset=['headline'])
    
    daily_news_df = daily_news_df.sort_values(by='sentiment_score', ascending=False)
    top_pos = daily_news_df.head(3)
    top_neg = daily_news_df.tail(3)
    
    import textwrap
    
    text_content = f"### Key Headlines ({len(daily_news_df)} Total)\n\n"
    
    text_content += "**Most Positive:**\n"
    for _, row in top_pos.iterrows():
        headline = row['headline']
        short_headline = textwrap.shorten(headline, width=60, placeholder="...")
        text_content += f"😄 ({row['sentiment_score']:.2f}) {short_headline}\n"
        
    text_content += "\n**Most Negative:**\n"
    for _, row in top_neg.iterrows():
        headline = row['headline']
        short_headline = textwrap.shorten(headline, width=60, placeholder="...")
        text_content += f"😡 ({row['sentiment_score']:.2f}) {short_headline}\n"
        
    ax2.text(0.05, 0.95, text_content, fontsize=10, va='top', ha='left', family='monospace')
    
    plt.tight_layout()
    plt.show()

def plot_advanced_sentiment_features(daily_sentiment_df: pd.DataFrame):
    """
    Plots advanced sentiment signal quality (Signal Continuity) for all companies 
    in a 2x2 grid to show how data gaps are handled.
    
    Args:
        daily_sentiment_df (pd.DataFrame): DataFrame with sentiment features.
    """
    if daily_sentiment_df.empty:
        print("No data to plot.")
        return
        
    companies = ['Apple', 'Amazon', 'Google', 'Microsoft']
    available = [c for c in companies if c in daily_sentiment_df['company'].unique()]
    
    if not available:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, company in enumerate(available):
        plot_sentiment_signal_quality(daily_sentiment_df, company, ax=axes[i])
        
    fig.suptitle("Feature Engineering: Signal Continuity (Filling Missing Days with Decay)", 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Global Legend (fake handle)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, alpha=0.6, label='Decayed Signal (Continuous)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Raw News Days')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

def plot_performance_comparison(results_df: pd.DataFrame, metric: str = "MSE"):
    """
    Plots a bar chart comparing model performance across different metrics.
    
    Args:
        results_df (pd.DataFrame): DataFrame with index as Model Name and columns as Metrics.
        metric (str): The specific metric to sort and highlight.
    """
    if results_df.empty:
        return
        
    plt.figure(figsize=(10, 6))
    
    # Sort by metric
    sorted_df = results_df.sort_values(metric, ascending=(metric not in ['R2', 'Sharpe', 'Directional Accuracy']))
    
    colors = ['#1f77b4' if 'LSTM' in idx else '#d62728' if 'Market' in idx or 'Baseline' in idx else 'gray' for idx in sorted_df.index]
    
    bars = plt.bar(sorted_df.index, sorted_df[metric], color=colors, alpha=0.8)
    
    plt.title(f"Model Comparison: {metric}", fontweight='bold', fontsize=14)
    plt.ylabel(metric)
    plt.xlabel("Model strategy")
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')
                 
    plt.tight_layout()
    plt.show()

def plot_cumulative_returns(y_true: pd.Series, model_predictions: dict):
    """
    Plots the cumulative equity curve of trading strategies based on model predictions.
    strategy_return = sign(pred) * y_true
    
    Args:
        y_true (pd.Series): Actual returns.
        model_predictions (dict): Dictionary {ModelName: y_pred_series}.
    """
    plt.figure(figsize=(12, 6))
    
    # Cumulative Product of (1 + r) is strictly correct for prices, but for simple log returns summing is approx ok.
    # Let's use cumulative sum of log returns -> Log Price Level. 
    # Or exp(cumsum) -> Normalized Price.
    
    # 1. Market (Buy & Hold)
    market_curve = np.exp(y_true.cumsum())
    plt.plot(market_curve.index, market_curve, label='Market (Buy & Hold)', color='black', linewidth=2, linestyle='--')
    
    for name, y_pred in model_predictions.items():
        # Align
        common_idx = y_true.index.intersection(y_pred.index)
        if common_idx.empty:
            continue
            
        truth = y_true.loc[common_idx]
        pred = y_pred.loc[common_idx]
        
        # Strategy: Buy if pred > 0, Sell if pred < 0.
        # Note: Shorting allowed? If yes, sign(pred)*truth. 
        # If Long-Only: (pred>0) * truth.
        # Let's assume Long-Short for "Signal Strength" demonstration, or standard "Directional Strategy".
        
        strat_returns = np.sign(pred) * truth
        strat_curve = np.exp(strat_returns.cumsum())
        
        # Color logic
        color = None
        if 'LSTM' in name: color = '#1f77b4' # Blue
        elif 'Baseline' in name: color = 'gray'
        
        plt.plot(common_idx, strat_curve, label=f'{name} Strategy', linewidth = 2 if 'LSTM' in name else 1.5, color=color)
        
    plt.title("Equity Curve Comparison (Cumulative Returns)", fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Normalized Wealth (Start=1.0)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

