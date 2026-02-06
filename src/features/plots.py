import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

from src.config import DAYNAMES, MONTHNAMES, COMPANY_COLORS, TICKERS
from src.utils import set_style

# --- Generic helpers ---
def avg_attr_by_time_plot(dfs: Dict[str, pd.DataFrame], column: str, time_precision: str) -> None:
    if all(column in df.columns for df in dfs.values()):
        plt.figure(figsize=(12, 6))
        width = 0.2
        multiplier = 0

        if time_precision == 'Day':
            time_names = DAYNAMES
            time_range = time_names
        else:
            time_names = MONTHNAMES
            time_range = range(1, 13)
        x = np.arange(len(time_names))
        
        for name, df in dfs.items():
            y = df.groupby(time_precision)[column].mean()
            y = y.reindex(time_range)
            offset = width * multiplier
            plt.bar(x + offset, y.values, width, label=name, color=COMPANY_COLORS[name], alpha=0.8)
            multiplier += 1
        
        plt.xlabel(time_precision)
        plt.ylabel(f'Avg {column}')
        plt.title(f'Average {column} by {time_precision}')
        plt.xticks(x + width * (len(TICKERS) - 1) / 2, time_names, rotation=45)
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()


# --- News ---
def date_groupby_line_plot(df: pd.DataFrame, yname: str, title: str) -> None:
    set_style()
    per_day = df.groupby(df["date"].dt.date).size()
    per_day.plot(kind="line")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(yname)
    plt.grid(True)
    plt.show()


def articles_over_time_by_dataset_plot(df: pd.DataFrame, is_log: bool, specific_years=None) -> None:
    set_style()
    grouped = (
        df.set_index("date")
        .groupby("dataset")
        .resample("ME", include_groups=False)
        .size()
        .unstack(level=0)
    )

    plt.figure()
    for column in grouped.columns:
        plt.plot(grouped.index, grouped[column], label=column)

    plt.xlabel("Time")
    plt.ylabel("Log Number of Articles" if is_log else "Number of Articles")
    if is_log:
        plt.yscale("log")

    title = "Articles Over Time by Dataset"
    if specific_years is not None:
        title += f" {specific_years}"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def article_volume_per_company_plot(df: pd.DataFrame) -> None:
    set_style()
    for company, group in df.groupby("company"):
        monthly_counts = group.set_index("date").resample("ME").size()
        plt.plot(monthly_counts.index, monthly_counts.values, label=company)

    plt.xlabel("Time")
    plt.ylabel("Number of Articles")
    plt.title("Monthly News Volume per Company")
    plt.legend(title="Company")
    plt.tight_layout()
    plt.show()


def pie_plot(counts: pd.Series, subject: str) -> None:
    set_style()
    plt.figure()
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%")
    plt.title(f"Distribution of {subject}")
    plt.show()


def table_visualize(df: pd.DataFrame, groupby) -> pd.DataFrame:
    return df.groupby(groupby).size().unstack(fill_value=0)


def plot_interaction_correlations(dfs: Dict[str, pd.DataFrame], interaction_features: list = None) -> None:
    """
    Plots the correlation of interaction features with Log_Return for each ticker.
    """
    set_style()
    if interaction_features is None:
        interaction_features = ['Vol_x_Return', 'MACD_x_RSI', 'Trend_x_RSI']
    
    correlation_data = {}
    
    for ticker, df in dfs.items():
        if 'Log_Return' not in df.columns:
            continue
            
        corrs = {}
        for feature in interaction_features:
            if feature in df.columns:
                corrs[feature] = df[feature].corr(df['Log_Return'])
        if corrs:
            correlation_data[ticker] = corrs

    if not correlation_data:
        print("No interaction features or Log_Return found for correlation analysis.")
        return

    # Plotting
    df_corr = pd.DataFrame(correlation_data).T
    
    if df_corr.empty:
        print("Correlation data is empty.")
        return

    df_corr.plot(kind='bar', figsize=(12, 6), colormap='viridis', width=0.8)
    plt.title("Correlation of Interaction Features with Log Returns")
    plt.ylabel("Pearson Correlation")
    plt.xlabel("Ticker")
    plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    plt.legend(title="Features", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

import math
import seaborn as sns

def plot_rsi_grid(dfs: Dict[str, pd.DataFrame], lookback: int = 200) -> None:
    """
    Plots a grid of RSI vs Price for all tickers in the dictionary.
    """
    set_style()
    tickers = list(dfs.keys())
    n_tickers = len(tickers)
    cols = 2
    rows = math.ceil(n_tickers / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows), sharex=True)
    if n_tickers == 1:
        axes = [axes]
    else:
        axes = axes.flatten() 
    
    for i, ticker in enumerate(tickers):
        ax = axes[i]
        df = dfs[ticker].iloc[-lookback:] # Last N days
        color = COMPANY_COLORS.get(ticker, '#333333')
        
        # Price (Left Axis)
        ax.plot(df.index, df['Close'], label='Price', color=color, alpha=0.8)
        ax.set_title(f"{ticker}: RSI Divergence", fontweight='bold')
        
        # RSI (Right Axis)
        ax_rsi = ax.twinx()
        if 'RSI' in df.columns:
            ax_rsi.plot(df.index, df['RSI'], label='RSI', color='#E63946', linewidth=1)
            ax_rsi.axhline(70, color='red', linestyle=':', alpha=0.3)
            ax_rsi.axhline(30, color='green', linestyle=':', alpha=0.3)
            ax_rsi.fill_between(df.index, df['RSI'], 70, where=(df['RSI']>=70), color='red', alpha=0.1)
            ax_rsi.fill_between(df.index, df['RSI'], 30, where=(df['RSI']<=30), color='green', alpha=0.1)
            ax_rsi.set_ylim(0, 100)
            if i % cols == 1: ax_rsi.set_ylabel("RSI")
        
        # Labels only on edges
        if i % cols == 0: ax.set_ylabel("Price ($)")
        
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_ma_distance_grid(dfs: Dict[str, pd.DataFrame], window: int) -> None:
    """
    Plots a grid of Trend Distance for all tickers.
    """
    set_style()
    tickers = list(dfs.keys())
    n_tickers = len(tickers)
    cols = 2
    rows = math.ceil(n_tickers / cols)
    
    dist_col = f'Dist_MA{window}'
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows), sharex=True)
    if n_tickers == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, ticker in enumerate(tickers):
        ax = axes[i]
        df = dfs[ticker]
        color = COMPANY_COLORS.get(ticker, 'purple')
        
        if dist_col in df.columns:
            ax.plot(df.index, df[dist_col], color=color, linewidth=1, alpha=0.9)
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.set_title(f"{ticker}: {window}-Day Trend Dist", fontweight='bold')
            if i % cols == 0: ax.set_ylabel("Dist (%)")
    
    # Hide unused
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_feature_target_correlation(dfs: Dict[str, pd.DataFrame], feature_cols: list, title: str = "Feature Correlation") -> None:
    """
    Plots a heatmap of correlations between features and the target (Log_Return).
    """
    set_style()
    tickers = list(dfs.keys())
    
    if len(tickers) == 0:
        return

    # Check for tickers with data
    valid_tickers = [t for t in tickers if 'Log_Return' in dfs[t].columns and any(c in dfs[t].columns for c in feature_cols)]
    
    if not valid_tickers:
        print("No valid data for correlation plot.")
        return

    # Dynamic sizing based on number of tickers
    fig, axes = plt.subplots(1, len(valid_tickers), figsize=(3 * len(valid_tickers), 3.5), sharey=True, squeeze=False)
    axes = axes.flatten()
    
    for i, ticker in enumerate(valid_tickers):
        ax = axes[i]
        
        cols_to_use = [c for c in feature_cols if c in dfs[ticker].columns]
        if not cols_to_use:
            continue
            
        sub_df = dfs[ticker][cols_to_use + ['Log_Return']].dropna()
        if sub_df.empty:
            continue
            
        corr = sub_df.corr().iloc[:-1, -1:] # Correlation with Target only
        
        sns.heatmap(corr, annot=True, cmap='RdBu', center=0, fmt=".2f", ax=ax, cbar=False)
        ax.set_title(f"{ticker}", fontweight='bold')
        if i > 0: ax.set_ylabel("") 
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
