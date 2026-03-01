import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
import math
import seaborn as sns

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


def plot_feature_target_correlation(dfs: Dict[str, pd.DataFrame], feature_cols: list, title: str = "Feature Correlation") -> None:
    """
    Plots a heatmap of correlations between features and the target (Log_Return).
    Displays the Pearson Correlation Coefficient.
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
    # Use squeeze=False to always get 2D array of axes, but handle 1D case simply
    fig, axes = plt.subplots(1, len(valid_tickers), figsize=(3 * len(valid_tickers), 4), sharey=True)
    if len(valid_tickers) == 1:
        axes = [axes]
    else:
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
        
    plt.suptitle(title, fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(dfs: Dict[str, pd.DataFrame], features: list = None, title: str = "Feature Correlation Matrix") -> None:
    """
    Plots a full correlation matrix (heatmap) for the specified features for each ticker.
    Useful for checking relationships between RSI, Volatility, Volume, etc.
    """
    set_style()
    tickers = list(dfs.keys())
    
    if features is None:
        features = ['RSI', 'Log_Return', 'Volume', 'MACD']
        
    valid_tickers = [t for t in tickers if all(f in dfs[t].columns for f in features)]
    
    if not valid_tickers:
        print(f"Skipping plot: Missing features {features} in data.")
        return

    cols = 2
    rows = math.ceil(len(valid_tickers) / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    if len(valid_tickers) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    for i, ticker in enumerate(valid_tickers):
        ax = axes[i]
        df = dfs[ticker][features].dropna()
        if df.empty:
            continue
            
        corr = df.corr() # Pearson by default
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', center=0, fmt=".2f", ax=ax, square=True)
        ax.set_title(f"{ticker}: Feature Correlations (Pearson)", fontweight='bold')
        
    # Hide unused
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
