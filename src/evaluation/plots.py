import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict

from src.config import DEF_SPLITS, COMPANY_COLORS
from src.utils import set_style, apply_academic_style, ensure_dataframe

ACADEMIC_PALETTE = ["#1B263B", "#0A9396", "#EE9B00", "#CA6702", "#9B2226"]


def plot_walk_forward_validation(n_splits: int = DEF_SPLITS, total_samples: int = 100) -> None:
    """Visualize strict walk-forward validation (expanding window)."""
    if n_splits < 1:
        raise ValueError("n_splits must be a positive integer.")

    set_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    # Boundaries evenly divide the total timeline into n_splits test slices plus a tail buffer.
    boundaries = np.linspace(0, total_samples, n_splits + 2)
    train_color, test_color = ACADEMIC_PALETTE[0], ACADEMIC_PALETTE[2]
    bar_height = 0.55

    for fold_idx in range(n_splits):
        train_end = boundaries[fold_idx + 1]
        test_end = boundaries[fold_idx + 2]
        train_width = train_end
        test_width = test_end - train_end

        ax.barh(
            y=fold_idx,
            width=train_width,
            left=0,
            height=bar_height,
            color=train_color,
            alpha=0.9,
            label="Train (expanding window)" if fold_idx == 0 else "",
        )
        ax.barh(
            y=fold_idx,
            width=test_width,
            left=train_end,
            height=bar_height,
            color=test_color,
            alpha=0.9,
            label="Test (next slice)" if fold_idx == 0 else "",
        )
        ax.vlines(
            train_end,
            fold_idx - bar_height / 2 - 0.05,
            fold_idx + bar_height / 2 + 0.05,
            colors="#d9d9d9",
            linestyles="--",
            linewidth=1.1,
        )

    ax.set_xlim(0, total_samples)
    ax.set_xticks(boundaries)
    ax.set_xticklabels([f"{int(x)}" if x.is_integer() else f"{x:.1f}" for x in boundaries])
    ax.set_yticks(range(n_splits))
    ax.set_yticklabels([f"Fold {i+1}" for i in range(n_splits)])
    ax.set_xlabel("Time Index")
    ax.legend(loc="lower right", frameon=False)
    ax.margins(y=0.12)
    apply_academic_style(ax, "Strict Walk-Forward Validation (Expanding Window)")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_return_distributions(df: pd.DataFrame | Dict[str, pd.DataFrame]) -> None:
    """KDE of log-return distributions across tickers."""
    set_style()
    df = ensure_dataframe(df)
    
    if "Ticker" not in df.columns or "Log_Return" not in df.columns:
        print("Return distribution plot requires 'Ticker' and 'Log_Return' columns.")
        return
    plt.figure(figsize=(10, 5))
    for ticker, group in df.groupby("Ticker"):
        sns.kdeplot(group["Log_Return"].dropna(), label=ticker, color=COMPANY_COLORS.get(ticker, None))
    plt.title("Log Return Distributions by Ticker", fontsize=16)
    plt.xlabel("Log Return")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame | Dict[str, pd.DataFrame]) -> None:
    """Correlation heatmap of log returns across tickers."""
    set_style()
    df = ensure_dataframe(df)

    if "Ticker" not in df.columns or "Log_Return" not in df.columns:
        print("Correlation heatmap requires 'Ticker' and 'Log_Return' columns.")
        return
    pivot_ret = df.pivot_table(index=df.index, columns="Ticker", values="Log_Return")
    # Drop tickers with no data to avoid empty/NaN-only heatmaps
    pivot_ret = pivot_ret.dropna(axis=1, how="all")
    if pivot_ret.empty:
        print("No log return data available to plot correlation heatmap.")
        return
    corr = pivot_ret.corr().round(2)
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="Blues", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Correlation of Log Returns", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_stationarity_check(df: pd.DataFrame, target_col: str = "Log_Return", window: int = 30, ticker: str = "Stock") -> None:
    """
    Plot stationarity checks: Raw Price, Log Returns, and Rolling Statistics.
    
    Args:
        df: DataFrame containing price data.
        target_col: Name of the return column. Calculated if missing.
        window: Window size for rolling statistics.
        ticker: Ticker symbol for display.
    """
    set_style()
    
    # Ensure dataframe is a copy to avoid side effects
    plot_df = df.copy()
    
    # Robustly handle index
    if not isinstance(plot_df.index, pd.DatetimeIndex):
        plot_df.index = pd.to_datetime(plot_df.index)
        
    # Calculate Log_Return if missing
    if target_col not in plot_df.columns:
        if "Close" in plot_df.columns:
            plot_df[target_col] = np.log(plot_df["Close"] / plot_df["Close"].shift(1))
        elif "Adj Close" in plot_df.columns:
            plot_df[target_col] = np.log(plot_df["Adj Close"] / plot_df["Adj Close"].shift(1))
            
    plot_df = plot_df.dropna()
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # 1. Raw Price
    price_col = "Close" if "Close" in plot_df.columns else "Adj Close"
    if price_col in plot_df.columns:
        axes[0].plot(plot_df.index, plot_df[price_col], label=f"{price_col} Price", color="#1B263B")
        axes[0].set_title(f"{ticker} Raw Price ($P_t$): Non-Stationary", fontweight="bold")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, alpha=0.3)
    
    # 2. Log Returns
    if target_col in plot_df.columns:
        axes[1].plot(plot_df.index, plot_df[target_col], label="Log Returns", color="#CA6702", alpha=0.8)
        axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[1].set_title(f"Log Returns ($Y_t$): Stationary", fontweight="bold")
        axes[1].legend(loc="upper left")
        axes[1].grid(True, alpha=0.3)
        
        # 3. Rolling Statistics
        rolling_mean = plot_df[target_col].rolling(window=window).mean()
        rolling_std = plot_df[target_col].rolling(window=window).std()
        
        axes[2].plot(plot_df.index, plot_df[target_col], label="Log Returns", color="#CA6702", alpha=0.2)
        axes[2].plot(plot_df.index, rolling_mean, label=f"{window}-Day Mean", color="#0A9396", linewidth=2)
        axes[2].plot(plot_df.index, rolling_std, label=f"{window}-Day Std", color="#9B2226", linestyle="--", linewidth=2)
        axes[2].set_title(f"Rolling Statistics ({window} Days)", fontweight="bold")
        axes[2].legend(loc="upper left")
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
