import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, Literal
from sklearn.model_selection import TimeSeriesSplit

import src.config as project_config
from src.config import COMPANY_COLORS
from src.utils import set_style, apply_academic_style, ensure_dataframe

ACADEMIC_PALETTE = ["#1B263B", "#0A9396", "#EE9B00", "#CA6702", "#9B2226"]


def infer_calendar_index(
    data: Dict[str, pd.DataFrame],
    how: Literal["intersection", "union"] = "intersection",
) -> pd.DatetimeIndex:
    """Infer a shared market timeline from multiple ticker dataframes."""
    if not data:
        raise ValueError("data must contain at least one ticker dataframe.")

    index_sets = []
    for _, df in data.items():
        idx = pd.to_datetime(pd.Index(df.index))
        if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
            idx = idx.tz_localize(None)
        index_sets.append(set(idx))

    if how == "intersection":
        combined = set.intersection(*index_sets)
    elif how == "union":
        combined = set.union(*index_sets)
    else:
        raise ValueError("how must be 'intersection' or 'union'.")

    return pd.DatetimeIndex(sorted(combined))


def plot_walk_forward_validation(
    n_splits: int | None = None,
    total_samples: int | None = None,
    date_index: pd.Index | None = None,
    toc_safe: bool = True,
) -> None:
    """Visualize strict walk-forward validation (expanding window) using TimeSeriesSplit.

    Args:
        n_splits: Number of folds. If None, taken from src.config.SPLITS.
        total_samples: Timeline length when date_index is not provided.
        date_index: Optional datetime-like index for date-based x-axis labeling.
        toc_safe: Leave extra left margin for notebook TOC overlays.
    """
    if n_splits is None:
        n_splits = project_config.SPLITS
    if n_splits < 1:
        raise ValueError("n_splits must be a positive integer.")

    resolved_dates = None
    if date_index is not None:
        resolved_dates = pd.to_datetime(pd.Index(date_index))
        if isinstance(resolved_dates, pd.DatetimeIndex) and resolved_dates.tz is not None:
            resolved_dates = resolved_dates.tz_localize(None)
        total_samples = len(resolved_dates)
    elif total_samples is None:
        total_samples = 100

    if total_samples <= n_splits:
        raise ValueError("total_samples must be greater than n_splits.")

    set_style()
    splitter = TimeSeriesSplit(n_splits=n_splits)
    indices = np.arange(total_samples)
    splits = list(splitter.split(indices))

    fig_height = max(4.2, 1.8 + 0.55 * n_splits)
    fig, ax = plt.subplots(figsize=(12.5, fig_height))

    train_color = "#34495E"   # muted slate
    val_color = "#1F7A8C"     # muted teal
    base_color = "#EEF2F7"    # subtle timeline background
    bar_height = 0.58

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_start, train_end = int(train_idx[0]), int(train_idx[-1]) + 1
        val_start, val_end = int(val_idx[0]), int(val_idx[-1]) + 1
        train_width = train_end - train_start
        val_width = val_end - val_start

        # Full timeline background for each fold (for visual context)
        ax.barh(
            y=fold_idx,
            width=total_samples,
            left=0,
            height=bar_height,
            color=base_color,
            alpha=1.0,
            edgecolor="none",
            label="Future / not used in this fold" if fold_idx == 0 else "",
        )
        ax.barh(
            y=fold_idx,
            width=train_width,
            left=train_start,
            height=bar_height,
            color=train_color,
            alpha=0.9,
            label="Train (expanding window)" if fold_idx == 0 else "",
        )
        ax.barh(
            y=fold_idx,
            width=val_width,
            left=val_start,
            height=bar_height,
            color=val_color,
            alpha=0.9,
            label="Validation (next future slice)" if fold_idx == 0 else "",
        )
        ax.vlines(
            val_start,
            fold_idx - bar_height / 2 - 0.05,
            fold_idx + bar_height / 2 + 0.05,
            colors="#d9d9d9",
            linestyles="--",
            linewidth=1.1,
        )

    ax.set_xlim(0, total_samples)
    if resolved_dates is not None and len(resolved_dates) > 0:
        tick_count = min(7, len(resolved_dates))
        tick_idx = np.linspace(0, len(resolved_dates) - 1, num=tick_count, dtype=int)
        tick_idx = np.unique(tick_idx)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(resolved_dates[tick_idx].strftime("%Y-%m-%d"), rotation=20, ha="right")
        ax.set_xlabel("Date")
    else:
        tick_count = 7 if total_samples >= 60 else 6
        ax.set_xticks(np.linspace(0, total_samples, num=tick_count, dtype=int))
        ax.set_xlabel("Chronological Index")
    ax.set_yticks(range(n_splits))
    ax.set_yticklabels([f"Fold {i + 1}" for i in range(n_splits)])
    ax.invert_yaxis()
    ax.set_ylabel("CV Fold")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.grid(False, axis="y")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        ncol=3,
    )
    apply_academic_style(ax, "Strict Walk-Forward Validation")
    ax.set_facecolor("white")
    if toc_safe:
        # Leave extra left margin for notebook TOC overlays.
        fig.subplots_adjust(left=0.18, right=0.98, top=0.86, bottom=0.12)
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()


def plot_target_alignment(horizon: int = 1, lookback: int = 10) -> None:
    """Visualize temporal alignment between available features at t and forward target y_t^(h)."""
    if horizon < 1:
        raise ValueError("horizon must be a positive integer.")
    if lookback < 1:
        raise ValueError("lookback must be a positive integer.")

    set_style()
    fig, ax = plt.subplots(figsize=(12, 3.7))

    left = -lookback + 1
    right = horizon

    ax.axvspan(left, 0, color="#34495E", alpha=0.14, label="Feature window")
    ax.axvspan(0, right, color="#1F7A8C", alpha=0.14, label="Target horizon")
    ax.axvline(0, color="#2f2f2f", linestyle="--", linewidth=1.1)

    ax.plot([left, right], [0, 0], color="#4a4a4a", linewidth=1.4)
    ax.scatter([left, 0, right], [0, 0, 0], color="#1f1f1f", s=25, zorder=3)

    ax.text(left, 0.18, f"t-{lookback - 1}", ha="center", fontsize=10)
    ax.text(0, 0.2, "t (prediction timestamp)", ha="center", fontsize=10)
    ax.text(right, 0.18, f"t+{horizon}", ha="center", fontsize=10)
    ax.text(
        0.5 * (left + 0),
        -0.19,
        "X_t built only from information available up to t",
        ha="center",
        fontsize=10,
        color="#2f2f2f",
    )
    ax.text(
        0.5 * right,
        -0.19,
        r"$y_t^{(h)} = \log(P_{t+h}) - \log(P_t)$",
        ha="center",
        fontsize=10,
        color="#2f2f2f",
    )

    ax.set_ylim(-0.36, 0.34)
    ax.set_yticks([])
    ax.set_xlim(left - 0.8, right + 0.8)
    ax.set_xlabel("Time")
    ax.legend(loc="upper left", frameon=False)
    apply_academic_style(ax, "Temporal Alignment of Features and Forward Target")
    plt.tight_layout()
    plt.show()


def plot_return_distributions(
    df: pd.DataFrame | Dict[str, pd.DataFrame],
    title: str = "Log-Return Distributions by Ticker",
) -> None:
    """KDE of log-return distributions across tickers."""
    set_style()
    df = ensure_dataframe(df)
    
    if "Ticker" not in df.columns or "Log_Return" not in df.columns:
        print("Return distribution plot requires 'Ticker' and 'Log_Return' columns.")
        return
    plt.figure(figsize=(10, 5))
    for ticker, group in df.groupby("Ticker"):
        sns.kdeplot(group["Log_Return"].dropna(), label=ticker, color=COMPANY_COLORS.get(ticker, None))
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Daily Log Return")
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame | Dict[str, pd.DataFrame],
    method: str = "pearson",
) -> None:
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
    corr = pivot_ret.corr(method=method).round(2)
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(
        corr,
        annot=True,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.5,
        cbar_kws={"label": f"{method.title()} correlation"},
    )
    plt.title(f"Cross-Ticker Log-Return Correlation ({method.title()})", fontsize=16, fontweight="bold")
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
