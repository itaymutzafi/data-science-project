"""Plotting utilities for consistent academic visualization.

All plotting logic is centralized here to ensure uniform styling across the project.
"""

from pathlib import Path
from typing import Dict, List

from cycler import cycler
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf

from src.config import (
    COMPANY_COLORS,
    DAYNAMES,
    MONTHNAMES,
    TICKER,
    TICKERS,
    TICKER_TO_COMPANY_MAP,
)
from src.utils import statistic_tests as st

_STYLE_APPLIED = False
ACADEMIC_PALETTE = ["#1B263B", "#0A9396", "#EE9B00", "#CA6702", "#9B2226"]


def set_style() -> None:
    """Apply global plot styling once per session."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return

    sns.set_theme(style="whitegrid", context="paper", palette=ACADEMIC_PALETTE)
    plt.rcParams.update(
        {
            "figure.figsize": (12, 6),
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "lines.linewidth": 2.0,
            "grid.alpha": 0.25,
            "figure.dpi": 150,
            "axes.facecolor": "#fbfbfd",
            "axes.edgecolor": "#cccccc",
            "axes.prop_cycle": cycler(color=ACADEMIC_PALETTE),
        }
    )
    _STYLE_APPLIED = True


def _apply_academic_style(ax: plt.Axes, title: str | None = None) -> None:
    """Format axes with consistent academic styling."""
    if title:
        ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.set_facecolor("#fbfbfd")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def save_fig(fig: plt.Figure, filename: str, folder: str = "results") -> Path:
    """Save a figure to disk with consistent settings."""
    output_dir = Path(folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, bbox_inches="tight", dpi=plt.rcParams.get("figure.dpi", 150))
    return path


# --- Generic helpers ---
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


# --- Validation / methodology plots ---
def plot_walk_forward_validation(n_splits: int = 5, total_samples: int = 100) -> None:
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
    _apply_academic_style(ax, "Strict Walk-Forward Validation (Expanding Window)")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


# --- Price / returns / sentiment plots ---

def plot_context_comparison(data: pd.Series | Dict[str, pd.DataFrame], aux_data: pd.DataFrame, target_name: str = "Target") -> None:
    """Compare stock(s) vs macro/segment peers plus VIX.
    
    Args:
        data: Either a single Series (Close prices) or a Dict of DataFrames (df_s).
        aux_data: DataFrame containing 'Nasdaq_100', 'VIX_Index', 'NVIDIA_Segment_Leader'.
        target_name: Label for the single series if provided.
    """
    set_style()

    # Pre-process aux_data
    if not isinstance(aux_data.index, pd.DatetimeIndex):
        try:
            aux_data.index = pd.to_datetime(aux_data.index)
        except Exception:
            pass
    if aux_data.index.tz is not None:
        aux_data = aux_data.copy()
        aux_data.index = aux_data.index.tz_localize(None)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax1, ax2 = axes

    # -- Handle input data (Series vs Dict) --
    if isinstance(data, dict):
        # Plot multiple stocks
        stocks_to_plot = {}
        for ticker, df in data.items():
            if "Close" in df.columns:
                stocks_to_plot[ticker] = df["Close"]
        
        # Determine common start date for normalization fallback
        # We will normalize each stock to its first available point in the common range with aux_data
        
        for ticker, series in stocks_to_plot.items():
            # Align timezone
            if series.index.tz is not None:
                series = series.copy()
                series.index = series.index.tz_localize(None)
            
            # Intersection with aux info to ensure we plot on valid range
            common_idx = series.index.intersection(aux_data.index)
            if common_idx.empty:
                continue
            
            sub_series = series.loc[common_idx]
            # Normalize
            if not sub_series.empty:
                norm_series = (sub_series / sub_series.iloc[0]) * 100
                ax1.plot(norm_series, label=f"{ticker}", linewidth=2, color=COMPANY_COLORS.get(ticker, None))

    else:
        # Legacy: Single Series
        target_series = data
        if target_series.index.tz is not None:
            target_series = target_series.copy()
            target_series.index = target_series.index.tz_localize(None)
        
        common_idx = target_series.index.intersection(aux_data.index)
        if not common_idx.empty:
            target = target_series.loc[common_idx]
            norm_target = (target / target.iloc[0]) * 100
            ax1.plot(norm_target, label=f"{target_name} (Target)", linewidth=2.5, color=COMPANY_COLORS.get(target_name, "#1f77b4"))

    # -- Plot Benchmarks (Use the last common_idx or full aux_data range if possible) --
    # For simplicity, we plot benchmarks over the full range of aux_data that overlaps with data
    # But normalization requires a base point. We'll use the start of the aux_data window 
    # that matches the broad time period of our stocks.
    
    # Heuristic: slice aux_data to roughly the data range
    # Finding min/max dates from data input
    if isinstance(data, dict):
        all_dates = []
        for df in data.values():
             if not df.empty: all_dates.extend(df.index)
        if all_dates:
            min_date, max_date = min(all_dates), max(all_dates)
            if hasattr(min_date, 'tz') and min_date.tz: min_date = min_date.tz_localize(None)
            if hasattr(max_date, 'tz') and max_date.tz: max_date = max_date.tz_localize(None)
        else:
            min_date, max_date = aux_data.index.min(), aux_data.index.max()
    else:
        min_date = data.index.min()
        max_date = data.index.max()
        if hasattr(min_date, 'tz') and min_date.tz: min_date = min_date.tz_localize(None)
        if hasattr(max_date, 'tz') and max_date.tz: max_date = max_date.tz_localize(None)

    # Slice aux and normalize
    aux_slice = aux_data.loc[(aux_data.index >= min_date) & (aux_data.index <= max_date)]
    if not aux_slice.empty:
        norm_macro = (aux_slice["Nasdaq_100"] / aux_slice["Nasdaq_100"].iloc[0]) * 100
        norm_segment = (aux_slice["NVIDIA_Segment_Leader"] / aux_slice["NVIDIA_Segment_Leader"].iloc[0]) * 100

        ax1.plot(norm_macro, label="Nasdaq 100 (Macro Baseline)", linestyle="--", alpha=0.8, color="black")
        ax1.plot(norm_segment, label="NVIDIA (Segment Peer)", linestyle=":", alpha=0.8, color=COMPANY_COLORS.get("NVDA", "green"))
    
    ax1.set_title(f"Contextual Analysis: Stocks vs. Market & Segment", fontsize=12)
    ax1.set_ylabel("Normalized Price (Base=100)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # -- VIX Plot --
    color_vix = "tab:red"
    # Plot VIX over the same slice
    if not aux_slice.empty:
        ax2.plot(aux_slice["VIX_Index"], color=color_vix, label="VIX (Volatility Index)", linewidth=1.5)
        ax2.fill_between(aux_slice.index, aux_slice["VIX_Index"], alpha=0.1, color=color_vix)
    
    ax2.set_ylabel("VIX Index", color=color_vix)
    ax2.tick_params(axis="y", labelcolor=color_vix)
    ax2.set_title("Market Risk Sentiment (VIX)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()





def plot_sentiment_distribution(df: pd.DataFrame, ticker: str | None = None, sentiment_col: str = "sentiment_mean") -> None:
    """Plot distribution of sentiment scores for a single ticker dataframe."""
    set_style()
    if df.empty or sentiment_col not in df.columns:
        print(f"No sentiment column '{sentiment_col}' available to plot.")
        return
    series = df[sentiment_col].dropna()
    if series.empty:
        print("No sentiment data available after dropping NaNs.")
        return
    plt.figure(figsize=(8, 4))
    color = COMPANY_COLORS.get(ticker, "#1f77b4") if ticker else "#1f77b4"
    sns.histplot(series, bins=40, kde=True, color=color, alpha=0.8)
    plt.title(f"{ticker or 'Sentiment'} Distribution", fontsize=16)
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def plot_sentiment_label_distribution(df: pd.DataFrame | Dict[str, pd.DataFrame], sentiment_col: str = "sentiment_mean") -> None:
    """Bar chart of positive/neutral/negative counts from sentiment scores."""
    set_style()
    df = _ensure_dataframe(df)

    if df.empty or sentiment_col not in df.columns:
        print(f"No sentiment column '{sentiment_col}' available to plot.")
        return
    series = df[sentiment_col].dropna()
    if series.empty:
        print("No sentiment data available after dropping NaNs.")
        return
    labels = pd.cut(series, bins=[-np.inf, -0.1, 0.1, np.inf], labels=["Negative", "Neutral", "Positive"])
    counts = labels.value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, palette=["#1f77b4", "#aaaaaa", "#d62728"])
    plt.title("Sentiment Label Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_sentiment_vs_price(df: pd.DataFrame, sentiment_col: str = "sentiment_mean_lag1", days: int | None = None) -> None:
    """Plot price against sentiment series (expects lagged sentiment)."""
    set_style()
    if sentiment_col not in df.columns:
        print(f"Column '{sentiment_col}' not found. Skipping plot.")
        return

    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Stock Price", color="tab:blue")

    plot_data = df.iloc[-days:] if days else df
    title_suffix = f"(Last {days} Days)" if days else "(Full Period)"
    ax1.plot(plot_data.index, plot_data["Close"], color="tab:blue", label="Close Price")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Sentiment Score", color="tab:orange")
    ax2.plot(plot_data.index, plot_data[sentiment_col], color="tab:orange", linestyle="--", label="Sentiment", alpha=0.6)
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    title_prefix = df["Ticker"].iloc[0] if "Ticker" in df.columns and not df.empty else "Price"
    plt.title(f"{title_prefix} Price vs. Rolling Sentiment {title_suffix}")
    fig.tight_layout()
    plt.show()


def plot_sentiment_vs_price_grid(
    df: pd.DataFrame | Dict[str, pd.DataFrame],
    tickers: List[str],
    sentiment_col: str = "sentiment_mean_lag1",
    days: int | None = None,
) -> None:
    """2x2 grid overlaying price vs sentiment for multiple tickers (uses lagged sentiment)."""
    set_style()
    df = _ensure_dataframe(df)

    if df.empty:
        print("No data available to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
    axes = axes.flatten()

    # Allow graceful fallback if requested sentiment_col is missing
    sentiment_fallbacks = [sentiment_col, "sentiment_mean_lag1", "Sentiment_Score"]

    for ax, ticker in zip(axes, tickers):
        subset = df[df["Ticker"] == ticker].sort_index()
        if subset.empty:
            ax.text(0.5, 0.5, f"No data for {ticker}", ha="center", va="center")
            continue

        # Determine which sentiment column to use
        use_col = next((c for c in sentiment_fallbacks if c in subset.columns), None)
        if not use_col:
            ax.text(
                0.5,
                0.5,
                f"Missing sentiment for {ticker}",
                ha="center",
                va="center",
            )
            continue

        price_col = "Close" if "Close" in subset.columns else "Adj Close" if "Adj Close" in subset.columns else None
        if not price_col:
            ax.text(
                0.5,
                0.5,
                f"Missing price column for {ticker}",
                ha="center",
                va="center",
            )
            continue

        plot_data = subset.iloc[-days:] if days else subset
        ax.plot(plot_data.index, plot_data[price_col], color=COMPANY_COLORS.get(ticker, "#1f77b4"), label="Close")
        ax2 = ax.twinx()
        ax2.plot(plot_data.index, plot_data[use_col], color="tab:orange", linestyle="--", alpha=0.6, label="Sentiment")
        ax.set_title(f"{ticker}: Price vs Sentiment", fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Price vs Sentiment Across Tickers", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_sentiment_signal(df: pd.DataFrame, ticker: str, sentiment_col: str = "sentiment_mean_lag1", days: int | None = None) -> None:
    """Wrapper to plot sentiment vs price for a single ticker using lagged sentiment."""
    subset = df[df["Ticker"] == ticker].copy()
    if subset.empty:
        print(f"No data for ticker {ticker}")
        return
    plot_sentiment_vs_price(subset, sentiment_col=sentiment_col, days=days)


def plot_sentiment_summary(
    df: pd.DataFrame,
    ticker_col: str = "Ticker",
    sentiment_col: str = "sentiment_mean_lag1",
    window: int = 14,
) -> None:
    """Plot smoothed sentiment trends across tickers (expects lagged sentiment)."""
    set_style()
    if df.empty or sentiment_col not in df.columns or ticker_col not in df.columns:
        print("Sentiment summary requires sentiment and ticker columns.")
        return

    df_sorted = df.copy()
    if not isinstance(df_sorted.index, pd.DatetimeIndex):
        df_sorted.index = pd.to_datetime(df_sorted.index)

    pivot = df_sorted.pivot_table(index=df_sorted.index, columns=ticker_col, values=sentiment_col)
    smooth = pivot.rolling(window=window, min_periods=1).mean()

    plt.figure(figsize=(12, 6))
    for col in smooth.columns:
        plt.plot(smooth.index, smooth[col], label=col, color=COMPANY_COLORS.get(col, None), alpha=0.9)

    plt.title(f"Decay-Smoothed Sentiment Trend (window={window} days)", fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Sentiment Score")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_market_overview(df: pd.DataFrame | Dict[str, pd.DataFrame]) -> None:
    """2x2 grid of adjusted close prices for all tickers."""
    set_style()
    df = _ensure_dataframe(df)

    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if "Ticker" not in df.columns or price_col not in df.columns:
        print("Market overview requires 'Ticker' and price columns.")
        return
    pivot_close = df.pivot_table(index=df.index, columns="Ticker", values=price_col)
    tickers = sorted(pivot_close.columns)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.flatten()
    for ax, ticker in zip(axes, tickers):
        series = pivot_close[ticker].dropna()
        ax.plot(series.index, series.values, color=COMPANY_COLORS.get(ticker, None))
        ax.set_title(f"{ticker} Close", fontsize=12)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Market Overview: Close Prices", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame | Dict[str, pd.DataFrame]) -> None:
    """Correlation heatmap of log returns across tickers."""
    set_style()
    df = _ensure_dataframe(df)

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


def plot_correlation_matrix(df: pd.DataFrame, tickers: List[str] | None = None) -> None:
    """Alias for correlation heatmap with optional ticker filter for legacy calls."""
    if tickers:
        df = df[df["Ticker"].isin(tickers)]
    plot_correlation_heatmap(df)


def plot_return_grid(df: pd.DataFrame | Dict[str, pd.DataFrame]) -> None:
    """2x2 grid of log returns for all tickers."""
    set_style()
    df = _ensure_dataframe(df)

    if "Ticker" not in df.columns or "Log_Return" not in df.columns:
        print("Return grid requires 'Ticker' and 'Log_Return' columns.")
        return
    pivot_ret = df.pivot_table(index=df.index, columns="Ticker", values="Log_Return")
    tickers = sorted(pivot_ret.columns)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.flatten()
    for ax, ticker in zip(axes, tickers):
        series = pivot_ret[ticker].dropna()
        ax.plot(series.index, series.values, color=COMPANY_COLORS.get(ticker, None))
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(f"{ticker} Log Returns", fontsize=12)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Log Returns by Ticker", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def _ensure_dataframe(data: pd.DataFrame | Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Helper: Normalize input to a single DataFrame with a Ticker column."""
    if isinstance(data, dict):
        processed_frames = []
        for ticker, df in data.items():
            temp = df.copy()
            # Ensure datetime index and remove timezone for consistent plotting/merging
            if not isinstance(temp.index, pd.DatetimeIndex):
                temp.index = pd.to_datetime(temp.index)
            if temp.index.tz is not None:
                temp.index = temp.index.tz_localize(None)

            temp["Ticker"] = ticker
            processed_frames.append(temp)
        return pd.concat(processed_frames, axis=0) if processed_frames else pd.DataFrame()
    return data


def plot_return_distributions(df: pd.DataFrame | Dict[str, pd.DataFrame]) -> None:
    """KDE of log-return distributions across tickers."""
    set_style()
    df = _ensure_dataframe(df)
    
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


def plot_sentiment_trends(daily_sentiment_df: pd.DataFrame) -> None:
    """Plot raw and smoothed sentiment trends for the four companies."""
    set_style()
    if daily_sentiment_df.empty:
        print("No sentiment data available to plot.")
        return

    df = daily_sentiment_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "sentiment_ma_7d" not in df.columns:
        df["sentiment_ma_7d"] = df.groupby("company")["sentiment_mean"].transform(lambda x: x.rolling(7, min_periods=1).mean())

    companies = ["Apple", "Amazon", "Google", "Microsoft"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    grid_map = {"Apple": 0, "Amazon": 1, "Google": 2, "Microsoft": 3}

    for company in companies:
        if company not in grid_map:
            continue
        ax = axes_flat[grid_map[company]]
        subset = df[df["company"] == company].sort_values("date")
        if subset.empty:
            ax.text(0.5, 0.5, "No Data Available", ha="center", va="center")
            ax.set_title(company, fontweight="bold")
            continue

        color = COMPANY_COLORS.get(company, "blue")
        raw_days = subset[subset["news_count"] > 0]
        if not raw_days.empty:
            ax.scatter(raw_days["date"], raw_days["sentiment_mean"], color=color, alpha=0.3, s=15, label="Daily Raw Sentiment")

        trend_data = subset.dropna(subset=["sentiment_ma_7d"])
        if not trend_data.empty:
            ax.plot(trend_data["date"], trend_data["sentiment_ma_7d"], color=color, linewidth=2.5, label="7-Day Moving Avg")

        ax.set_title(f"{company}", fontweight="bold", fontsize=12)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
        if grid_map[company] == 0:
            ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.suptitle("Sentiment Trends: Raw Signals vs. Smoothed Trends", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()


def plot_sentiment_decay_verification() -> None:
    """Visualize the exponential decay logic on synthetic data."""
    set_style()
    from src.features.sentiment_analysis import process_sentiment_timeseries

    demo_dates = pd.date_range("2024-01-01", periods=15)
    demo_df = pd.DataFrame({"date": demo_dates, "company": "Demo", "sentiment_mean": np.nan, "news_count": 0})
    demo_df.loc[0, "sentiment_mean"] = 1.0
    demo_df.loc[0, "news_count"] = 5
    demo_df.loc[5, "sentiment_mean"] = -0.8
    demo_df.loc[5, "news_count"] = 3

    processed_demo = process_sentiment_timeseries(demo_df)

    plt.figure(figsize=(10, 5))
    plt.plot(processed_demo["date"], processed_demo["sentiment_mean"], marker="o", label="Decayed Sentiment", color="purple")
    events = demo_df.dropna(subset=["sentiment_mean"])
    plt.scatter(events["date"], events["sentiment_mean"], color="red", s=100, label="News Events", zorder=5)
    plt.title("Verification: Exponential Decay of Sentiment Signal ($S_t = S_{t-1} \\times 0.85$)", fontweight="bold")
    plt.ylabel("Sentiment Score")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_sentiment_signal_quality(daily_sentiment_df: pd.DataFrame, company: str = "Apple", ax=None) -> None:
    """Plot decayed vs. raw sentiment points for a single company."""
    set_style()
    if daily_sentiment_df.empty:
        return

    subset = daily_sentiment_df[daily_sentiment_df["company"] == company].copy().sort_values("date")
    if subset.empty:
        return

    subset = subset.iloc[-180:]
    main_color = COMPANY_COLORS.get(company, "#34A853")
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    raw_points = subset[subset["news_count"] > 0]
    ax.plot(subset["date"], subset["sentiment_mean"], color=main_color, linestyle="-", alpha=0.6, label="Decayed Signal")
    ax.scatter(raw_points["date"], raw_points["sentiment_mean"], color=main_color, s=25, label="Raw News Days", zorder=5)
    ax.set_title(f"{company}", fontweight="bold")
    ax.grid(True, alpha=0.3)


def plot_day_sentiment_breakdown(daily_news_df: pd.DataFrame, date: str, company: str, daily_score: float) -> None:
    """Show sentiment distribution and key headlines for a single day."""
    set_style()
    if daily_news_df.empty:
        print(f"No news found for {company} on {date}")
        return

    scores = daily_news_df["sentiment_score"].values
    fig = plt.figure(figsize=(14, 7))
    grid = plt.GridSpec(1, 2, width_ratios=[1.2, 1])

    ax1 = fig.add_subplot(grid[0])
    sns.histplot(scores, kde=True, ax=ax1, color="skyblue", bins=10)
    ax1.axvline(daily_score, color="red", linestyle="--", linewidth=2, label=f"Daily Mean: {daily_score:.2f}")
    ax1.set_title(f"Sentiment Distribution: {company} on {date}", fontweight="bold")
    ax1.set_xlabel("Sentiment Score (-1 to 1)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(grid[1])
    ax2.axis("off")
    daily_news_df = daily_news_df.drop_duplicates(subset=["headline"]).sort_values(by="sentiment_score", ascending=False)
    top_pos = daily_news_df.head(3)
    top_neg = daily_news_df.tail(3)

    import textwrap

    text_content = f"### Key Headlines ({len(daily_news_df)} Total)\n\n"
    text_content += "**Most Positive:**\n"
    for _, row in top_pos.iterrows():
        short_headline = textwrap.shorten(row["headline"], width=60, placeholder="...")
        text_content += f"(+{row['sentiment_score']:.2f}) {short_headline}\n"

    text_content += "\n**Most Negative:**\n"
    for _, row in top_neg.iterrows():
        short_headline = textwrap.shorten(row["headline"], width=60, placeholder="...")
        text_content += f"(-{abs(row['sentiment_score']):.2f}) {short_headline}\n"

    ax2.text(0.05, 0.95, text_content, fontsize=10, va="top", ha="left", family="monospace")
    plt.tight_layout()
    plt.show()


def plot_advanced_sentiment_features(daily_sentiment_df: pd.DataFrame) -> None:
    """Grid plot of decayed vs. raw sentiment across companies."""
    set_style()
    if daily_sentiment_df.empty:
        print("No data to plot.")
        return

    companies = ["Apple", "Amazon", "Google", "Microsoft"]
    available = [c for c in companies if c in daily_sentiment_df["company"].unique()]
    if not available:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, company in enumerate(available):
        plot_sentiment_signal_quality(daily_sentiment_df, company, ax=axes[i])
    from matplotlib.lines import Line2D

    fig.suptitle("Feature Engineering: Signal Continuity (Filling Missing Days with Decay)", fontsize=16, fontweight="bold", y=0.95)
    legend_elements = [
        Line2D([0], [0], color="green", lw=2, label="Decayed Signal (Forward Filled)"),
        Line2D([0], [0], marker="o", color="w", label="Raw Sentiment", markerfacecolor="green", markersize=8),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.92))
    plt.tight_layout(rect=[0, 0, 1, 0.90])
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


# --- Technical indicator visualizations ---
def return_plot(dfs: Dict[str, pd.DataFrame]) -> None:
    """Plot return seasonality by day and month plus box plots."""
    set_style()
    if all("Return" in df.columns for df in dfs.values()):
        print("\n3. Return Seasonality - Daily (All Companies)")
        plt.figure(figsize=(12, 6))
        x = np.arange(len(DAYNAMES))
        width = 0.2
        multiplier = 0

        for name, df in dfs.items():
            daily_ret = df.groupby("Day")["Return"].mean().reindex(DAYNAMES)
            offset = width * multiplier
            plt.bar(x + offset, daily_ret.values, width, label=name, color=COMPANY_COLORS[name], alpha=0.8)
            multiplier += 1

        plt.xlabel("Day of Week")
        plt.ylabel("Avg Return")
        plt.title("Average Return by Day of Week - Comparison")
        plt.xticks(x + width * (len(TICKERS) - 1) / 2, DAYNAMES, rotation=45)
        plt.legend()
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

        print("4. Return Seasonality - Monthly (All Companies)")
        plt.figure(figsize=(14, 6))
        x = np.arange(len(MONTHNAMES))
        width = 0.2
        multiplier = 0
        for name, df in dfs.items():
            monthly_ret = df.groupby("Month")["Return"].mean().reindex(range(1, 13))
            offset = width * multiplier
            plt.bar(x + offset, monthly_ret.values, width, label=name, color=COMPANY_COLORS[name], alpha=0.8)
            multiplier += 1

        plt.xlabel("Month")
        plt.ylabel("Avg Return")
        plt.title("Average Return by Month - Comparison")
        plt.xticks(x + width * (len(TICKERS) - 1) / 2, MONTHNAMES, rotation=45)
        plt.legend()
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

        print("\n9. Statistical Seasonality Tests")
        for name, df in dfs.items():
            if "Return" in df.columns:
                print(f"\n{name}:")
                st.display_seasonality_results(st.test_seasonality(df, "Return", "Day"))
                st.display_seasonality_results(st.test_seasonality(df, "Return", "Month"))


def volatility(dfs: Dict[str, pd.DataFrame]) -> None:
    """Plot rolling volatility and seasonal breakdowns."""
    set_style()
    if all("Return" in df.columns for df in dfs.values()):
        print("7. Volatility Analysis (All Companies)")
        for df in dfs.values():
            df["Vol20"] = df["Return"].rolling(20).std()

        plt.figure(figsize=(14, 6))
        for name, df in dfs.items():
            if "Vol20" in df.columns:
                plt.plot(df.index, df["Vol20"], label=f"{name} (20-day)", color=COMPANY_COLORS[name], alpha=0.8, linewidth=1.5)
        plt.title("Rolling Volatility (20-day) - Comparison")
        plt.ylabel("Volatility (Std Dev)")
        plt.xlabel("Year")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        print("Volatility by Day of Week (All Companies)")
        plt.figure(figsize=(12, 6))
        x = np.arange(len(DAYNAMES))
        width = 0.2
        multiplier = 0
        for name, df in dfs.items():
            daily_vol20 = df.groupby("Day")["Vol20"].mean().reindex(DAYNAMES)
            offset = width * multiplier
            plt.bar(x + offset, daily_vol20.values, width, label=name, color=COMPANY_COLORS[name], alpha=0.8)
            multiplier += 1

        plt.xlabel("Day of Week")
        plt.ylabel("Avg Volatility (20-day)")
        plt.title("Average Volatility by Day of Week - Comparison")
        plt.xticks(x + width * (len(TICKERS) - 1) / 2, DAYNAMES, rotation=45)
        plt.legend()
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

        print("Volatility by Month (All Companies)")
        plt.figure(figsize=(14, 6))
        x = np.arange(len(MONTHNAMES))
        width = 0.2
        multiplier = 0
        for name, df in dfs.items():
            monthly_vol20 = df.groupby("Month")["Vol20"].mean().reindex(range(1, 13))
            offset = width * multiplier
            plt.bar(x + offset, monthly_vol20.values, width, label=name, color=COMPANY_COLORS[name], alpha=0.8)
            multiplier += 1

        plt.xlabel("Month")
        plt.ylabel("Avg Volatility (20-day)")
        plt.title("Average Volatility by Month - Comparison")
        plt.xticks(x + width * (len(TICKERS) - 1) / 2, MONTHNAMES, rotation=45)
        plt.legend()
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()


def moving_average(dfs: Dict[str, pd.DataFrame]) -> None:
    """Plot close price with short moving averages."""
    set_style()
    if all("Close" in df.columns for df in dfs.values()):
        for df in dfs.values():
            df["MA20"] = df["Close"].rolling(20).mean()
            df["MA50"] = df["Close"].rolling(50).mean()

        plt.figure(figsize=(14, 6))
        for name, df in dfs.items():
            plt.plot(df.index, df["Close"], label=f"{name} Close", color=COMPANY_COLORS[name], alpha=0.5, linewidth=1)
            if "MA20" in df.columns:
                plt.plot(df.index, df["MA20"], label=f"{name} MA20", color=COMPANY_COLORS[name], linestyle="--", alpha=0.7, linewidth=1)
        plt.title("Close Price and 20-day Moving Average - Comparison")
        plt.ylabel("Price")
        plt.xlabel("Year")
        plt.legend(ncol=2, fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_corrletion_companies(df_s: Dict[str, pd.DataFrame]) -> None:
    """Heatmap of Apple vs peer correlations."""
    set_style()
    cols = [
        "Close",
        "Volume",
        "MSFT - Close",
        "MSFT - Volume",
        "GOOG - Close",
        "GOOG - Volume",
        "AMZN - Close",
        "AMZN - Volume",
    ]
    corr_df = df_s["AAPL"][cols].corr()
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Between APPLE and other stocks")
    plt.tight_layout()
    plt.show()


# --- Filings / reports helpers ---
def create_days_to_report(df: pd.DataFrame, report_dates: List) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    report_dates = pd.to_datetime(report_dates)
    filings_df = pd.DataFrame({"filing_date": report_dates}).sort_values("filing_date")
    nearest = pd.merge_asof(df, filings_df, left_index=True, right_on="filing_date", direction="nearest")
    nearest.index = df.index
    df["Days To Nearest Report"] = (nearest["filing_date"] - nearest.index).dt.days
    df["Days To Nearest Report"] = df["Days To Nearest Report"].fillna(np.inf)
    return df


def create_reports_dic() -> Dict:
    reports_by_company = {}
    for company_name in TICKERS:
        ticker = yf.Ticker(company_name)
        reports_by_company[company_name] = ticker.get_sec_filings()
    return reports_by_company


def plot_sec_fiilings_dates(reports_by_company: Dict) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, ticker in enumerate(TICKERS):
        filings_list = reports_by_company[ticker]
        filings_df = pd.DataFrame(filings_list)
        filings_df["date"] = pd.to_datetime(filings_df["date"])
        company_name = TICKER_TO_COMPANY_MAP.get(ticker, ticker)
        color = COMPANY_COLORS.get(company_name, COMPANY_COLORS.get(ticker, None))
        ax.scatter(filings_df["date"], [i] * len(filings_df), label=company_name, color=color, alpha=0.7, s=20)

    ax.set_yticks(range(len(TICKERS)))
    ax.set_yticklabels([TICKER_TO_COMPANY_MAP[t] for t in TICKERS])
    ax.set_xlabel("Filing Date")
    ax.set_title("SEC Filing Dates per Company")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.show()


def reports(dfs: Dict[str, pd.DataFrame]) -> None:
    set_style()
    reports_by_company = create_reports_dic()
    plot_sec_fiilings_dates(reports_by_company)

    for company, df in dfs.items():
        reports_list = reports_by_company[company]
        reports_df = pd.DataFrame(reports_list)
        report_dates = reports_df["date"]
        dfs[company] = create_days_to_report(df, report_dates)


# --- Evaluation plots ---
def plot_performance_comparison(results_df: pd.DataFrame, metric: str = "MSE") -> None:
    """Bar chart of model performance for a single metric."""
    set_style()
    if results_df.empty:
        return

    plt.figure(figsize=(10, 6))
    sorted_df = results_df.sort_values(metric, ascending=(metric not in ["R2", "Sharpe", "Directional Accuracy"]))
    colors = [
        "#1f77b4" if "LSTM" in idx else "#d62728" if "Market" in idx or "Baseline" in idx else "gray"
        for idx in sorted_df.index
    ]
    bars = plt.bar(sorted_df.index, sorted_df[metric], color=colors, alpha=0.8)
    _apply_academic_style(plt.gca(), f"Model Comparison: {metric}")
    plt.ylabel(metric)
    plt.xlabel("Model strategy")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.4f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.show()


def plot_cumulative_returns(y_true: pd.Series, model_predictions: Dict[str, pd.Series]) -> None:
    """Plot cumulative equity curves for strategies derived from model predictions."""
    set_style()
    plt.figure(figsize=(12, 6))
    market_curve = np.exp(y_true.cumsum())
    plt.plot(market_curve.index, market_curve, label="Market (Buy & Hold)", color="black", linewidth=2, linestyle="--")

    for name, y_pred in model_predictions.items():
        common_idx = y_true.index.intersection(y_pred.index)
        if common_idx.empty:
            continue
        truth = y_true.loc[common_idx]
        pred = y_pred.loc[common_idx]
        strat_returns = np.sign(pred) * truth
        strat_curve = np.exp(strat_returns.cumsum())
        color = None
        if "LSTM" in name:
            color = "#1f77b4"
        elif "Baseline" in name:
            color = "gray"
        plt.plot(common_idx, strat_curve, label=f"{name} Strategy", linewidth=2 if "LSTM" in name else 1.5, color=color)

    _apply_academic_style(plt.gca(), "Equity Curve Comparison (Cumulative Returns)")
    plt.xlabel("Date")
    plt.ylabel("Normalized Wealth (Start=1.0)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# --- Higher-level convenience plots ---
def plot_close_comparison(dataset: pd.DataFrame) -> None:
    """Plot close prices for all tickers in the dataset."""
    set_style()
    pivot_close = dataset.pivot_table(index=dataset.index, columns="Ticker", values="Close")
    plt.figure(figsize=(12, 6))
    for col in pivot_close.columns:
        plt.plot(pivot_close.index, pivot_close[col], label=col, color=COMPANY_COLORS.get(col, None))
    plt.title("Close Price Comparison Over Time", fontsize=18)
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_bollinger_bands(df: pd.DataFrame, ticker: str) -> None:
    """Plot price with Bollinger Bands for a single ticker dataframe."""
    set_style()
    df_plot = df[["Close", "BB_Upper", "BB_Lower"]].dropna()
    color = COMPANY_COLORS.get(ticker, "#1f77b4")
    plt.figure(figsize=(12, 6))
    plt.plot(df_plot.index, df_plot["Close"], label=f"{ticker} Close", color=color)
    plt.plot(df_plot.index, df_plot["BB_Upper"], label="Bollinger Upper", color="gray", linestyle="--", alpha=0.8)
    plt.plot(df_plot.index, df_plot["BB_Lower"], label="Bollinger Lower", color="gray", linestyle="--", alpha=0.8)
    plt.fill_between(df_plot.index, df_plot["BB_Lower"], df_plot["BB_Upper"], color="gray", alpha=0.1)
    plt.title(f"{ticker}: Price with Bollinger Bands", fontsize=18)
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rolling_volatility(df: pd.DataFrame, ticker: str, window: int = 20) -> float:
    """Plot rolling volatility of log returns and return sentiment correlation."""
    set_style()
    df_vol = df.sort_index().copy()
    df_vol["RollingVol"] = df_vol["Log_Return"].rolling(window).std()
    vol_color = COMPANY_COLORS.get(ticker, "#1f77b4")
    plt.figure(figsize=(12, 5))
    plt.plot(df_vol.index, df_vol["RollingVol"], label=f"{window}-day Rolling Volatility", color=vol_color)
    plt.title(f"{ticker}: Rolling Volatility of Log Returns", fontsize=18)
    plt.ylabel("Volatility (std)")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.show()
    corr = df_vol[["Log_Return", "sentiment_mean"]].dropna().corr().iloc[0, 1]
    return float(corr)


def plot_pred_vs_actual(y_true: pd.Series, predictions: Dict[str, pd.Series], title: str, last_n: int = 200) -> None:
    """Plot predicted vs actual returns over a recent window for multiple models."""
    if not predictions:
        return
    set_style()
    aligned_idx = y_true.index
    zoom_idx = aligned_idx[-last_n:] if len(aligned_idx) > last_n else aligned_idx
    plt.figure(figsize=(12, 5))
    plt.plot(zoom_idx, y_true.loc[zoom_idx], label="Actual", color="black")
    for name, preds in predictions.items():
        idx = zoom_idx.intersection(preds.index)
        plt.plot(idx, preds.loc[idx], label=f"Predicted ({name})")
    _apply_academic_style(plt.gca(), title)
    plt.ylabel("Log Return")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_predictions(y_true: pd.Series, predictions: Dict[str, pd.Series], title: str = "Predictions vs Actual", last_n: int = 200) -> None:
    """Alias for plot_pred_vs_actual for convenience."""
    plot_pred_vs_actual(y_true, predictions, title, last_n)


def plot_feature_importance(importances: pd.Series, top_k: int = 5, ticker: str | None = None) -> None:
    """Plot top-k permutation importances."""
    if importances.empty:
        return
    set_style()
    top_vals = importances.head(top_k)
    plt.figure(figsize=(8, 4))
    bar_color = COMPANY_COLORS.get(ticker, "#1f77b4") if ticker else "#1f77b4"
    plt.barh(top_vals.index[::-1], top_vals.values[::-1], color=bar_color)
    _apply_academic_style(plt.gca(), "Permutation Importance (Validation)")
    plt.xlabel("MSE Increase")
    plt.tight_layout()
    plt.show()


def plot_loss_curve(losses: List[float], ticker: str | None = None) -> None:
    """Plot training loss per epoch."""
    if not losses:
        return
    set_style()
    color = COMPANY_COLORS.get(ticker, "#1f77b4") if ticker else "#1f77b4"
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label="Training Loss", color=color)
    plt.title("Training Loss per Epoch", fontsize=16)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_accuracy_comparison(results: pd.DataFrame) -> None:
    """Grouped bar chart comparing baseline vs LSTM directional accuracy across tickers."""
    if results.empty:
        print("No results to plot.")
        return
    # Accept flexible column names; try to map
    cols = list(results.columns)
    if "NaiveBaseline" in cols and "LSTM" in cols:
        df_plot = results[["NaiveBaseline", "LSTM"]].rename(columns={"NaiveBaseline": "Baseline", "LSTM": "LSTM"})
    elif "Baseline DA" in cols and "LSTM DA" in cols:
        df_plot = results[["Baseline DA", "LSTM DA"]].rename(columns={"Baseline DA": "Baseline", "LSTM DA": "LSTM"})
    else:
        print("Results must include baseline and LSTM accuracy columns.")
        return

    set_style()
    ax = df_plot.plot(kind="bar", figsize=(10, 5), color=["#999999", COMPANY_COLORS.get("AAPL", "#1f77b4")])
    ax.set_ylabel("Directional Accuracy")
    _apply_academic_style(ax, "Directional Accuracy by Ticker: Baseline vs LSTM")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_metric_heatmap(data: pd.DataFrame, title: str = "Model Performance Heatmap") -> None:
    """Heatmap for model screening metrics (rows=models, cols=metrics)."""
    if data.empty:
        print("No data to plot.")
        return
    set_style()
    plt.figure(figsize=(8, 5))
    sns.heatmap(data, annot=True, fmt=".3f", cmap="YlGnBu")
    _apply_academic_style(plt.gca(), title)
    plt.tight_layout()
    plt.show()


def plot_model_leaderboard(results_df: pd.DataFrame) -> None:
    """Professional leaderboard heatmap for model screening."""
    if results_df.empty:
        print("No results to plot.")
        return
    set_style()
    plt.figure(figsize=(10, 6))
    sns.heatmap(results_df, annot=True, fmt=".3f", cmap="RdYlGn", cbar=True, linewidths=0.5)
    _apply_academic_style(plt.gca(), "Model Performance Leaderboard (Walk-Forward Validation)")
    plt.xlabel("Metrics")
    plt.ylabel("Models")
    plt.tight_layout()
    plt.show()
