"""Moving-average and momentum feature utilities."""

import pandas as pd
import matplotlib.pyplot as plt
import math
from typing import Dict, List

from src.config import COMPANY_COLORS, FEATURE_WINDOWS
from src.utils import set_style


def add_ma_features(dfs: Dict[str, pd.DataFrame], windows: List[int] = FEATURE_WINDOWS) -> None:
    """Add moving-average columns for configured windows."""
    added = []

    for name, df in dfs.items():
        if "Close" in df.columns:
            for win in windows:
                col_name = f"MA{win}"
                df[col_name] = df["Close"].rolling(window=win).mean()

            added.append(name)

    print(f"Added Moving Average features: {windows} for {len(added)} stocks.")
    print(f"Note: First {max(windows)} rows will contain NaNs (Warm-up Period).")


def add_ma_distance_features(dfs: Dict[str, pd.DataFrame], windows: List[int] = FEATURE_WINDOWS) -> List[str]:
    """Add normalized distance-from-MA features for each window."""
    columns = [f"Dist_MA{win}" for win in windows]

    for name, df in dfs.items():
        for win in windows:
            ma_col = f"MA{win}"
            if ma_col not in df.columns:
                df[ma_col] = df["Close"].rolling(window=win).mean()

            col_name = f"Dist_MA{win}"
            df[col_name] = (df["Close"] - df[ma_col]) / df[ma_col]
            df[col_name] = df[col_name].fillna(0)

        dfs[name] = df
        print(f"{name}: Adding Features: Dist_MA with windows {windows}")

    return columns


def add_macd_feature(dfs: Dict[str, pd.DataFrame]) -> None:
    """Add MACD, signal, and histogram features."""
    added = []
    fast = 12
    slow = 26
    signal = 9

    global MOMENTUM_FEATURES

    for name, df in dfs.items():
        if "Close" in df.columns:
            ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
            ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

            df["MACD"] = ema_fast - ema_slow

            df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()

            df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

            added.append(name)

    print(f"{added}: Added MACD feature")


def ma_plot(dfs: Dict[str, pd.DataFrame], window_sizes: List[int] = FEATURE_WINDOWS) -> None:
    """Plot Close and moving averages for all tickers."""
    num_plots = len(dfs)
    if num_plots == 0:
        return

    cols = 2
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(14, 7 * rows))
    axes = axes.flatten()

    for i, (name, df) in enumerate(dfs.items()):
        ax = axes[i]

        color = COMPANY_COLORS.get(name) or COMPANY_COLORS.get(name.upper(), "blue")
        ax.plot(df.index, df["Close"], label=f"{name} Close", color=color, alpha=0.9, linewidth=1.5)

        for win in window_sizes:
            ma_col = f"MA{win}"
            if ma_col in df.columns:
                linestyle = "--" if win < 100 else "-"
                width = 1.2 if win < 100 else 1.8
                alpha = 0.7

                ax.plot(df.index, df[ma_col], label=f"MA {win}", linestyle=linestyle, linewidth=width, alpha=alpha)

        ax.set_title(f"{name} - Price Trend", fontsize=12, fontweight="bold")
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Price ($)", fontsize=10)
        ax.legend(loc="upper left", fontsize=8, frameon=True)
        ax.grid(True, alpha=0.15, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def macd_plot(dfs: Dict[str, pd.DataFrame]) -> None:
    """Plot MACD and signal lines for all tickers."""
    plt.figure(figsize=(13, 5))

    for name, df in dfs.items():
        if not {"MACD", "MACD_Signal"}.issubset(df.columns):
            print(f"MACD columns missing in {name}")
            continue

        plt.plot(df.index, df["MACD"], color=COMPANY_COLORS[name], linewidth=2, linestyle="-", label=f"{name} MACD")
        plt.plot(
            df.index,
            df["MACD_Signal"],
            color=COMPANY_COLORS[name],
            linewidth=1.8,
            linestyle="--",
            alpha=0.85,
            label=f"{name} Signal",
        )

    plt.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("MACD & Signal")
    plt.xlabel("Date")
    plt.ylabel("MACD")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_ma_distance_grid(dfs: Dict[str, pd.DataFrame], window: int) -> None:
    """Plot MA-distance trends for all tickers in a grid."""
    set_style()
    tickers = list(dfs.keys())
    n_tickers = len(tickers)
    cols = 2
    rows = math.ceil(n_tickers / cols)

    dist_col = f"Dist_MA{window}"

    _, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows), sharex=True)
    if n_tickers == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, ticker in enumerate(tickers):
        ax = axes[i]
        df = dfs[ticker]
        color = COMPANY_COLORS.get(ticker, "purple")

        if dist_col in df.columns:
            ax.plot(df.index, df[dist_col], color=color, linewidth=1, alpha=0.9)
            ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.set_title(f"{ticker}: {window}-Day Trend Dist", fontweight="bold")
            if i % cols == 0:
                ax.set_ylabel("Dist (%)")

    for j in range(len(tickers), len(axes)):
        axes[j].axis("off")

    for ax in axes:
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()
