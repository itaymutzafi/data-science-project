"""RSI feature engineering and visualization utilities."""

import matplotlib.pyplot as plt
import math
import pandas as pd
from typing import Dict

from src.config import COMPANY_COLORS
from src.utils import set_style


def add_rsi_feature(dfs: Dict[str, pd.DataFrame], window: int = 14) -> None:
    """Add RSI feature to each dataframe using Close prices."""
    for name, df in dfs.items():
        if "Close" in df.columns:
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

            rs = gain / loss
            df["RSI"] = 100 - (100 / (1 + rs))
            df["RSI"] = df["RSI"].fillna(50)
            dfs[name] = df

            print(f"{name}: Added RSI with window {window}")


def plot_rsi_grid(dfs: Dict[str, pd.DataFrame], lookback: int = 200) -> None:
    """Plot recent RSI and price for all tickers in a grid."""
    set_style()
    tickers = list(dfs.keys())
    n_tickers = len(tickers)
    cols = 2
    rows = math.ceil(n_tickers / cols)

    _, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows), sharex=True)
    if n_tickers == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, ticker in enumerate(tickers):
        ax = axes[i]
        df = dfs[ticker].iloc[-lookback:]
        color = COMPANY_COLORS.get(ticker, "#333333")

        l1 = ax.plot(df.index, df["Close"], label="Price", color=color, alpha=0.8)
        ax.set_title(f"{ticker}: RSI Divergence last {lookback} days", fontweight="bold")

        ax_rsi = ax.twinx()
        l2 = []
        if "RSI" in df.columns:
            l2 = ax_rsi.plot(df.index, df["RSI"], label="RSI", color="#E63946", linewidth=1)
            ax_rsi.axhline(70, color="red", linestyle=":", alpha=0.3)
            ax_rsi.axhline(30, color="green", linestyle=":", alpha=0.3)
            ax_rsi.fill_between(df.index, df["RSI"], 70, where=(df["RSI"] >= 70), color="red", alpha=0.1)
            ax_rsi.fill_between(df.index, df["RSI"], 30, where=(df["RSI"] <= 30), color="green", alpha=0.1)
            ax_rsi.set_ylim(0, 100)
            if i % cols == 1:
                ax_rsi.set_ylabel("RSI")

        if i % cols == 0:
            ax.set_ylabel("Price ($)")

        lns = l1 + l2
        labels = [line.get_label() for line in lns]
        ax.legend(lns, labels, loc="upper left", fontsize="small")
        ax.tick_params(axis="x", rotation=45)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
