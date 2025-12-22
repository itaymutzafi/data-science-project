import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

from src.config import COMPANY_COLORS

def add_ma_features(dfs: Dict[str, pd.DataFrame], windows: List[int]) -> None:
    added = []

    for name, df in dfs.items():
        if "Close" in df.columns:
            for win in windows:
                df[f"MA{win}"] = df['Close'].rolling(win).mean().bfill()
            added.append(name)

    print(f"{added}: Added Moving Average with windows {windows} features")

def add_macd_feature(dfs: Dict[str, pd.DataFrame]) -> None:
    added = []
    fast = 12
    slow = 26
    signal = 9

    for name, df in dfs.items():
        if "Close" in df.columns:
            ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
            ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

            df["MACD"] = ema_fast - ema_slow
            df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
            df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

            added.append(name)

    print(f"{added}: Added MACD feature")

def ma_plot(dfs: Dict[str, pd.DataFrame], window_size: int) -> None:
    plt.figure(figsize=(14, 6))
    ma_col = f"MA{window_size}"

    for name, df in dfs.items():
        plt.plot(df.index, df['Close'], label=f"{name} Close", color=COMPANY_COLORS[name], alpha=0.5, linewidth=1)
        if ma_col in df.columns:
            plt.plot(df.index, df[ma_col], label=f"{name} {ma_col}", color=COMPANY_COLORS[name], linestyle='--', alpha=0.7, linewidth=1)

    plt.title(f"Close Price and {ma_col}-day Moving Average - Comparison")
    plt.ylabel("Price")
    plt.xlabel("Year")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def macd_plot(dfs: Dict[str, pd.DataFrame]) -> None:
    plt.figure(figsize=(13, 5))

    for name, df in dfs.items():
        if not {"MACD", "MACD_Signal"}.issubset(df.columns):
            print(f"MACD columns missing in {name}")
            continue

        plt.plot(df.index, df["MACD"], color=COMPANY_COLORS[name], linewidth=2, linestyle="-", label=f"{name} MACD")
        plt.plot(df.index, df["MACD_Signal"], color=COMPANY_COLORS[name], linewidth=1.8, \
            linestyle="--", alpha=0.85, label=f"{name} Signal")

    plt.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("MACD & Signal")
    plt.xlabel("Date")
    plt.ylabel("MACD")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.show()
