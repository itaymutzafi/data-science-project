import pandas as pd
import matplotlib.pyplot as plt
import math
from typing import Dict, List

from src.config import COMPANY_COLORS, FEATURE_WINDOWS

def add_ma_features(dfs: Dict[str, pd.DataFrame], windows: List[int] = FEATURE_WINDOWS) -> None:
    """
    Adds Moving Average features to the dataframe.
    
    Args:
        dfs: Dictionary of DataFrames.
        windows: List of window sizes. Defaults to FEATURE_WINDOWS from config.
    """
    added = []
    
    for name, df in dfs.items():
        if "Close" in df.columns:
            for win in windows:
                col_name = f"MA{win}"
                # Standard causal rolling mean
                df[col_name] = df['Close'].rolling(window=win).mean()
                
            added.append(name)

    print(f"Added Moving Average features: {windows} for {len(added)} stocks.")
    print(f"Note: First {max(windows)} rows will contain NaNs (Warm-up Period).")


def add_ma_distance_features(dfs: Dict[str, pd.DataFrame], windows: List[int] = FEATURE_WINDOWS) -> None:
    """
    Adds Normalized Distance from Moving Averages.
    Formula: (Price - MA) / MA
    
    Dynamically generates columns based on config (e.g. 'Dist_MA50', 'Dist_MA200').
    """
    print(f"Adding Features: Dist_MA {windows}...")
    count = 0
    for name, df in dfs.items():
        for win in windows:
            ma_col = f'MA{win}'
            # Ensure baseline MA exists
            if ma_col not in df.columns:
                df[ma_col] = df['Close'].rolling(window=win).mean()
            
            # Calculate Normalized Distance
            # Result is a percentage (e.g., 0.05 = 5% above trend)
            col_name = f'Dist_MA{win}'
            df[col_name] = (df['Close'] - df[ma_col]) / df[ma_col]
            
            # Fill NaNs
            df[col_name] = df[col_name].fillna(0)
        
        dfs[name] = df
        count += 1
    print(f" -> Success: Distances added to {count} tickers.")

def add_macd_feature(dfs: Dict[str, pd.DataFrame]) -> None:
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
    """
    Plots Close price against multiple Moving Averages for all companies in a 2x2 grid.
    Uses specific company colors for the Close price.
    """    
    num_plots = len(dfs)
    if num_plots == 0:
        return

    cols = 2
    rows = math.ceil(num_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 7 * rows))
    axes = axes.flatten()  # Flatten to easy 1D indexing

    for i, (name, df) in enumerate(dfs.items()):
        ax = axes[i]
        
        # Determine Color
        color = COMPANY_COLORS.get(name) or COMPANY_COLORS.get(name.upper(), 'blue')

        # Plot Close Price
        ax.plot(df.index, df['Close'], label=f"{name} Close", color=color, alpha=0.9, linewidth=1.5)

        # Plot MAs
        for win in window_sizes:
            ma_col = f"MA{win}"
            if ma_col in df.columns:
                linestyle = '--' if win < 100 else '-'
                width = 1.2 if win < 100 else 1.8
                alpha = 0.7
                
                ax.plot(df.index, df[ma_col], label=f"MA {win}", 
                        linestyle=linestyle, linewidth=width, alpha=alpha)

        ax.set_title(f"{name} - Price Trend", fontsize=12, fontweight='bold')
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Price ($)", fontsize=10)
        ax.legend(loc='upper left', fontsize=8, frameon=True)
        ax.grid(True, alpha=0.15, linestyle=':')
        
        # Despine
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

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
