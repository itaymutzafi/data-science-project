import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

from src.config import COMPANY_COLORS, FEATURE_WINDOWS

MA_FEATURES = []

def add_ma_features(dfs: Dict[str, pd.DataFrame], windows: List[int] = FEATURE_WINDOWS) -> None:
    """
    Adds Moving Average features to the dataframe.
    
    Args:
        dfs: Dictionary of DataFrames.
        windows: List of window sizes. Defaults to FEATURE_WINDOWS from config.
    """
    added = []
    
    # reset global list to avoid duplicates if re-run
    global MA_FEATURES
    
    for name, df in dfs.items():
        if "Close" in df.columns:
            for win in windows:
                col_name = f"MA{win}"
                # Standard causal rolling mean
                df[col_name] = df['Close'].rolling(window=win).mean()
                
                if col_name not in MA_FEATURES:
                    MA_FEATURES.append(col_name)
            added.append(name)

    print(f"Added Moving Average features: {windows} for {len(added)} stocks.")
    print(f"Note: First {max(windows)} rows will contain NaNs (Warm-up Period).")

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
            MA_FEATURES.append("MACD")

            df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
            MA_FEATURES.append("MACD_Signal")

            df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
            MA_FEATURES.append("MACD_Hist")

            added.append(name)

    print(f"{added}: Added MACD feature")

def ma_plot(dfs: Dict[str, pd.DataFrame], window_sizes: List[int] = FEATURE_WINDOWS) -> None:
    """
    Plots Close price against multiple Moving Averages with premium aesthetics.
    """
    for name, df in dfs.items():
        plt.figure(figsize=(14, 7))
        
        # Plot Close Price
        plt.plot(df.index, df['Close'], label=f"{name} Close", color='black', alpha=0.6, linewidth=1.5)
        
        # Plot MAs
        for win in window_sizes:
            ma_col = f"MA{win}"
            if ma_col in df.columns:
                # Use slightly different styles for different MAs
                linestyle = '--' if win < 100 else '-'
                width = 1.5 if win < 100 else 2.0
                alpha = 0.9
                
                # Dynamic coloring or fallback
                color = COMPANY_COLORS.get(name, 'blue') 
                # Adjust shade based on window? For now, keep it simple but distinct
                
                plt.plot(df.index, df[ma_col], label=f"MA {win}", 
                         linestyle=linestyle, linewidth=width, alpha=alpha)

        plt.title(f"{name} - Price Trend Analysis (Moving Averages)", fontsize=14, fontweight='bold')
        plt.ylabel("Price ($)", fontsize=12)
        plt.xlabel("Date", fontsize=12)
        plt.legend(loc='upper left', fontsize=10, frameon=True, shadow=True)
        plt.grid(True, alpha=0.15, linestyle=':') # Subtle modern grid
        
        # Remove top and right spines for cleaner look
        sns.despine() if 'sns' in globals() else None
        
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
