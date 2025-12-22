import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

from src.config import COMPANY_COLORS, DAYNAMES, MONTHNAMES, TICKERS

def add_volatility_features(dfs: Dict[str, pd.DataFrame], windows: List[int]) -> None:
    added = []

    for name, df in dfs.items():
        if "Return" in df.columns:
            for win in windows:
                df[f"Vol{win}"] = df['Return'].rolling(win).std().bfill()
            added.append(name)

    print(f"{added}: Added Volatility with windows {windows} features")

def volatility_comparison_plot(dfs: Dict[str, pd.DataFrame], window_size: int) -> None:
    plt.figure(figsize=(14, 6))
    vol_col = f"Vol{window_size}"

    for name, df in dfs.items():
        if vol_col in df.columns:
            plt.plot(df.index, df[vol_col], label=f"{name} ({window_size}-day)", color=COMPANY_COLORS[name], alpha=0.8, linewidth=1.5)

    plt.title(f"Rolling Volatility ({window_size}-day)")
    plt.ylabel("Volatility")
    plt.xlabel("Year")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
