import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

from src.config import COMPANY_COLORS, VOLATILITY_WINDOWS, OUTLIER_THRESHOLD
import seaborn as sns

VOL_FEATURE = []

def add_volatility_features(dfs: Dict[str, pd.DataFrame], windows: List[int] = VOLATILITY_WINDOWS) -> None:
    """
    Adds Volatility features (Standard Deviation of Returns).
    Handles outliers by capping at OUTLIER_THRESHOLD * STD.
    """
    added = []
    
    global VOL_FEATURE
    
    for name, df in dfs.items():
        if "Return" in df.columns:
            for win in windows:
                col_name = f"Vol{win}"
                
                # Calculate rolling std
                vol_series = df['Return'].rolling(window=win).std()
                
                # Outlier Handling: Cap extreme volatilities
                # We calculate the z-score of the VOLATILITY itself to find extreme anomalies
                # relative to its own distribution, or just use raw threshold if domain knowledge suggests.
                # Here, let's use a robust method: anything > threshold * mean_vol is capped.
                # Actually, simple Z-score on the vol series:
                
                mean_vol = vol_series.mean()
                std_vol = vol_series.std()
                
                if std_vol > 0:
                    upper_limit = mean_vol + (OUTLIER_THRESHOLD * std_vol)
                    # Cap in place (Winsorization) to preserve data integrity for models
                    vol_series = vol_series.clip(upper=upper_limit)
                
                df[col_name] = vol_series
                
                if col_name not in VOL_FEATURE:
                    VOL_FEATURE.append(col_name)
                    
            added.append(name)

    print(f"Added Volatility features: {windows} for {len(added)} stocks.")
    print(f"Note: Outliers capped at {OUTLIER_THRESHOLD} sigma.")

def volatility_comparison_plot(dfs: Dict[str, pd.DataFrame], window_sizes: List[int] = VOLATILITY_WINDOWS) -> None:
    """
    Comparison estimate of volatility across companies.
    """
    for win in window_sizes:
        vol_col = f"Vol{win}"
        
        plt.figure(figsize=(14, 6))
        
        has_data = False
        for name, df in dfs.items():
            if vol_col in df.columns:
                has_data = True
                plt.plot(df.index, df[vol_col], label=f"{name}", 
                         color=COMPANY_COLORS.get(name, 'grey'), alpha=0.8, linewidth=1.5)

        if not has_data:
            plt.close()
            continue

        plt.title(f"Market Volatility ({win}-day Rolling Std Dev)", fontsize=14, fontweight='bold')
        plt.ylabel("Volatility (Std Dev of Returns)", fontsize=12)
        plt.xlabel("Date", fontsize=12)
        plt.legend(frameon=True, shadow=True)
        plt.grid(True, alpha=0.2, linestyle='--')
        sns.despine()
        plt.tight_layout()
        plt.show()
