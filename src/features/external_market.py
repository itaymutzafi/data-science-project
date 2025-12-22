import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

from src.config import *

def add_auxiliary_features(dfs: Dict[str, pd.DataFrame], aux_data: pd.DataFrame) -> None:
    """
    Merges selected auxiliary features (e.g., Nasdaq, VIX) into each company's DataFrame 
    and plots them for visual inspection.
    """
    if aux_data.empty:
        print("Auxiliary data is empty. Skipping feature integration.")
        return

    # Ensure aux_data index is tz-naive for merging
    if aux_data.index.tz is not None:
        aux_data = aux_data.copy()
        aux_data.index = aux_data.index.tz_localize(None)

    features_to_add = aux_data.columns.tolist()
    print(f"Integrating Auxiliary Features: {features_to_add}")

    for name, df in dfs.items():
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Merge left on index
        # We use join to keep the original index
        # Check if columns already exist to avoid duplication/suffixes
        new_cols = [c for c in features_to_add if c not in df.columns]
        
        if new_cols:
            dfs[name] = df.join(aux_data[new_cols], how='left')
            # Forward fill to handle missing daily data if aux matches higher timeframe or gaps
            dfs[name][new_cols] = dfs[name][new_cols].ffill()
        
            # Forward fill to handle missing daily data if aux matches higher timeframe or gaps
            dfs[name][new_cols] = dfs[name][new_cols].ffill()
        
    # --- Visualization 1: Combined Trend Plot (All Stocks + Aux Features) ---
    print(f"\nVisualizing Combined Context for {len(dfs)} companies...")
    plt.figure(figsize=(16, 8))
    
    # 1. Plot Aux Features (Background context)
    for col in features_to_add:
        # We can take the data from aux_data directly for the full range, 
        # or from the first df to ensure alignment. Using aux_data with alignment:
        series = aux_data[col].dropna()
        # Align to the date range of the stocks roughly
        common_start = min(df.index.min() for df in dfs.values())
        common_end = max(df.index.max() for df in dfs.values())
        
        # Ensure timezone naive comparisons
        if common_start.tz is not None: common_start = common_start.tz_localize(None)
        if common_end.tz is not None: common_end = common_end.tz_localize(None)
        
        mask = (series.index >= common_start) & (series.index <= common_end)
        series = series.loc[mask]
        
        if not series.empty:
            norm = (series - series.mean()) / series.std()
            color = AUX_COLORS.get(col, 'grey')
            plt.plot(series.index, norm, label=f"{col} (Macro)", 
                     color=color, linestyle='--', alpha=0.6, linewidth=1.5)

    # 2. Plot Stocks (Foreground focus)
    for ticker, df in dfs.items():
        if 'Close' not in df.columns:
            continue
        series = df['Close'].dropna()
        if not series.empty:
            norm = (series - series.mean()) / series.std()
            color = COMPANY_COLORS.get(ticker, COMPANY_COLORS.get(TICKER_TO_COMPANY_MAP.get(ticker), 'black'))
            plt.plot(series.index, norm, label=f"{ticker}", 
                     color=color, linewidth=2.5, alpha=0.9)

    plt.title("Market Context: Stocks vs. Macro Indicators (Normalized Z-Score)", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Z-Score (Std Dev from Mean)")
    plt.legend(ncol=2) # Two columns for legend to save space
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Visualization 2: Combined Correlation Matrix (Grouped Bar) ---
    corr_data = {}
    
    for ticker, df in dfs.items():
        if 'Return' not in df.columns:
            continue
            
        valid_cols = [c for c in features_to_add if c in df.columns]
        if not valid_cols:
            continue
            
        # correlation between Stock Return and Aux features
        corrs = df[['Return'] + valid_cols].corr()['Return'].drop('Return')
        corr_data[ticker] = corrs
    
    if corr_data:
        corr_df = pd.DataFrame(corr_data) # Rows = Aux features, Cols = Tickers
        
        # Transpose for easier grouping by ticker if preferred, 
        # OR keep as is to compare how 'VIX' affects AAPL vs AMZN.
        # Let's group by Stock (Ticker) on X-axis, and bars for each Aux feature.
        
        # We want: X-axis = Tickers, Bars = Aux Features
        ax = corr_df.T.plot(kind='bar', figsize=(14, 6), width=0.8, 
                            color=[AUX_COLORS.get(col, 'grey') for col in corr_df.index])
        
        plt.title("Correlation: Daily Returns vs. Macro Indicators", fontsize=14)
        plt.ylabel("Pearson Correlation")
        plt.xlabel("Company")
        plt.axhline(0, color='black', linewidth=0.8)
        plt.grid(True, axis='y', alpha=0.3)
        plt.legend(title="Macro Indicator")
        plt.tight_layout()
        plt.show()
