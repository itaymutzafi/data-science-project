import pandas as pd
import numpy as np

def add_vol_return_interaction(df, window=20):
    """
    Adds Conviction (Volume-Weighted Return) interaction feature.
    """
    if 'Volume' in df.columns and 'Log_Return' in df.columns:
        # Robustness: Add epsilon to avoid division by zero
        vol_ma = df['Volume'].rolling(window).mean() + 1e-6
        rel_vol = df['Volume'] / vol_ma
        df['Vol_x_Return'] = df['Log_Return'] * rel_vol
    return df

def add_macd_rsi_interaction(df):
    """
    Adds Confluence (MACD * RSI Centered) interaction feature.
    """
    if 'RSI' in df.columns and 'MACD' in df.columns:
        rsi_centered = df['RSI'] - 50
        df['MACD_x_RSI'] = df['MACD'] * rsi_centered
    return df

def add_trend_rsi_interaction(df, trend_col):
    """
    Adds Trend * Momentum (Trend * RSI Centered) interaction feature.
    """
    if 'RSI' in df.columns and trend_col in df.columns:
        rsi_centered = df['RSI'] - 50
        df['Trend_x_RSI'] = df[trend_col] * rsi_centered
    return df

def add_confluence_features(dfs, feature_windows=[20, 50, 200]):
    """
    Adds Interaction Terms to capture signal confluence.
    Dynamic Note: Uses feature_windows[1] (Mid-Term Trend) for interaction if available.
    
    Args:
        dfs (dict): Dictionary of dataframes.
        feature_windows (list): List of window sizes for MA features. Default [20, 50, 200].
    """
    print("Adding Interaction Features (Confluence)...")
    
    target_trend_win = feature_windows[1] if len(feature_windows) > 1 else 50
    trend_col = f'Dist_MA{target_trend_win}'
    print(f" -> Using '{trend_col}' for Trend-Momentum interaction.")
    
    count = 0
    for name, df in dfs.items():
        # 1. Conviction
        df = add_vol_return_interaction(df, window=20)
        
        # 2. Confluence
        df = add_macd_rsi_interaction(df)
            
        # 3. Trend * Momentum
        df = add_trend_rsi_interaction(df, trend_col)
        
        dfs[name] = df
        count += 1
    
    print(f" -> Success: Interactions added to {count} tickers.")
    return dfs
