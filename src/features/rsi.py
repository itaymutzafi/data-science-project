import pandas as pd

def add_rsi_feature(dfs, window=14):
    """
    Adds Relative Strength Index (RSI) feature to the dataframes.
    
    Args:
        dfs (dict): Dictionary of dataframes.
        window (int): Lookback window for RSI calculation.
    """
    print(f"Adding Feature: RSI ({window})...")
    count = 0
    for name, df in dfs.items():
        if 'Close' in df.columns:
            delta = df['Close'].diff()
            # Separate gains and losses
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            # Calculate RS and RSI
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Fill NaNs (Warm-up period) with Neutral 50
            df['RSI'] = df['RSI'].fillna(50)
            dfs[name] = df
            count += 1
            
    print(f" -> Success: RSI added to {count} tickers.")
