"""Data access module (skeleton).

This module defines the I/O contract for the project. Implementations
are intentionally omitted here: functions raise NotImplementedError so the
team can implement them in a controlled, reviewed way.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import yfinance as yf




def fetch_sample_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Fetches raw OHLCV data for initial research and stationarity tests.
    Implements local caching to avoid repeated API calls.
    """
    # Define cache path
    cache_dir = Path("data/raw")
    cache_path = cache_dir / f"{ticker}.parquet"
    
    # Check cache
    if cache_path.exists():
        print(f"Loading {ticker} data from cache...")
        df = pd.read_parquet(cache_path)
        # Ensure index is timezone-naive for consistency
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
        
    # Download if not in cache
    print(f"Fetching {period} of data for {ticker} from yfinance...")
    df = yf.Ticker(ticker).history(period=period)
    
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")
        
    # Remove timezone for simplicity in plots and saving
    df.index = df.index.tz_localize(None)
    
    # Save to cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    print(f"Saved {ticker} data to {cache_path}")
    
    return df

def load_csv(filepath: str) -> pd.DataFrame:
    """
    Loads data from a CSV file.
    """
    raise NotImplementedError("load_csv not implemented yet.")
