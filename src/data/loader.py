"""Data access module.

This module handles data fetching from external APIs (Yahoo Finance)
and local file I/O operations (CSV, Parquet).
"""
from datetime import date
from pathlib import Path
from typing import List, Union
import pandas as pd
import yfinance as yf
from yfinance import Ticker
import numpy as np
from src.config import AUX_DATA_PATH, AUX_TICKER_MAP


def fetch_sample_data(ticker: Ticker, start_time: date, end_time: date, period: str = "5y") -> pd.DataFrame:
    """
    Fetches raw OHLCV data for initial research and stationarity tests.
    Implements local caching to avoid repeated API calls.
    """
    # Define cache path
    ticker_name = ticker.ticker
    project_root = Path(__file__).resolve().parents[2]
    cache_dir = project_root / "data" / "raw"
    cache_path = cache_dir / f"{ticker_name}.parquet"
    
    # Check cache
    if cache_path.exists():
        print(f"Loading {ticker_name} data from cache... - don't generate a new df!!!")
        df = pd.read_parquet(cache_path)
        # Ensure index is timezone-naive for consistency
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
        
    # Download if not in cache
    print(f"Fetching {period} of data for {ticker_name} from yfinance...")
    df = ticker.history(start = start_time, end = end_time)
    
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker_name}")
        
    # Remove timezone for simplicity in plots and saving
    df.index = df.index.tz_localize(None)
    
    # Save to cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    print(f"Saved {ticker} data to {cache_path}")
    
    return df


def fetch_auxiliary_data(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Fetches comprehensive market data for enrichment analysis.
    
    Categories:
    1. Macro-Economic: QQQ (Tech Index), ^VIX (Volatility), ^TNX (10Y Yield).
    2. Segment/Supply-Chain: NVDA (AI Hardware Leader).
    
    Implements caching using AUX_DATA_PATH.
    
    Returns:
        pd.DataFrame: A unified DataFrame with renamed columns for clarity.
    """
    # 1. Check Cache
    project_root = Path(__file__).resolve().parents[2]
    # Handle relative path from config
    if Path(AUX_DATA_PATH).is_absolute():
        cache_path = Path(AUX_DATA_PATH)
    else:
        cache_path = project_root / AUX_DATA_PATH
        
    if cache_path.exists():
        print(f"Loading auxiliary data from cache: {cache_path}")
        try:
            data = pd.read_parquet(cache_path)
            # Ensure index is timezone-naive
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            return data
        except Exception as e:
            print(f"Error loading cache: {e}. Fetching fresh data.")

    # 2. Download from yfinance
    print("Fetching auxiliary market data from yfinance...")
    try:
        # Download data
        # Note: auto_adjust=True is default in newer yfinance, so 'Close' is adjusted.
        # If start_date/end_date are None, yfinance defaults to max or similar, 
        # but for auxiliary data we usually want a broad range or matching the main data.
        # Here we'll default to a reasonable period if not specified, or let yfinance handle it.
        # However, yf.download without start/end might be too much. 
        # Let's use a default period if dates are missing, or just download everything.
        
        tickers = list(AUX_TICKER_MAP.keys())
        if start_date and end_date:
            data = yf.download(tickers, start=start_date, end=end_date)['Close']
        else:
            # Default to 5 years if not specified, to match sample data
            data = yf.download(tickers, period="5y")['Close']
            
        # Rename columns using the mapping
        data = data.rename(columns=AUX_TICKER_MAP)
        
        # Handle missing values (forward fill)
        data = data.ffill()
        
        # Ensure timezone-naive index
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
            
        # 3. Save to Cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(cache_path)
        print(f"Saved auxiliary data to {cache_path}")
            
        return data
    except Exception as e:
        print(f"Error fetching auxiliary data: {e}")
        return pd.DataFrame()


def merge_df_by_date(
    main_df: pd.DataFrame,
    feature_dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    feature_names: Union[str, List[str]] = None,
    output_filename : str = "merged_df.parquet",
    how: str = "left") -> pd.DataFrame:
    
    # Normalize inputs: convert single DataFrame to list for uniform processing
    feature_dfs_list = [feature_dfs] if isinstance(feature_dfs, pd.DataFrame) else feature_dfs

    if isinstance(feature_names, str):
        feature_names_list = [feature_names]
    else: 
        feature_names_list = feature_names

    # Ensure index is DatetimeIndex
    if not isinstance(main_df.index, pd.DatetimeIndex):
        main_df.index = pd.to_datetime(main_df.index)
    
    columns_to_merge = ['Close', 'Volume']

    print(f"Start Merging")

    res_df = main_df.copy()
   
    # Merge each feature file
    for i, feature_df in enumerate(feature_dfs_list):
        feature_name = feature_names_list[i]

        # Ensure feature index is DatetimeIndex
        if not isinstance(feature_df.index, pd.DatetimeIndex):
            feature_df.index = pd.to_datetime(feature_df.index)
        
        # Select only Open, Close, and Volume columns
        available_cols = [col for col in columns_to_merge if col in feature_df.columns]
        
        if not available_cols:
            print(f"Warning: No matching columns {columns_to_merge} found in {feature_name}. Skipping...")
            continue
        
        feature_df_selected = feature_df[available_cols].copy()
        feature_df_selected.columns = [f"{feature_name} - {col}" for col in feature_df_selected.columns]
        
        # Merge with main dataframe
        res_df = res_df.join(feature_df_selected, how=how)
        print(f"Merged {feature_name} - Added {len(feature_df_selected.columns)} columns: {list(feature_df_selected.columns)}")
       
    print(f"Final merged DataFrame shape: {res_df.shape}")
    project_root = Path(__file__).resolve().parents[2]
    cache_dir = project_root / "data" / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / output_filename
    
    res_df.to_parquet(cache_path)
    print(f"Merged data saved to: {cache_path}")

    return res_df
    
def fetch_data_for_eda(ticker: str, start_time: date, end_time: date):
    Ticker = yf.Ticker(ticker)
    df = fetch_sample_data(Ticker, start_time, end_time)
    df['Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    return df


def load_stock_data(path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads stock data from a Parquet file, ensuring the index is a timezone-naive DatetimeIndex.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Stock data file not found at: {path}")
    
    print(f"Loading stock data from {path}...")
    df = pd.read_parquet(path)
    
    # Ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
            
    # Remove timezone information if present
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
        
    return df
