"""Data access module.

This module handles data fetching from external APIs (Yahoo Finance)
and local file I/O operations (CSV, Parquet).
"""
from datetime import date
from pathlib import Path
from typing import Union, Dict
import pandas as pd
import yfinance as yf
from yfinance import Ticker

from src.config import (
    AUX_DATA_PATH,
    AUX_TICKER_MAP,
    LEGACY_AUX_DATA_PATH,
    LEGACY_RAW_PRICE_DIR,
    RAW_PRICE_DIR,
    TICKERS,
    WARMUP_START_DATE,
    START_DATE,
    END_DATE
)
from src.utils.feature_names import canonicalize_feature_columns

def fetch_wide_data() -> Dict[str, pd.DataFrame]:
    """
    Fetch price data from WARMUP_START_DATE to END_DATE.
    Single source for feature engineering and Prophet - ensures MA200 etc. have warmup.
    """
    return fetch_data_for_eda(WARMUP_START_DATE, END_DATE)

def fetch_sample_data(ticker: Ticker, start_time: date, end_time: date, period: str = "5y", save_file: bool = True) -> pd.DataFrame:
    """
    Fetches raw OHLCV data for initial research and stationarity tests.
    Implements local caching to avoid repeated API calls.
    Only fetches new data if the requested date range differs from cached data.
    """
    # Define cache paths
    ticker_name = ticker.ticker
    primary_cache_dir = Path(RAW_PRICE_DIR)
    primary_cache_path = primary_cache_dir / f"{ticker_name}.parquet"
    legacy_cache_path = Path(LEGACY_RAW_PRICE_DIR) / f"{ticker_name}.parquet"
    cache_path = primary_cache_path if primary_cache_path.exists() else legacy_cache_path
    
    # Check cache
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        # Ensure index is timezone-naive for consistency
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        cached_start = df.index.min().date()
        cached_end = df.index.max().date()
        
        # Check coverage with tolerance for market holidays/weekends
        start_diff = (start_time - cached_start).days
        end_diff = (end_time - cached_end).days
        
        start_covered = abs(start_diff) <= 5
        end_covered = abs(end_diff) <= 5
        
        if start_covered and end_covered:
            # One-time migration: if loaded from legacy, materialize in primary cache layout.
            if cache_path != primary_cache_path:
                primary_cache_dir.mkdir(parents=True, exist_ok=True)
                df.to_parquet(primary_cache_path)
            print(f"Data for {ticker_name} is up-to-date (covers {start_time} to {end_time}). Using cached data.")
            return df
        else:
            # Need to fetch new data - calculate years
            years_diff = (end_time - start_time).days / 365.25
            print(f"Cache doesn't cover requested range. Fetching {years_diff:.1f} year(s) of data for {ticker_name} from yfinance...")
    else:
        # No cache exists - calculate years
        years_diff = (end_time - start_time).days / 365.25
        print(f"Fetching {years_diff:.1f} year(s) of data for {ticker_name} from yfinance...")
        
    # Download data with fallback
    try:
        df = ticker.history(start = start_time, end = end_time)
        
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker_name}")
            
        # Remove timezone for simplicity in plots and saving
        df.index = df.index.tz_localize(None)
        
        # Save to cache
        if save_file:
            primary_cache_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(primary_cache_path)
            print(f"Saved {ticker_name} data to {primary_cache_path}")
        
        return df

    except Exception as e:
        print(f"Failed to fetch data for {ticker_name} from yfinance: {e}")
        
        for fallback_path in (primary_cache_path, legacy_cache_path):
            if fallback_path.exists():
                print(f"WARNING: Falling back to EXISTING CACHE for {ticker_name}. Data coverage might be incomplete.")
                df = pd.read_parquet(fallback_path)
                
                # Ensure index is timezone-naive for consistency
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                return df
        print(f"CRITICAL: No local cache available for {ticker_name} to fallback to.")
        raise e


def fetch_auxiliary_data(start_date: str = None, end_date: str = None, verbose: bool = False) -> pd.DataFrame:
    """
    Fetches comprehensive market data for enrichment analysis.
    
    Categories:
    1. Macro-Economic: QQQ (Tech Index), ^VIX (Volatility), ^TNX (10Y Yield).
    2. Segment/Supply-Chain: NVDA (AI Hardware Leader).
    
    Implements caching using AUX_DATA_PATH.
    
    Returns:
        pd.DataFrame: A unified DataFrame with renamed columns for clarity.
    """
    # 1. Check Cache (primary first, then legacy path)
    primary_cache_path = Path(AUX_DATA_PATH)
    legacy_cache_path = Path(LEGACY_AUX_DATA_PATH)

    for cache_path in (primary_cache_path, legacy_cache_path):
        if cache_path.exists():
            if verbose:
                print(f"Loading auxiliary data from cache: {cache_path}")
            try:
                data = pd.read_parquet(cache_path)
                data = canonicalize_feature_columns(data)
                # Ensure index is timezone-naive
                if data.index.tz is not None:
                    data.index = data.index.tz_localize(None)

                # One-time migration from legacy cache to primary layout.
                if cache_path != primary_cache_path:
                    primary_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    data.to_parquet(primary_cache_path)

                return data
            except Exception as e:
                if verbose:
                    print(f"Error loading cache: {e}. Fetching fresh data.")

    # 2. Download from yfinance
    if verbose:
        print("Fetching auxiliary market data from yfinance...")
    try:
        # Download data (default to 5y if dates not specified to match sample data)
        
        tickers = list(AUX_TICKER_MAP.keys())
        if start_date and end_date:
            data = yf.download(tickers, start=start_date, end=end_date)['Close']
        else:
            # Default to 5 years if not specified, to match sample data
            data = yf.download(tickers, period="5y")['Close']
            
        # Rename columns using the mapping
        data = data.rename(columns=AUX_TICKER_MAP)
        data = canonicalize_feature_columns(data)
        
        # Handle missing values (forward fill)
        data = data.ffill()
        
        # Ensure timezone-naive index
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
            
        # 3. Save to Cache
        primary_cache_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(primary_cache_path)
        if verbose:
            print(f"Saved auxiliary data to {primary_cache_path}")
            
        return data
    except Exception as e:
        if verbose:
            print(f"Error fetching auxiliary data: {e}")
        return pd.DataFrame()

    
def fetch_data_for_eda(start_time: date, end_time: date) -> Dict[str, pd.DataFrame]:
    stocks_data = {}

    for ticker in TICKERS:
        Ticker = yf.Ticker(ticker)
        stocks_data[ticker] = fetch_sample_data(Ticker, start_time, end_time)
    return stocks_data


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

def filter_to_model_range(data: Dict[str, pd.DataFrame],
    model_start: date = None,
    ) -> Dict[str, pd.DataFrame]:
    """
    Keep only rows from model_start onwards. Use after feature engineering.
    """
    model_start = model_start or START_DATE
    cutoff = pd.Timestamp(model_start)
    return {t: df.loc[df.index >= cutoff].copy() for t, df in data.items()}