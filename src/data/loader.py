"""Data access module.

This module handles data fetching from external APIs (Yahoo Finance)
and local file I/O operations (CSV, Parquet).
"""

import os
from datetime import date
from pathlib import Path
from typing import List, Union

import pandas as pd
import yfinance as yf
import pandas as pd
import yfinance as yf
from yfinance import Ticker

from src.config import AUX_DATA_PATH, AUX_TICKER_MAP


def fetch_sample_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Fetches raw OHLCV data for initial research and stationarity tests.
    Implements local caching to avoid repeated API calls.
    """
    # Define cache path
    project_root = Path(__file__).resolve().parents[2]
    cache_dir = project_root / "data" / "raw"
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


def fetch_and_save(ticker: Ticker, start_date: date, end_date: date, interval: str = "1d", folder: str = "raw") -> str:
    """
    Fetch historical stock data from Yahoo Finance and save it to CSV file(s).

    Can handle either a single ticker or multiple tickers. For multiple tickers,
    each ticker is saved to a separate CSV file.

    Parameters
    ----------
    ticker : Ticker
        Stock ticker symbol(s) (e.g., 'AAPL', ['AAPL', 'MSFT', 'GOOGL']).
    start_date : date,
        Start date for historical data.
    end_date : date,
        End date for historical data.
    interval : str, optional
        Data interval (default: '1d'). Valid intervals: '1m', '2m', '5m', '15m', 
        '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'.
    folder : str, optional
        Subfolder under 'data' to save the file(s) (default: 'raw').

    Returns
    -------
    str
        Path(s) to the saved CSV file(s).
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_folder = os.path.join(project_root, "data", folder)
    
    # Create the folder if it doesn't exist
    os.makedirs(data_folder, exist_ok=True)  
    
    data = ticker.history(start=start_date, end=end_date, interval=interval)
        
    filename = f"{ticker.ticker}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    file_path = os.path.join(data_folder, filename)
        
    # Save to CSV
    data.to_csv(file_path)
        
    print(f"Data for {ticker.ticker} saved to {file_path}")
    print(f"Fetched {len(data)} rows of data")
        
    return file_path


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


def merge_csv_by_date(
    main_file: str,
    feature_files: Union[str, List[str]],
    folder: str = "raw",
    output_file: str = None,
    output_folder: str = None,
    how: str = "left"
) -> str:
    """
    Merge multiple stock CSV files by Date, including only Open, Close, and Volume columns.
    
    Parameters
    ----------
    main_file : str
        Filename of the main CSV file (e.g., 'AAPL_20230101_20231231.csv').
    feature_files : str or list of str
        Filename(s) of stock CSV files to merge (e.g., 'MSFT_20230101_20231231.csv' or ['MSFT_...', 'GOOGL_...']).
    folder : str, optional
        Subfolder under 'data' where CSV files are located (default: 'raw').
    output_file : str, optional
        Output filename. If None, overwrites main_file.
    output_folder : str, optional
        Output folder. If None, uses the same folder as input files.
    how : str, optional
        Type of merge (default: 'left'). Options: 'left', 'right', 'outer', 'inner'.
    
    Returns
    -------
    str
        Path to the merged CSV file.
    
    Note
    ----
    Only Open, Close, and Volume columns are merged from feature files.
    Main file keeps all its columns.
    
    Example
    -------
    >>> merge_csv_by_date('AAPL_20230101_20231231.csv', 'MSFT_20230101_20231231.csv')
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_folder = os.path.join(project_root, "data", folder)
    
    # Normalize inputs: convert single values to lists for uniform processing
    feature_files_list = [feature_files] if isinstance(feature_files, str) else feature_files
    
    # Auto-generate prefixes from filenames (extract ticker symbol before first '_')
    # Extract just the filename if full paths are provided, then get ticker symbol
    feature_prefixes_list = [os.path.splitext(os.path.basename(f))[0].split('_')[0] for f in feature_files_list]
    
    # Load main CSV file
    # Check if it's a full path or just a filename
    if os.path.isabs(main_file):
        main_path = main_file  # Already a full path
    else:
        main_path = os.path.join(data_folder, main_file)  # Just filename, join with folder
    
    if not os.path.exists(main_path):
        raise FileNotFoundError(f"Main file not found: {main_path}")
    
    print(f"Loading main file: {main_file}")
    main_df = pd.read_csv(main_path)
    main_df.set_index("Date", inplace=True)
    
    # Merge each feature file
    for i, feature_file in enumerate(feature_files_list):
        # Check if it's a full path or just a filename
        if os.path.isabs(feature_file):
            feature_path = feature_file  # Already a full path
        else:
            feature_path = os.path.join(data_folder, feature_file)  # Just filename, join with folder
        
        if not os.path.exists(feature_path):
            print(f"Warning: Feature file not found: {feature_path}. Skipping...")
            continue
        
        print(f"Loading feature file: {feature_file}")
        feature_df = pd.read_csv(feature_path)
        feature_df.set_index("Date", inplace=True)
        
        # Get prefix for this file
        prefix = feature_prefixes_list[i]
        
        # Select only Open, Close, and Volume columns
        columns_to_merge = ['Open', 'Close', 'Volume']
        available_cols = [col for col in columns_to_merge if col in feature_df.columns]
        
        if not available_cols:
            print(f"Warning: No matching columns (Open, Close, Volume) found in {feature_file}. Skipping...")
            continue
        
        feature_df_selected = feature_df[available_cols].copy()
        feature_df_selected.columns = [f"{prefix} - {col}" for col in feature_df_selected.columns]
        
        # Merge with main dataframe
        main_df = main_df.join(feature_df_selected, how=how)
        print(f"Merged {feature_file} - Added {len(feature_df_selected.columns)} columns: {list(feature_df_selected.columns)}")
    
    # Reset index to make Date a column again
    main_df.reset_index(inplace=True)
    
    # Determine output path
    if output_folder is None:
        output_folder_path = data_folder
    else:
        output_folder_path = project_root / "data" / output_folder
        output_folder_path.mkdir(parents=True, exist_ok=True)
    
    if output_file is None:
        output_file = main_file
    
    output_path = output_folder_path / output_file
    
    # Save merged CSV
    main_df.to_csv(output_path, index=False)
    print(f"Merged data saved to: {output_path}")
    print(f"Total rows: {len(main_df)}, Total columns: {len(main_df.columns)}")

    return output_path