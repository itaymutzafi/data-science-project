import os
import pandas as pd
import yfinance as yf
from datetime import date
from typing import Union, List

def fetch_and_save(
    ticker: str, start_date: date, end_date: date, interval: str = "1d", folder: str = "raw") -> str:
    """
    Fetch historical stock data from Yahoo Finance and save it to CSV file(s).

    Can handle either a single ticker or multiple tickers. For multiple tickers,
    each ticker is saved to a separate CSV file.

    Parameters
    ----------
    ticker : str 
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
    
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date, interval=interval)
        
    filename = f"{ticker}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    file_path = os.path.join(data_folder, filename)
        
    # Save to CSV
    data.to_csv(file_path)
        
    print(f"Data for {ticker} saved to {file_path}")
    print(f"Fetched {len(data)} rows of data")
        
    return file_path


def load_csv(filename: str, folder: str = "raw") -> pd.DataFrame:
    """
    Load a CSV file from a subfolder inside the data directory.

    Parameters
    ----------
    filename : str
        CSV file name.
    folder : str
        Subfolder under 'data' ('raw', 'processed', etc.)

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Notes
    -----
    The expected structure will be finalized once the dataset is known.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(project_root, "data", folder, filename)
    return pd.read_csv(path)


def merge_csv_by_column(
    main_file: str,
    feature_files: Union[str, List[str]],
    columns_to_merge: Union[str, List[str]],
    merge_column: str = "Date",
    folder: str = "raw",
    output_file: str = None,
    output_folder: str = None,
    how: str = "left"
) -> str:
    """
    Merge selected columns from multiple CSV files into a main CSV file by a common column.
    
    This generic function allows you to merge specific columns from feature CSV files
    (e.g., S&P 500, NASDAQ, other stocks) into a main CSV file. The files are merged
    based on a common column (e.g., Date, ID, Timestamp, etc.). Only the specified
    columns are merged, and they are renamed with the format "PREFIX - COLUMN" to avoid conflicts.

    Parameters
    ----------
    main_file : str
        Filename of the main/base CSV file (e.g., 'AAPL_20230101_20231231.csv').
    feature_files : str or list of str
        Filename(s) of feature CSV files to merge into the main file.
        Each CSV should be for one stock. Prefixes are auto-generated from filenames.
        (e.g., 'MSFT_20230101_20231231.csv' or ['MSFT_20230101_20231231.csv', 'GOOGL_20230101_20231231.csv']).
    columns_to_merge : str or list of str
        Column name(s) to select and merge from all feature files (e.g., 'Open' or ['Open', 'Close']).
        The same columns are extracted from each feature file and merged.
    merge_column : str, optional
        Name of the column to merge on (default: 'Date'). Can be any column type (date, string, number, etc.).
    folder : str, optional
        Subfolder under 'data' where input CSV files are located (default: 'raw').
    output_file : str, optional
        Output filename. If None, will overwrite main_file.
    output_folder : str, optional
        Output folder. If None, uses the same folder as input files.
    how : str, optional
        Type of merge (default: 'left'). Options: 'left', 'right', 'outer', 'inner'.

    Returns
    -------
    str
        Path to the merged CSV file.

    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_folder = os.path.join(project_root, "data", folder)
    
    # Normalize inputs: convert single values to lists for uniform processing
    feature_files_list = [feature_files] if isinstance(feature_files, str) else feature_files
    
    # Normalize columns_to_merge: if string, convert to list; if list, use as-is
    if isinstance(columns_to_merge, str):
        columns_to_merge_list = [columns_to_merge]
    else:
        columns_to_merge_list = columns_to_merge
    
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
    main_df.set_index(merge_column, inplace=True)
    
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
        feature_df.set_index(merge_column, inplace=True)
        
        # Get prefix for this file
        prefix = feature_prefixes_list[i]
        
        # Check if columns exist in feature file and filter to only existing columns
        cols_to_select = [col for col in columns_to_merge_list if col in feature_df.columns]
        missing_cols = [col for col in columns_to_merge_list if col not in feature_df.columns]
        
        if missing_cols:
            print(f"Warning: Columns {missing_cols} not found in {feature_file}. Available columns: {list(feature_df.columns)}")
        
        if not cols_to_select:
            print(f"Warning: No valid columns to merge from {feature_file}. Skipping...")
            continue
        
        # Select only the specified columns and rename them
        feature_df_selected = feature_df[cols_to_select].copy()
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
        output_folder_path = os.path.join(project_root, "data", output_folder)
        os.makedirs(output_folder_path, exist_ok=True)
    
    if output_file is None:
        output_file = main_file
    
    output_path = os.path.join(output_folder_path, output_file)
    
    # Save merged CSV
    main_df.to_csv(output_path, index=False)
    print(f"Merged data saved to: {output_path}")
    print(f"Total rows: {len(main_df)}, Total columns: {len(main_df.columns)}")
    
    return output_path