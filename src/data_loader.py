import os
import pandas as pd
import yfinance as yf
from datetime import date, datetime, timezone
from typing import Union, List, Optional
from yfinance import Ticker

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

def create_news_column(
    ticker: str,
    column_name: Optional[str] = None,
    max_items: int = 20,
    include_summary: bool = True
) -> pd.DataFrame:
    """
    Build a single textual feature column from Yahoo Finance news items.

    Parameters
    ----------
    ticker : str
        Ticker symbol to pull news for (e.g., 'AAPL', 'MSFT').
    column_name : str, optional
        Custom column name. Defaults to '{ticker}_news'.
    max_items : int, optional
        Maximum number of news articles to keep (default: 20).
    include_summary : bool, optional
        If True, append the news summary after the title for each row.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'published': timezone-aware datetime of the article
        - '<column_name>': concatenated text (title [+ summary])
        - 'publisher': publisher name
        - 'link': URL to the news item
    """
    column_name = column_name or f"{ticker}_news"
    ticker_obj = yf.Ticker(ticker)

    try:
        raw_news = ticker_obj.get_news() or []
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch news for {ticker}") from exc

    rows: List[dict] = []
    for item in raw_news[:max_items]:
        publish_ts = item.get("providerPublishTime")
        published_dt = (
            datetime.fromtimestamp(publish_ts, tz=timezone.utc)
            if publish_ts is not None
            else None
        )
        title = (item.get("title") or "").strip()
        summary = (item.get("summary") or "").strip()

        text = title
        if include_summary and summary:
            text = f"{title} — {summary}"

        rows.append(
            {
                "published": published_dt,
                column_name: text,
                "publisher": item.get("publisher"),
                "link": item.get("link"),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["published", column_name, "publisher", "link"])

    return pd.DataFrame(rows)


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

def main():
    # Merge multiple stocks by date
    start = date(2020, 1, 1)
    end = date(2025, 12, 3)
    
    main_file = fetch_and_save(yf.Ticker("AAPL"), start, end)
    feature_file = fetch_and_save(yf.Ticker("MSFT"), start, end)
    feature_file2 = fetch_and_save(yf.Ticker("GOOGL"), start, end)
    
    # Pass full paths - merge_csv_by_date handles both absolute paths and filenames
    result = merge_csv_by_date(
        main_file=main_file,  # Full path
        feature_files=[feature_file, feature_file2],  # Full paths
    )
    print(f"Merged file: {result}")
    return result

main()
