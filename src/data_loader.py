import os
import pandas as pd
import yfinance as yf
from datetime import date

def fetch_and_save(ticker: str, start_date: date, end_date: date, folder: str = "raw") -> str:
    """
    Fetch historical stock data from Yahoo Finance and save it to a CSV file.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'GOOGL').
    start_date : date
        Start date for historical data.
    end_date : date
        End date for historical data.
    folder : str, optional
        Subfolder under 'data' to save the file (default: 'raw').

    Returns
    -------
    str
        Path to the saved CSV file.

    Examples
    --------
    >>> from datetime import date
    >>> fetch_and_save('AAPL', date(2023, 1, 1), date(2023, 12, 31))
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_folder = os.path.join(project_root, "data", folder)
    
    # Create the folder if it doesn't exist
    os.makedirs(data_folder, exist_ok=True)
    
    # Fetch data from Yahoo Finance
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    
    # Generate filename based on ticker and date range
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
    
