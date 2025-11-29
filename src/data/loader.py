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
    """
    print(f"Fetching {period} of data for {ticker}...")
    df = yf.Ticker(ticker).history(period=period)
    df.index = df.index.tz_localize(None) # Remove timezone for simplicity in plots
    return df
