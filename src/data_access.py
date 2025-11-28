"""Data access module (skeleton).

This module defines the I/O contract for the project. Implementations
are intentionally omitted here: functions raise NotImplementedError so the
team can implement them in a controlled, reviewed way.
"""

from pathlib import Path
from typing import Optional
import pandas as pd


def load_csv(filename: str, folder: str = "raw") -> pd.DataFrame:
    """Load a CSV from the project `data/` folder.

    To be implemented by the data engineer.
    """
    raise NotImplementedError("load_csv must be implemented by the team")


def save_csv(df: pd.DataFrame, filename: str, folder: str = "processed") -> None:
    """Save dataframe to `data/{folder}/{filename}`.

    To be implemented.
    """
    raise NotImplementedError("save_csv must be implemented by the team")


def fetch_yfinance(ticker: str, start_date: str, end_date: str, progress: Optional[bool] = False):
    """Fetch OHLCV data for `ticker` between `start_date` and `end_date`.

    Implementation should use `yfinance` or an approved data provider.
    """
    raise NotImplementedError("fetch_yfinance must be implemented by the team")
