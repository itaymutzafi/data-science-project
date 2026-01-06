"""Preprocessing utilities for the stock prediction pipeline."""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class LogReturnTransformer(BaseEstimator, TransformerMixin):
    """
    Computes logarithmic returns from price data.

    Formula: ln(P_t) - ln(P_{t-1})
    """

    def __init__(self, price_col: str = "Close", new_col: str = "Log_Returns"):
        self.price_col = price_col
        self.new_col = new_col

    def fit(self, X, y=None):
        """Stateless transformer, nothing to learn."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            X: DataFrame containing the price column.
        Returns:
            DataFrame with the new log return column.
            Note: The first row will be NaN due to the shift.
        """
        X_new = X.copy()
        X_new[self.new_col] = np.log(X_new[self.price_col] / X_new[self.price_col].shift(1))
        return X_new


class TimeSeriesScaler:
    """StandardScaler wrapper that scales selected columns without data leakage."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.columns: Optional[List[str]] = None
        self._fitted = False

    def fit(self, df: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> "TimeSeriesScaler":
        """
        Fit scaler on the specified columns of the training data.

        Args:
            df: Training DataFrame.
            columns: Columns to scale. If None, all numeric columns are scaled.
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        self.columns = list(columns)
        self.scaler.fit(df[self.columns].values)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """
        Transform data using fitted scaler, preserving non-scaled columns.

        Args:
            df: DataFrame to scale.
            columns: Columns to scale. If None, uses columns seen at fit time.
        """
        if not self._fitted:
            raise RuntimeError("TimeSeriesScaler must be fitted before calling transform.")

        cols = list(columns) if columns is not None else self.columns
        if cols is None:
            raise ValueError("No columns provided for scaling.")

        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Columns not found in input data: {missing}")

        transformed = df.copy()
        transformed[cols] = self.scaler.transform(transformed[cols].values)
        return transformed


class MultiTickerScaler:
    """
    Scales a dictionary of DataFrames using a shared scaler fitted ONLY on training data.
    Prevents data leakage by respecting time boundaries.
    """
    
    def __init__(self, scaler_type: str = "standard"):
        self.scaler = StandardScaler() if scaler_type == "standard" else None
        self.columns: Optional[List[str]] = None
        self._fitted = False
        
    def fit(self, dfs: Dict[str, pd.DataFrame], train_end_date: str, columns: Optional[List[str]] = None) -> "MultiTickerScaler":
        """
        Fits the scaler on the concatenated training partition of all tickers.
        """
        train_parts = []
        
        for ticker, df in dfs.items():
            # Slice training data (up to cutoff)
            if not isinstance(df.index, pd.DatetimeIndex):
                # Attempt to set index if 'date' or 'Date' exists, else assume index is date
                if 'Date' in df.columns: df = df.set_index('Date')
                elif 'date' in df.columns: df = df.set_index('date')
                
            # Sort to be safe
            df = df.sort_index()
            
            # Select columns
            if columns is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            else:
                numeric_cols = columns
                
            self.columns = numeric_cols
            
            # Filter by date
            train_subset = df.loc[:train_end_date, self.columns]
            train_parts.append(train_subset)
            
        if not train_parts:
            raise ValueError("No training data found to fit scaler.")
            
        # Concatenate all training data to learn global stats
        full_train = pd.concat(train_parts)
        self.scaler.fit(full_train)
        self._fitted = True
        return self

    def transform(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Transforms all DataFrames in the dictionary using the fitted scaler.
        """
        if not self._fitted:
            raise RuntimeError("Scaler must be fitted before transform.")
            
        scaled_dfs = {}
        for ticker, df in dfs.items():
            df_copy = df.copy()
            # Ensure index alignment
            
            # transform
            df_copy[self.columns] = self.scaler.transform(df_copy[self.columns])
            scaled_dfs[ticker] = df_copy
            
        return scaled_dfs

def create_sequences(
    data: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
    seq_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert time-series data to sliding windows for sequence models.

    y[i] corresponds to the target immediately following X[i].

    Args:
        data: 2D array-like of shape (T, F).
        target: 1D array-like of shape (T,).
        seq_length: Number of time steps per sequence window.

    Returns:
        X: Array of shape (N, seq_length, F)
        y: Array of shape (N,)
    """
    data_arr = np.asarray(data)
    target_arr = np.asarray(target)

    if data_arr.ndim != 2:
        raise ValueError("data must be 2D (time, features).")
    if target_arr.ndim != 1:
        raise ValueError("target must be 1D.")
    if len(data_arr) != len(target_arr):
        raise ValueError("data and target must have the same length.")
    if seq_length <= 0:
        raise ValueError("seq_length must be positive.")
    if len(data_arr) <= seq_length:
        return np.empty((0, seq_length, data_arr.shape[1])), np.empty((0,))

    X, y = [], []
    for i in range(len(data_arr) - seq_length):
        X.append(data_arr[i : i + seq_length])
        y.append(target_arr[i + seq_length])

    return np.array(X), np.array(y)
