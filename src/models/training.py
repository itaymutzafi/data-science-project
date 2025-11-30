"""Trainer module.

Contains model training utilities.
"""

from typing import Any
import pandas as pd

def train_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    Trains a model.
    """
    raise NotImplementedError("train_model not implemented yet.")

def time_series_split(data: pd.DataFrame, n_splits: int = 5):
    """
    Performs time series split.
    """
    raise NotImplementedError("time_series_split not implemented yet.")
