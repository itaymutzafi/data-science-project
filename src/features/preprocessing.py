"""Preprocessing transformers for the stock prediction pipeline.

This module implements Scikit-Learn compatible transformers for feature engineering.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class LogReturnTransformer(BaseEstimator, TransformerMixin):
    """
    Computes logarithmic returns from price data.
    
    Formula: ln(P_t) - ln(P_{t-1})
    """
    def __init__(self, price_col: str = 'Close', new_col: str = 'Log_Returns'):
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
