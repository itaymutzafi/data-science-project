"""Baseline models for time series forecasting.

This module contains simple baseline models to serve as performance benchmarks.
"""

import numpy as np
import pandas as pd
from typing import Optional

class NaiveBaseline:
    """
    Predicts the next value based on the assumption of no change (Random Walk).
    For log-returns, the naive assumption is often 0 (price doesn't change).
    Alternatively, it could be the previous day's return (Momentum).
    
    Strategy: Predicts 0.0 for all steps (Martingale assumption for returns).
    """
    def __init__(self, strategy: str = "zero"):
        self.strategy = strategy
        
    def fit(self, X, y=None):
        pass
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Args:
            X: Input features (must have index matching target).
        Returns:
            Series of zeros matching X's index.
        """
        if self.strategy == "zero":
            return pd.Series(0.0, index=X.index)
        elif self.strategy == "last":
            # Assuming X contains the previous return or we need to pass it.
            # For simplicity in this project's pipeline, we'll stick to 'zero' 
            # as the primary naive baseline for *returns*.
            raise NotImplementedError("Last-value strategy for returns requires lag feature access")
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

class RandomBaseline:
    """
    Predicts random values drawn from a normal distribution matching 
    the training data's statistics (mean, std).
    """
    def __init__(self, seed: int = 42):
        self.mu = 0.0
        self.sigma = 1.0
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
    def fit(self, y_train: pd.Series):
        """Learn mean and std from training target."""
        self.mu = y_train.mean()
        self.sigma = y_train.std()
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(
            self.rng.normal(self.mu, self.sigma, size=len(X)),
            index=X.index
        )

class MarketBenchmark:
    """
    Represents a passive 'Buy & Hold' strategy.
    
    For regression evaluation (MSE), this model predicts the historical mean return 
    (Best Constant Predictor), which minimizes MSE for a constant prediction.
    
    For trading evaluation (Sharpe), the constant prediction (assuming it's positive)
    results in a consistent 'Long' signal, effectively mimicking a Buy & Hold strategy
    on the target asset.
    """
    def __init__(self):
        self.mu = 0.0
        
    def fit(self, y_train: pd.Series):
        """Learn the historical mean return."""
        self.mu = y_train.mean()
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict the historical mean for all steps."""
        return pd.Series(self.mu, index=X.index)
