"""Baseline models for time series forecasting.

This module contains simple baseline models to serve as performance benchmarks.
"""

import numpy as np
import pandas as pd
from typing import Optional

from .base import BaseModel


class NaiveBaseline(BaseModel):
    """
    Predicts the next value based on the assumption of no change (Random Walk).
    For log-returns, the naive assumption is often 0 (price doesn't change).
    Alternatively, it could be the previous day's return (Momentum).
    
    Strategy: Predicts 0.0 for all steps (Martingale assumption for returns).
    """
    def __init__(self, strategy: str = "zero"):
        self.strategy = strategy
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveBaseline":
        # Naive models don't learn from data, but we accept arguments for API consistency.
        return self
        
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
            raise NotImplementedError("Last-value strategy requires lag feature access.")
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


class RandomBaseline(BaseModel):
    """
    Predicts random values drawn from a normal distribution matching 
    the training data's statistics (mean, std).
    """
    def __init__(self, seed: int = 42):
        self.mu = 0.0
        self.sigma = 1.0
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomBaseline":
        """Learn mean and std from training target.

        Args:
            X: Feature matrix.
            y: Target series.

        Returns:
            self
        """
        self.mu = float(y.mean())
        self.sigma = float(y.std())
        # Reset RNG so each fit starts from the same seed for reproducibility
        self.rng = np.random.default_rng(self.seed)
        return self
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate random predictions aligned to the input index."""
        return pd.Series(
            self.rng.normal(self.mu, self.sigma, size=len(X)),
            index=X.index
        )


class MarketBenchmark(BaseModel):
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
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MarketBenchmark":
        """Learn the historical mean return."""
        self.mu = y.mean()
        return self
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict the historical mean for all steps."""
        return pd.Series(self.mu, index=X.index)


class CAPMBaseline(BaseModel):
    """
    Capital Asset Pricing Model (CAPM) Baseline.
    
    Prediction = Risk_Free_Rate + Beta * (Market_Return - Risk_Free_Rate)
    
    This model assumes that the expected return of an asset is determined by its
    sensitivity to market risk (Beta).
    """
    def __init__(self, beta: Optional[float] = None, risk_free_rate: float = 0.02/252):
        """
        Args:
            beta: Fixed Beta value. If None, it will be estimated during fit.
            risk_free_rate: Daily risk-free rate (default: ~2% annual).
        """
        self.beta = beta
        self.risk_free_rate = risk_free_rate
        self.market_return = 0.0
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CAPMBaseline":
        """
        Fits Beta if not provided, using SPY (Market) returns if available in X.
        Otherwise, defaults to Beta=1.0 (Market Performer).
        """
        if self.beta is None:
            market_col = [c for c in X.columns if 'SPY' in c or 'Market' in c]
            if market_col:
                # Simple covariance/variance estimate
                market_ret = X[market_col[0]]
                covariance = np.cov(y, market_ret)[0][1]
                variance = np.var(market_ret)
                self.beta = covariance / variance if variance != 0 else 1.0
            else:
                self.beta = 1.0 # Default to market beta
                
        # Estimate expected market return (historical mean of market proxy or target if proxy missing)
        market_col = [c for c in X.columns if 'SPY' in c or 'Market' in c]
        if market_col:
             self.market_return = X[market_col[0]].mean()
        else:
             self.market_return = y.mean() # Fallback
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predicts returns using CAPM formula.
        """
        # E[Ri] = Rf + Beta * (E[Rm] - Rf)
        # We use the estimated historical market return as E[Rm]
        
        prediction = self.risk_free_rate + self.beta * (self.market_return - self.risk_free_rate)
        return pd.Series(prediction, index=X.index)


class ClassificationBaseline(BaseModel):
    """Baseline for binary classification tasks."""
    def __init__(self, strategy: str = "random"):
        """
        Args:
            strategy: 'random' (coin flip), 'always_0', 'always_1', or 'majority'
        """
        self.strategy = strategy
        self.majority_class = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ClassificationBaseline":
        if self.strategy == "majority":
            self.majority_class = int(y.mode().iloc[0]) if len(y.mode()) > 0 else 1
        return self
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.strategy == "random":
            return pd.Series(np.random.randint(0, 2, len(X)), index=X.index)
        elif self.strategy == "always_0":
            return pd.Series(0, index=X.index)
        elif self.strategy == "always_1":
            return pd.Series(1, index=X.index)
        elif self.strategy == "majority":
            return pd.Series(self.majority_class, index=X.index)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
