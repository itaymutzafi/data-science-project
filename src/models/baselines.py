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
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Learn mean and std from training target."""
        self.mu = y.mean()
        self.sigma = y.std()
        
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
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Learn the historical mean return."""
        self.mu = y.mean()
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict the historical mean for all steps."""
        return pd.Series(self.mu, index=X.index)

class CAPMBaseline:
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
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits Beta if not provided, using SPY (Market) returns if available in X_train.
        Otherwise, defaults to Beta=1.0 (Market Performer).
        """
        if self.beta is None:
            # Try to find market return in features (assuming 'SPY_Ret' or similar exists)
            # For this baseline, if we don't have explicit market data in X, 
            # we might just default to 1.0 or calculate from a proxy if we had it.
            # Here we'll assume a simple logic: if 'SPY' column exists, use it.
            # Otherwise, Beta = 1.0.
            
            market_col = [c for c in X_train.columns if 'SPY' in c or 'Market' in c]
            if market_col:
                # Simple covariance/variance estimate
                market_ret = X_train[market_col[0]]
                covariance = np.cov(y_train, market_ret)[0][1]
                variance = np.var(market_ret)
                self.beta = covariance / variance if variance != 0 else 1.0
            else:
                self.beta = 1.0 # Default to market beta
                
        # Estimate expected market return (historical mean of market proxy or target if proxy missing)
        # In a strict CAPM, this should be E[Rm]. We'll use the mean of y_train as a proxy 
        # for "Market" if we don't have a separate market column, effectively making it 
        # similar to MarketBenchmark but with the CAPM formulation structure.
        # Ideally, X_train should have market returns.
        
        market_col = [c for c in X_train.columns if 'SPY' in c or 'Market' in c]
        if market_col:
             self.market_return = X_train[market_col[0]].mean()
        else:
             self.market_return = y_train.mean() # Fallback

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predicts returns using CAPM formula.
        """
        # E[Ri] = Rf + Beta * (E[Rm] - Rf)
        # We use the estimated historical market return as E[Rm]
        
        prediction = self.risk_free_rate + self.beta * (self.market_return - self.risk_free_rate)
        return pd.Series(prediction, index=X.index)
