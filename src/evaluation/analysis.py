import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from src.evaluation.plots import plot_stationarity_check


def check_stationarity(series: pd.Series, name: str) -> None:
    """Performs the Augmented Dickey-Fuller (ADF) test for stationarity.

    Args:
        series (pd.Series): The time series to test.
        name (str): A label for the series (e.g., "Raw Price").

    Prints:
        ADF Statistic, p-value, and hypothesis test result.
    """
    result = adfuller(series.dropna())
    print(f"\n--- Augmented Dickey-Fuller Test: {name} ---")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value:       {result[1]:.4f}")
    
    is_stationary = result[1] < 0.05
    status = "Stationary (Reject H0)" if is_stationary else "Non-Stationary (Fail to reject H0)"
    print(f"Result:        {status}")


def run_stationarity_analysis(df: pd.DataFrame, ticker: str = "Target") -> None:
    """
    Execute full stationarity analysis workflow:
    1. Calculate Log Returns (if missing).
    2. Visualize Price vs Returns + Rolling Stats.
    3. Perform ADF Hypothesis Tests.
    
    Args:
        df: DataFrame containing 'Close' prices.
        ticker: Ticker symbol for reporting.
    """
    analysis_df = df.copy()
    
    # ensure index
    if not isinstance(analysis_df.index, pd.DatetimeIndex):
        analysis_df.index = pd.to_datetime(analysis_df.index)
    
    # 1. Calc Log Return if needed
    target = "Log_Return"
    if target not in analysis_df.columns:
        price_col = "Close" if "Close" in analysis_df.columns else "Adj Close"
        if price_col in analysis_df.columns:
            analysis_df[target] = np.log(analysis_df[price_col] / analysis_df[price_col].shift(1))
            
    analysis_df = analysis_df.dropna()
    
    # 2. Visualize
    plot_stationarity_check(analysis_df, target_col=target, ticker=ticker)
    
    # 3. Hypothesis Testing (ADF)
    price_col = "Close" if "Close" in analysis_df.columns else "Adj Close"
    if price_col in analysis_df.columns:
        check_stationarity(analysis_df[price_col], "Raw Price")
    if target in analysis_df.columns:
        check_stationarity(analysis_df[target], "Log Returns")
