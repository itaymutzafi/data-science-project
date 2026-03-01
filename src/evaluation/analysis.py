"""Stationarity analysis helpers."""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from src.evaluation.plots import plot_stationarity_check


def check_stationarity(series: pd.Series, name: str) -> None:
    """Run the Augmented Dickey-Fuller test and print the result summary."""
    result = adfuller(series.dropna())
    print(f"\n--- Augmented Dickey-Fuller Test: {name} ---")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value:       {result[1]:.4f}")

    is_stationary = result[1] < 0.05
    status = "Stationary (Reject H0)" if is_stationary else "Non-Stationary (Fail to reject H0)"
    print(f"Result:        {status}")


def run_stationarity_analysis(df: pd.DataFrame, ticker: str = "Target") -> None:
    """Run plotting and ADF tests for price and return stationarity."""
    analysis_df = df.copy()

    if not isinstance(analysis_df.index, pd.DatetimeIndex):
        analysis_df.index = pd.to_datetime(analysis_df.index)

    target = "Log_Return"
    if target not in analysis_df.columns:
        price_col = "Close" if "Close" in analysis_df.columns else "Adj Close"
        if price_col in analysis_df.columns:
            analysis_df[target] = np.log(analysis_df[price_col] / analysis_df[price_col].shift(1))

    analysis_df = analysis_df.dropna()

    plot_stationarity_check(analysis_df, target_col=target, ticker=ticker)

    price_col = "Close" if "Close" in analysis_df.columns else "Adj Close"
    if price_col in analysis_df.columns:
        check_stationarity(analysis_df[price_col], "Raw Price")
    if target in analysis_df.columns:
        check_stationarity(analysis_df[target], "Log Returns")
