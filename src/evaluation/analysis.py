import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

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

def plot_price_vs_returns(df: pd.DataFrame, target_col: str) -> None:
    """Visualizes the contrast between raw prices and log-returns.

    Generates a dual-axis plot showing the non-stationary price series
    versus the stationary log-returns series.

    Args:
        df (pd.DataFrame): Dataframe containing 'Close' and target_col.
        target_col (str): Name of the log-returns column.
    """
    fig, axes = plt.subplots(2, 1, sharex=True)
    
    # Plot 1: Raw Price
    axes[0].plot(df.index, df['Close'], label='Close Price', color='#1f77b4')
    axes[0].set_title("Raw Price ($P_t$): Non-Stationary", fontweight='bold')
    axes[0].legend(loc='upper left')
    
    # Plot 2: Log Returns
    axes[1].plot(df.index, df[target_col], label='Log Returns', color='#ff7f0e', alpha=0.8)
    axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[1].set_title("Log Returns ($Y_t$): Stationary", fontweight='bold')
    axes[1].legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()
