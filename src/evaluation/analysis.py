import pandas as pd

from statsmodels.tsa.stattools import adfuller
import numpy as np
from src.models.baselines import NaiveBaseline, RandomBaseline, MarketBenchmark
from src.evaluation.metrics import evaluate_regression
from IPython.display import display

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


def run_baseline_analysis(y_train: pd.Series, y_test: pd.Series, X_test: pd.DataFrame) -> None:
    """Runs and compares robust baselines: Naive, Random (Monte Carlo), and Market Benchmark.
    
    Args:
        y_train (pd.Series): Training target values.
        y_test (pd.Series): Test target values.
        X_test (pd.DataFrame): Test features (dummy for baselines).
    """
    # 1. Naive Baseline (Zero Return)
    naive = NaiveBaseline(strategy="zero")
    y_pred_naive = naive.predict(X_test)
    metrics_naive = evaluate_regression(y_test, y_pred_naive)

    # 2. Monte Carlo Random Baseline (100 Runs)
    n_simulations = 100
    mse_list = []
    sharpe_list = []

    for i in range(n_simulations):
        random_model = RandomBaseline(seed=i)
        random_model.fit(y_train)
        y_pred_random = random_model.predict(X_test)
        m = evaluate_regression(y_test, y_pred_random)
        mse_list.append(m['MSE'])
        sharpe_list.append(m['Strategy Sharpe'])

    metrics_random_avg = {
        'MSE': np.mean(mse_list),
        'Strategy Sharpe': np.mean(sharpe_list),
        'Directional Accuracy': 0.5 # Expected for random
    }

    # 3. Market Benchmark (Buy & Hold)
    market_bench = MarketBenchmark()
    market_bench.fit(y_train)
    y_pred_market = market_bench.predict(X_test)
    metrics_market = evaluate_regression(y_test, y_pred_market)

    # 4. Summary Table
    results_df = pd.DataFrame({
        'Naive (Zero)': metrics_naive,
        'Random (MC Avg)': metrics_random_avg,
        'Market (Buy&Hold)': metrics_market
    }).T

    # Filter for key metrics
    results_df = results_df[['MSE', 'Strategy Sharpe', 'Directional Accuracy']]
    print("--- Baseline Comparison ---")
    display(results_df)
