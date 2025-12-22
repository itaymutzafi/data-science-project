import pandas as pd

from statsmodels.tsa.stattools import adfuller
import numpy as np
from src.models.baselines import NaiveBaseline, RandomBaseline, MarketBenchmark, CAPMBaseline
from src.evaluation.metrics import evaluate_regression
from IPython.display import display
from typing import Dict

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


def run_baseline_analysis(y_train: pd.Series, y_test: pd.Series, X_test: pd.DataFrame, X_train: pd.DataFrame = None) -> None:
    """Runs and compares robust baselines: Naive, Random (Monte Carlo), Market Benchmark, and CAPM.
    
    Args:
        y_train (pd.Series): Training target values.
        y_test (pd.Series): Test target values.
        X_test (pd.DataFrame): Test features.
        X_train (pd.DataFrame, optional): Training features (needed for CAPM).
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

    # 4. CAPM Baseline
    # Requires X_train for beta estimation if available, otherwise defaults
    capm = CAPMBaseline()
    if X_train is not None:
        capm.fit(X_train, y_train)
    else:
        # Fallback if X_train not provided (though it should be)
        capm.fit(pd.DataFrame(index=y_train.index), y_train) 
        
    y_pred_capm = capm.predict(X_test)
    metrics_capm = evaluate_regression(y_test, y_pred_capm)

    # 5. Summary Table
    results_df = pd.DataFrame({
        'Naive (Zero)': metrics_naive,
        'Random (MC Avg)': metrics_random_avg,
        'Market (Buy&Hold)': metrics_market,
        'CAPM': metrics_capm
    }).T

    # Filter for key metrics
    results_df = results_df[['MSE', 'Strategy Sharpe', 'Directional Accuracy']]
    print("--- Baseline Comparison ---")
    display(results_df)


def get_permutation_importance(model, X_val: pd.DataFrame, y_val: pd.Series, n_repeats: int = 5) -> pd.Series:
    """Compute permutation importance: MSE delta per feature when shuffled."""
    pred_base = model.predict(X_val)
    base_idx = y_val.index.intersection(pred_base.index)
    base_mse = ((y_val.loc[base_idx] - pred_base.loc[base_idx]) ** 2).mean()
    importances: Dict[str, float] = {}

    for col in X_val.columns:
        deltas = []
        for _ in range(n_repeats):
            shuffled = X_val.copy()
            shuffled[col] = np.random.permutation(shuffled[col].values)
            pred = model.predict(shuffled)
            idx = y_val.index.intersection(pred.index)
            mse = ((y_val.loc[idx] - pred.loc[idx]) ** 2).mean()
            deltas.append(mse - base_mse)
        importances[col] = float(np.mean(deltas))

    return pd.Series(importances).sort_values(ascending=False)


def permutation_importance(model, X_val: pd.DataFrame, y_val: pd.Series, n_repeats: int = 5) -> pd.Series:
    """Alias for get_permutation_importance."""
    return get_permutation_importance(model, X_val, y_val, n_repeats)
