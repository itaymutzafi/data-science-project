"""Evaluation module.

Defines the business and technical metrics used in the project.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Any

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Compute (annualized) Sharpe ratio for a daily returns series.
    
    Formula: sqrt(252) * (mean(R) - Rf) / std(R)
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252  # Assuming Rf is annual
    mean_excess_return = excess_returns.mean()
    std_excess_return = excess_returns.std()
    
    if std_excess_return == 0:
        return 0.0
        
    return np.sqrt(252) * (mean_excess_return / std_excess_return)


def evaluate_regression(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Compute regression & business metrics (MSE, R2, DA, Sharpe).
    
    Args:
        y_true: Actual log returns.
        y_pred: Predicted log returns.
        
    Returns:
        Dictionary of metrics.
    """
    # 1. Standard Regression Metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # 2. Business Metrics
    # Directional Accuracy: % of times sign(y_pred) == sign(y_true)
    # We use sign(y) where 0 is considered positive or handled consistently
    correct_direction = np.sign(y_true) == np.sign(y_pred)
    da = np.mean(correct_direction)
    
    # Sharpe Ratio of the STRATEGY (assuming we trade based on prediction sign)
    # Simple strategy: if pred > 0 buy, else sell/hold. 
    # Here we just compute Sharpe of the *predicted* returns to see if the model 
    # captures the distribution properties, OR more commonly, 
    # we compute the Sharpe of a portfolio that follows the model's signals.
    # For this stage (Baselines), we'll just return the Sharpe of the y_true 
    # to show we can calculate it, or maybe the Sharpe of the residuals?
    # Actually, usually we want to know: "If I followed this model, what would be my Sharpe?"
    # Strategy Return = sign(y_pred) * y_true (simplified)
    strategy_returns = np.sign(y_pred) * y_true
    strategy_sharpe = calculate_sharpe_ratio(strategy_returns)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Directional Accuracy": da,
        "Strategy Sharpe": strategy_sharpe
    }


def print_eval(metrics: Dict[str, float], model_name: str = "Model"):
    """Pretty-print evaluation metrics for reports."""
    print(f"\n--- Performance: {model_name} ---")
    print(f"MSE:  {metrics['MSE']:.6f} (Lower is better)")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"MAE:  {metrics['MAE']:.6f}")
    print(f"R2:   {metrics['R2']:.6f} (Higher is better)")
    print(f"DA:   {metrics['Directional Accuracy']:.2%} (Directional Accuracy)")
    print(f"Sharpe: {metrics['Strategy Sharpe']:.4f} (Annualized Strategy Return)")
