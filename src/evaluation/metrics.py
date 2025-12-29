"""Evaluation module.

Defines the business and technical metrics used in the project.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, precision_score, recall_score, f1_score
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



def calculate_max_drawdown(returns: pd.Series) -> float:
    """Compute maximum drawdown for a returns series."""
    if returns.empty:
        return 0.0
    cumulative = (1 + returns.fillna(0)).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Compute (annualized) Sortino ratio (return / downside deviation)."""
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf
        
    downside_std = downside_returns.std()
    
    if downside_std == 0:
        return np.inf
        
    return np.sqrt(252) * (excess_returns.mean() / downside_std)


def calculate_calmar_ratio(returns: pd.Series) -> float:
    """Compute Calmar Ratio (Annualized Return / Max Drawdown)."""
    if returns.empty:
        return 0.0
        
    max_dd = abs(calculate_max_drawdown(returns))
    if max_dd == 0:
        return 0.0 # or inf, but 0 is safer for plotting
        
    # Annualized return
    # If returns are daily log returns, mean * 252 is approx annual log return
    ann_return = returns.mean() * 252
    
    return ann_return / max_dd


def calculate_profit_factor(returns: pd.Series) -> float:
    """Compute Profit Factor (Gross Profit / Gross Loss)."""
    if returns.empty:
        return 0.0
        
    profits = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    
    if losses == 0:
        return np.inf if profits > 0 else 0.0
        
    return profits / losses


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

    # Binary labels for classification-style metrics (positive vs non-positive)
    y_true_bin = (y_true > 0).astype(int)
    y_pred_bin = (y_pred > 0).astype(int)
    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    
    # Sharpe Ratio of the STRATEGY
    # Strategy: if pred > 0 buy, else sell/hold.
    # We compute the Sharpe of a portfolio that follows the model's signals.
    strategy_returns = np.sign(y_pred) * y_true
    strategy_sharpe = calculate_sharpe_ratio(strategy_returns)
    strategy_sortino = calculate_sortino_ratio(strategy_returns)
    strategy_calmar = calculate_calmar_ratio(strategy_returns)
    profit_factor = calculate_profit_factor(strategy_returns)
    max_dd = calculate_max_drawdown(strategy_returns)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Directional Accuracy": da,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Strategy Sharpe": strategy_sharpe,
        "Strategy Sortino": strategy_sortino,
        "Strategy Calmar": strategy_calmar,
        "Profit Factor": profit_factor,
        "Max Drawdown": max_dd,
        "IC": y_true.corr(y_pred) # Information Coefficient (Pearson)
    }


def print_eval(metrics: Dict[str, float], model_name: str = "Model"):
    """Pretty-print evaluation metrics for reports."""
    print(f"\n--- Performance: {model_name} ---")
    print(f"MSE:  {metrics['MSE']:.6f} (Lower is better)")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"MAE:  {metrics['MAE']:.6f}")
    print(f"R2:   {metrics['R2']:.6f} (Higher is better)")
    print(f"DA:   {metrics['Directional Accuracy']:.2%} (Directional Accuracy)")
    print(f"Precision: {metrics['Precision']:.2%}  Recall: {metrics['Recall']:.2%}  F1: {metrics['F1']:.2%}")
    print(f"IC:   {metrics['IC']:.4f} (Information Coefficient)")
    print(f"Sharpe: {metrics['Strategy Sharpe']:.4f} (Annualized Strategy Return)")
    print(f"Max Drawdown: {metrics['Max Drawdown']:.4f}")


def evaluate_classification(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Compute classification metrics."""
    # Ensure inputs are valid
    if y_true.empty or y_pred.empty:
        return {}
        
    acc = np.mean(y_true == y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1
    }
