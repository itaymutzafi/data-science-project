"""Tests for evaluation metrics."""

import pandas as pd
from src.evaluation.metrics import evaluate_regression, calculate_sharpe_ratio

def test_sharpe_ratio_calculation():
    """Test Sharpe Ratio logic."""
    # Case 1: Flat returns (0 std) -> 0 Sharpe
    returns = pd.Series([0.01, 0.01, 0.01])
    assert calculate_sharpe_ratio(returns) == 0.0
    
    # Case 2: Positive returns
    returns = pd.Series([0.01, 0.02, 0.015])
    sharpe = calculate_sharpe_ratio(returns)
    assert sharpe > 0

def test_evaluation_metrics():
    """Test that evaluation metrics return expected keys."""
    y_true = pd.Series([0.01, -0.01, 0.02])
    y_pred = pd.Series([0.00, 0.00, 0.00]) # Naive zero
    
    metrics = evaluate_regression(y_true, y_pred)
    
    expected_keys = ["MSE", "RMSE", "MAE", "R2", "Directional Accuracy", "Strategy Sharpe"]
    for key in expected_keys:
        assert key in metrics
