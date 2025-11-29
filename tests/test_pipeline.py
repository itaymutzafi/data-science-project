"""Unit tests for the data science pipeline.

This module tests the core functionality of the src/ modules to ensure
correctness without cluttering the main notebook.
"""

import pytest
import pandas as pd
import numpy as np
from src.baselines import NaiveBaseline, RandomBaseline
from src.evaluation import evaluate_regression, calculate_sharpe_ratio

@pytest.fixture
def sample_data():
    """Generates synthetic data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, size=100)
    df = pd.DataFrame({"Log_Returns": returns}, index=dates)
    return df

def test_sharpe_ratio_calculation():
    """Test Sharpe Ratio logic."""
    # Case 1: Flat returns (0 std) -> 0 Sharpe
    returns = pd.Series([0.01, 0.01, 0.01])
    assert calculate_sharpe_ratio(returns) == 0.0
    
    # Case 2: Positive returns
    returns = pd.Series([0.01, 0.02, 0.015])
    sharpe = calculate_sharpe_ratio(returns)
    assert sharpe > 0

def test_naive_baseline(sample_data):
    """Test Naive Baseline (Zero Strategy)."""
    model = NaiveBaseline(strategy="zero")
    preds = model.predict(sample_data)
    
    assert len(preds) == len(sample_data)
    assert (preds == 0).all()

def test_random_baseline(sample_data):
    """Test Random Baseline statistical properties."""
    model = RandomBaseline(seed=42)
    y = sample_data["Log_Returns"]
    model.fit(y)
    preds = model.predict(sample_data)
    
    assert len(preds) == len(sample_data)
    assert preds.std() > 0  # Should have variance

def test_evaluation_metrics():
    """Test that evaluation metrics return expected keys."""
    y_true = pd.Series([0.01, -0.01, 0.02])
    y_pred = pd.Series([0.00, 0.00, 0.00]) # Naive zero
    
    metrics = evaluate_regression(y_true, y_pred)
    
    expected_keys = ["MSE", "RMSE", "MAE", "R2", "Directional Accuracy", "Strategy Sharpe"]
    for key in expected_keys:
        assert key in metrics
