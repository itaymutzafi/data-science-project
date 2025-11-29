"""Shared fixtures for the test suite."""

import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_data():
    """Generates synthetic price data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    # Random walk price
    returns = np.random.normal(0, 0.01, size=100)
    price = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        "Close": price,
        "Log_Returns": returns # Ground truth for checking
    }, index=dates)
    return df
