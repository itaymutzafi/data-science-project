"""Tests for feature engineering transformers."""

import pandas as pd
import numpy as np
from src.features.preprocessing import LogReturnTransformer

def test_log_return_transformer(sample_data):
    """Test that LogReturnTransformer correctly computes returns."""
    transformer = LogReturnTransformer(price_col="Close", new_col="Log_Returns_Test")
    df_transformed = transformer.transform(sample_data)
    
    # Check column existence
    assert "Log_Returns_Test" in df_transformed.columns
    
    # Check calculation (first value should be NaN)
    assert pd.isna(df_transformed["Log_Returns_Test"].iloc[0])
    
    # Check values (ignoring the first NaN)
    # We calculated Log_Returns manually in conftest, so they should match
    # Note: sample_data['Log_Returns'] is the ground truth returns we generated
    # But wait, sample_data['Close'] was generated from it.
    # Let's verify the math: log(P_t / P_{t-1})
    
    calculated = df_transformed["Log_Returns_Test"].iloc[1:]
    # Re-calculate manually to be sure
    manual = np.log(sample_data["Close"] / sample_data["Close"].shift(1)).iloc[1:]
    
    pd.testing.assert_series_equal(calculated, manual, check_names=False)
