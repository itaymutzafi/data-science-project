
import sys
import os
import pandas as pd
import numpy as np
import pprint

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.moving_average import add_ma_features
from src.features.volatility import add_volatility_features
from src.features.external_market import add_macro_features
from src.features.sets import get_feature_buckets, LOGICAL_BLOCKS
from src.config import FEATURE_WINDOWS, VOLATILITY_WINDOWS

# Mock plt.show to prevent blocking during automated tests
import matplotlib.pyplot as plt
plt.show = lambda: None

def test_features_and_sets():
    print("Generating Dummy Data...")
    dates = pd.date_range(start="2020-01-01", periods=300, freq="B")
    data = {
        "Close": np.random.randn(300).cumsum() + 100,
        "Return": np.random.randn(300) * 0.01,
        "VIX_Index": np.random.rand(300) * 20 + 10 # 10 to 30
    }
    df = pd.DataFrame(data, index=dates)
    dfs = {"AAPL": df}
    aux_data = pd.DataFrame(data["VIX_Index"], index=dates, columns=['VIX_Index'])
    
    print("\n[Code] Running Feature Generation...")
    add_ma_features(dfs)
    add_volatility_features(dfs)
    add_macro_features(dfs, aux_data)
    
    generated_cols = set(dfs["AAPL"].columns)
    print(f"Generated Columns: {sorted(list(generated_cols))}")
    
    print("\n[Sets] Verifying Logical Blocks...")
    buckets = get_feature_buckets()
    
    # Check Trend
    expected_trend = {f"MA{w}" for w in FEATURE_WINDOWS}
    actual_trend = set(buckets["TREND"])
    assert expected_trend == actual_trend, f"Trend Mismatch: {expected_trend} vs {actual_trend}"
    
    # Check Volatility
    expected_vol = {f"Vol{w}" for w in VOLATILITY_WINDOWS}
    actual_vol = set(buckets["VOLATILITY"])
    assert expected_vol == actual_vol, f"Vol Mismatch: {expected_vol} vs {actual_vol}"
    
    print("Sets Check Passed.")
    
    print("\n[Consistency] Checking if generated columns match defined sets...")
    # Check if we generated what we expect in sets
    dataset_derived = set()
    dataset_derived.update(actual_trend)
    dataset_derived.update(actual_vol)
    
    # Check if these exist in the DF
    missing = dataset_derived - generated_cols
    assert not missing, f"Columns missing from DF that are in Sets: {missing}"
    
    print("Consistency Check Passed.")
    print("All Systems Go.")

if __name__ == "__main__":
    test_features_and_sets()
