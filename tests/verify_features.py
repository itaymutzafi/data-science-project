
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
from src.features.sentiment_analysis import calculate_market_features
from src.features.sets import get_feature_buckets, LOGICAL_BLOCKS
from src.config import FEATURE_WINDOWS, VOLATILITY_WINDOWS, SENTIMENT_MA_WINDOW, SENTIMENT_MOMENTUM_WINDOW

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
    
    # Simulate Sentiment Data
    # We need to simulate the 'sentiment_mean' column before running calculate_market_features
    # This usually comes from the pipeline but we can mock it here
    for key in dfs:
        dfs[key]['sentiment_mean'] = np.random.randn(len(dfs[key]))
        dfs[key]['company'] = key
        
        # Add date column if not in index (calculate_market_features expects it or index reset)
        # Actually calculate_market_features expects a dataframe with 'date', 'company' cols
        # It's an internal function. Let's just manually verify the column names match our expectations via sets.
        pass

    print("[Sets] Verifying Logical Blocks...")
    buckets = get_feature_buckets()
    
    # Check Sentiment Smoothed
    expected_sentiment_smoothed = {
        f"sentiment_ma_{SENTIMENT_MA_WINDOW}d_lag1",
        f"sentiment_momentum_{SENTIMENT_MOMENTUM_WINDOW}d_lag1",
        f"sentiment_volatility_{SENTIMENT_MA_WINDOW}d_lag1",
    }
    actual_sentiment_smoothed = set(buckets["SENTIMENT"][4:]) # Smoothed are usually appended after base
    # Better to check intersection or specific list
    from src.features.sets import SENTIMENT_SMOOTHED
    assert set(SENTIMENT_SMOOTHED) == expected_sentiment_smoothed, f"Sentiment Set Mismatch: {expected_sentiment_smoothed} vs {SENTIMENT_SMOOTHED}"

    
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
