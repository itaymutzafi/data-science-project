import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.features.sentiment_analysis import apply_exponential_decay, calculate_market_context, process_sentiment_timeseries, calculate_advanced_features

def test_exponential_decay():
    print("Testing Exponential Decay...")
    dates = pd.date_range(start='2020-01-01', periods=5, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'sentiment_mean': [1.0, np.nan, np.nan, 0.5, np.nan],
        'news_count': [5, 0, 0, 2, 0]
    })
    
    decay_factor = 0.8
    decayed_df = apply_exponential_decay(df, decay_factor=decay_factor)
    res = decayed_df['sentiment_mean'].values
    expected = np.array([1.0, 0.8, 0.64, 0.5, 0.4])
    
    if np.allclose(res, expected):
        print("PASS: Exponential Decay")
    else:
        print("FAIL: Exponential Decay")
        print(f"Expected: {expected}")
        print(f"Got: {res}")

def test_market_context():
    print("\nTesting Market Context...")
    dates = ['2020-01-01', '2020-01-01', '2020-01-01']
    companies = ['A', 'B', 'C']
    sentiments = [1.0, 0.5, -0.5]
    
    df = pd.DataFrame({
        'date': dates,
        'company': companies,
        'sentiment_mean': sentiments
    })
    
    context_df = calculate_market_context(df)
    res = context_df['market_sentiment'].values
    expected = np.array([0.0, 0.25, 0.75])
    
    if np.allclose(res, expected):
        print("PASS: Market Context")
    else:
        print("FAIL: Market Context")

def test_advanced_features():
    print("\nTesting Advanced Features (Momentum, Volatility)...")
    dates = pd.date_range('2020-01-01', periods=10)
    vals = [1, 2, 3, 4, 5, 5, 5, 5, 5, 5]
    df = pd.DataFrame({'date': dates, 'sentiment_mean': vals, 'company': 'A'})
    
    df = calculate_advanced_features(df)
    
    # Momentum 3d: Day 3 (4) - Day 0 (1) = 3
    mom = df['sentiment_momentum_3d'].values
    expected_mom_day3 = 3.0
    
    # Allow small float diff
    if abs(mom[3] - expected_mom_day3) < 1e-9:
        print("PASS: Momentum")
    else:
        print(f"FAIL: Momentum. Expected {expected_mom_day3}, got {mom[3]}")
        
    vol = df['sentiment_volatility_7d'].values
    if not np.isnan(vol[-1]):
        print("PASS: Volatility output exists")
    else:
        print("FAIL: Volatility output is NaN")

if __name__ == "__main__":
    test_exponential_decay()
    test_market_context()
    test_advanced_features()
