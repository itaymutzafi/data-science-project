from typing import List, Dict, Optional, Tuple
import pandas as pd
import random
from src.config import SENTIMENT_MA_WINDOW, SENTIMENT_MOMENTUM_WINDOW, VOLATILITY_WINDOWS, FEATURE_WINDOWS

BASIC_FEATURES = ["Close"] # chose only one and not ["Open", "High", "Low", "Close"]
VOLUME_FEATURE = ["Volume"]
DIV_FEATURE = ["Dividends"]
SPLIT_FEATURE = ["Stock Splits"]
TIME_FEATURES = ['Day_sin', 'Day_cos', 'Month_sin', 'Month_cos']
RETURN_FEATURES = ['Return', 'Log_Return']
VOL_FEATURE = [f"Vol{win}" for win in VOLATILITY_WINDOWS]
MA_FEATURES = [f"MA{win}" for win in FEATURE_WINDOWS]
MOMENTUM_FEATURES = ['MACD', 'MACD_Signal', 'MACD_Hist']
PEER_FEATURES = {
    "AAPL" : ['MSFT - Close', 'MSFT - Volume', 'MSFT - Log_Return', 
              'AMZN - Close', 'AMZN - Volume', 'AMZN - Log_Return', 
              'GOOG - Close', 'GOOG - Volume', 'GOOG - Log_Return'],
    "MSFT" : ['AAPL - Close', 'AAPL - Volume', 'AAPL - Log_Return', 
              'AMZN - Close', 'AMZN - Volume', 'AMZN - Log_Return', 
              'GOOG - Close', 'GOOG - Volume', 'GOOG - Log_Return'],
    "AMZN" : ['MSFT - Close', 'MSFT - Volume', 'MSFT - Log_Return', 
              'AAPL - Close', 'AAPL - Volume', 'AAPL - Log_Return', 
              'GOOG - Close', 'GOOG - Volume', 'GOOG - Log_Return'],
    "GOOG" : ['MSFT - Close', 'MSFT - Volume', 'MSFT - Log_Return', 
              'AMZN - Close', 'AMZN - Volume', 'AMZN - Log_Return', 
              'AAPL - Close', 'AAPL - Volume', 'AAPL - Log_Return']
}
MACRO_FEATURES = ['NVIDIA_Segment_Leader', 'Nasdaq_100', 'Treasury_10Y', 'VIX_Index', 'VIX_MA20', 'VIX_Gap']
REPORT_FEATURE = ["Days To Nearest Report"]
SENTIMENT_FEATURES = [
    "sentiment_mean_lag1",
    "news_count_lag1",
    "market_sentiment_lag1",
    "sentiment_trend_lag1",
    "sentiment_std_lag1",
    "Sentiment_Score",
    f"sentiment_ma_{SENTIMENT_MA_WINDOW}d_lag1",
    f"sentiment_momentum_{SENTIMENT_MOMENTUM_WINDOW}d_lag1",
    f"sentiment_volatility_{SENTIMENT_MA_WINDOW}d_lag1",
]
PROPHET_FEATURES = ['prophet_prediction_binary', 'prophet_prediction_continuous']

BLOCKS = {
    "Basic": BASIC_FEATURES,
    "Volume": VOLUME_FEATURE,
    "Dividends": DIV_FEATURE,
    "Splits": SPLIT_FEATURE,
    "Time": TIME_FEATURES,
    "Return": RETURN_FEATURES,
    "Volatility": VOL_FEATURE,
    "MovingAverage": MA_FEATURES,
    "Momentum": MOMENTUM_FEATURES,
    "Macro": MACRO_FEATURES,
    "Report": REPORT_FEATURE,
    "Sentiment": SENTIMENT_FEATURES,
    "Prophet": PROPHET_FEATURES,
    # "Peer" is added per ticker
}

def generate_diverse_combinations(dfs: Dict[str, pd.DataFrame], n: int, random_state) -> Dict[str, Dict[int, List[str]]]:
    ticker_diverse_sets = {}

    for ticker, _ in dfs.items():
        ticker_diverse_sets[ticker] = generate_diverse_combination_per_ticker(ticker, n, random_state)
    
    return ticker_diverse_sets


def generate_diverse_combination_per_ticker(ticker: str, n: int = 20, random_state: Optional[int] = 42) -> Dict[int, List[str]]:
    """Generates N random combinations using Logical Blocks strategy.

    Args:
        ticker
        n: Number of combinations.
        random_state: Seed for reproducibility. If None, no seeding is applied.
    """
    if random_state is not None:
        random.seed(random_state)

    combinations = {}

    blocks = BLOCKS.copy()
    blocks["Peer"] = PEER_FEATURES[ticker]
    block_names = list(blocks.keys())
    
    for i in range(n):
        num_blocks = random.randint(2, 4)

        # Always have basic feature
        other_blocks = [b for b in block_names if b != "Basic"]
        selected_block_names = ["Basic"] + random.sample(other_blocks, num_blocks)

        combo = []
        for name in selected_block_names:
            combo.append(random.choice(blocks[name]))

        # Remove potential duplicates and ensure valid list
        combo = list(set(combo))
        
        subset_id = i + 1

        combinations[subset_id] = combo
        
    print(f"[sets] Generated {len(combinations)} diverse combos (seed={random_state}) for {ticker}")

    return combinations


def print_feature_sets(ticker_diverse_sets: Dict[str, Dict[int, List[str]]]):
    for ticker, diverse_sets in ticker_diverse_sets.items():
        print(f"ֿ\nTicker: {ticker}")
        for k, v, in diverse_sets.items():
            print(f"{k}: {v}")


def build_feature_to_block_map() -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    feature_to_block = {}

    blocks = BLOCKS.copy()
    blocks["Peer"] = sorted({f for features in PEER_FEATURES.values() for f in features})

    for block, features in blocks.items():
        for f in features:
            feature_to_block[f] = block

    return feature_to_block, blocks
