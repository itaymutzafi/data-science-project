from typing import List, Dict, Optional
import pandas as pd
import random
from src.features import RETURN_FEATURES, REPORT_FEATURE, MACRO_FEATURES, TIME_FEATURES
from src.config import FEATURE_WINDOWS, VOLATILITY_WINDOWS, SENTIMENT_MA_WINDOW, SENTIMENT_MOMENTUM_WINDOW

BASIC_FEATURES = ["Open", "High", "Low", "Close"]
VOLUME_FEATURE = ["Volume"]
DIV_FEATURE = ["Dividends"]
SPLIT_FEATURE = ["Stock Splits"]

MA_FEATURES = [f"MA{w}" for w in FEATURE_WINDOWS] + ["MACD", "MACD_Signal", "MACD_Hist"]
VOL_FEATURE = [f"Vol{w}" for w in VOLATILITY_WINDOWS]

PRICE_FEATURES = BASIC_FEATURES + VOLUME_FEATURE + RETURN_FEATURES
TECHNICAL_FEATURES = MA_FEATURES + VOL_FEATURE + REPORT_FEATURE

SENTIMENT_FEATURES = [
    "sentiment_mean_lag1",
    "news_count_lag1",
    "market_sentiment_lag1",
    "sentiment_trend_lag1",
]
SENTIMENT_SMOOTHED = [
    f"sentiment_ma_{SENTIMENT_MA_WINDOW}d_lag1",
    f"sentiment_momentum_{SENTIMENT_MOMENTUM_WINDOW}d_lag1",
    f"sentiment_volatility_{SENTIMENT_MA_WINDOW}d_lag1",
]
# Logical Blocks for Coherent Feature Selection
LOGICAL_BLOCKS = {
    "Trend": [f"MA{w}" for w in FEATURE_WINDOWS],  # Moving averages pair well
    "Momentum": ["MACD", "MACD_Signal", "MACD_Hist"],
    "Volatility": [f"Vol{w}" for w in VOLATILITY_WINDOWS],
    "Sentiment": SENTIMENT_FEATURES + SENTIMENT_SMOOTHED,  # Slim core + advanced
    "Events": ["Days To Nearest Report"],
    "Macro": MACRO_FEATURES,
    "Prophet": ["prophet_prediction_binary", 'prophet_prediction_continuous']
}


def get_feature_buckets() -> Dict[str, List[str]]:
    """Returns the raw buckets for sampling."""
    return {
        "PRICE": [f for f in PRICE_FEATURES if f != "Log_Return"], # log return is often target or strict feature
        "TREND": [f"MA{w}" for w in FEATURE_WINDOWS], # Dynamic matching
        "MOMENTUM": ["MACD", "MACD_Signal", "MACD_Hist"],
        "VOLATILITY": [f"Vol{w}" for w in VOLATILITY_WINDOWS],
        "SENTIMENT": SENTIMENT_FEATURES + SENTIMENT_SMOOTHED,
        "MACRO": MACRO_FEATURES, # Now includes VIX_Gap, etc.
    }


def generate_diverse_combinations(n: int = 20, random_state: Optional[int] = 42) -> Dict[str, List[str]]:
    """Generates N random combinations using Logical Blocks strategy.

    Args:
        n: Number of combinations.
        random_state: Seed for reproducibility. If None, no seeding is applied.
    """
    if random_state is not None:
        random.seed(random_state)

    combinations = {}
    block_names = list(LOGICAL_BLOCKS.keys())
    
    # Base features that should likely always be present for context
    base_features = ["Open", "Close", "Volume", "Return", "Log_Return", "Vol20"]
    
    for i in range(n):
        # 1. Select random number of blocks (e.g., 2 to 4 blocks)
        num_blocks = random.randint(2, 4)
        
        # 2. Sample blocks without replacement
        selected_block_names = random.sample(block_names, num_blocks)
        
        # 3. Flatten features
        combo = list(base_features) # Start with base
        for name in selected_block_names:
            combo.extend(LOGICAL_BLOCKS[name])
            
        # 4. Remove potential duplicates and ensure valid list
        combo = list(set(combo))
        
        # Create descriptive name
        # e.g. "RND_Trend_Momentum"
        name_suffix = "_".join([name[:4] for name in selected_block_names])
        subset_name = f"BLOCKS_{i+1}_{name_suffix}"

        combinations[subset_name] = combo
        
    if random_state is not None:
        print(f"[sets] Generated {len(combinations)} diverse combos (seed={random_state}): {list(combinations.keys())[:3]}...")

    return combinations


def print_feature_sets(diverse_sets: Dict[str, List[str]]):
    for k, v, in diverse_sets.items():
        print(k)
        print (v)
        print("-------------")
