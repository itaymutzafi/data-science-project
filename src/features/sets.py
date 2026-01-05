"""
Feature Set Definitions.

Defines groups of features to be used in experimentation.
"""

from typing import List, Dict, Optional

# 1. Price / Volume Only
# Minimalist set, limited to columns that are actually built in the notebooks
PRICE_FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "Return", "Log_Return",
]

# 2. Technical Indicators (built in notebook/pipeline)
TECHNICAL_FEATURES = [
    "MA20", "MA50",
    "MACD", "MACD_Signal", "MACD_Hist",
    "Vol20",
    "Days To Nearest Report",
]

# 3. Sentiment Features (lagged to avoid leakage)
# Core set from the integrated sentiment pipeline
SENTIMENT_FEATURES = [
    "sentiment_mean_lag1",
    "news_count_lag1",
    "market_sentiment_lag1",
    "sentiment_trend_lag1",
]

# 4. Advanced Sentiment (lagged) - optional extras
SENTIMENT_SMOOTHED = [
    "sentiment_ma_7d_lag1",
    "sentiment_momentum_3d_lag1",
    "sentiment_volatility_7d_lag1",
]

# 5. Macro / Auxiliary (to be generated in notebook)
MACRO_FEATURES = [
    "VIX_Index",
    "Treasury_10Y",
    "Nasdaq_100_Return",
]

# 5. Notebook Playground Set (Matches current generation)
NOTEBOOK_FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "Return", "Log_Return",
    "MA20", "MA50",
    "MACD", "MACD_Signal", "MACD_Hist",
    "Vol20",
    "Days To Nearest Report",
    "sentiment_mean_lag1", "news_count_lag1", "market_sentiment_lag1", "sentiment_trend_lag1",
    "sentiment_ma_7d_lag1", "sentiment_momentum_3d_lag1", "sentiment_volatility_7d_lag1",
]

# 6. Grandmaster Set (The "Best" Combo)
GRANDMASTER_FEATURES = list(set(
    PRICE_FEATURES +
    TECHNICAL_FEATURES +
    SENTIMENT_FEATURES +
    SENTIMENT_SMOOTHED
))

# Combined Sets
def get_feature_set(name: str) -> List[str]:
    name = name.upper()
    if name == "PRICE":
        return PRICE_FEATURES
    elif name == "TECH":
        return TECHNICAL_FEATURES
    elif name == "SENTIMENT":
        return SENTIMENT_FEATURES
    elif name == "PRICE_TECH":
        return list(set(PRICE_FEATURES + TECHNICAL_FEATURES))
    elif name == "PRICE_SENTIMENT":
        return list(set(PRICE_FEATURES + SENTIMENT_FEATURES))
    elif name == "MOMENTUM":
        return MOMENTUM_FEATURES
    elif name == "SENTIMENT_SMOOTHED":
        return SENTIMENT_SMOOTHED
    elif name == "GRANDMASTER":
        return GRANDMASTER_FEATURES
    elif name == "NOTEBOOK_FEATURES":
        return NOTEBOOK_FEATURES
    elif name == "ALL":
        all_feats = PRICE_FEATURES + TECHNICAL_FEATURES + SENTIMENT_FEATURES + SENTIMENT_SMOOTHED + MACRO_FEATURES
        return list(set(all_feats))
    else:
        # Check if it's a generated random set (handled by caller, but if passed as list, we might need logic)
        raise ValueError(f"Unknown feature set: {name}")

def get_feature_buckets() -> Dict[str, List[str]]:
    """Returns the raw buckets for sampling."""
    return {
        "PRICE": [f for f in PRICE_FEATURES if f != "Log_Return"], # log return is often target or strict feature
        "TREND": ["MA20", "MA50"],
        "MOMENTUM": ["MACD", "MACD_Signal", "MACD_Hist"],
        "VOLATILITY": ["Vol20"],
        "SENTIMENT": SENTIMENT_FEATURES + SENTIMENT_SMOOTHED,
        "MACRO": MACRO_FEATURES,
    }


# Logical Blocks for Coherent Feature Selection
LOGICAL_BLOCKS = {
    "Trend": ["MA20", "MA50"],  # Moving averages pair well
    "Momentum": ["MACD", "MACD_Signal", "MACD_Hist"],
    "Volatility": ["Vol20"],
    "Sentiment": SENTIMENT_FEATURES + SENTIMENT_SMOOTHED,  # Slim core + advanced
    "Events": ["Days To Nearest Report"],
    "Macro": MACRO_FEATURES,
}

def generate_diverse_combinations(n: int = 20, random_state: Optional[int] = 42) -> Dict[str, List[str]]:
    """Generates N random combinations using Logical Blocks strategy.

    Args:
        n: Number of combinations.
        random_state: Seed for reproducibility. If None, no seeding is applied.
    """
    import random
    
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
        combinations[f"BLOCKS_{i+1}_{name_suffix}"] = combo
        
    if random_state is not None:
        print(f"[sets] Generated {len(combinations)} diverse combos (seed={random_state}): {list(combinations.keys())[:3]}...")

    return combinations

def generate_random_subspaces(
    n: int = 20,
    min_k: int = 3,
    max_k: int = 8,
    random_state: Optional[int] = 42
) -> Dict[str, List[str]]:
    """Generates N random feature subsets of varying size."""
    import random
    
    if random_state is not None:
        random.seed(random_state)

    # Pool of all available features (excluding potential targets if known, but here we list inputs)
    # We exclude 'Log_Return' from the pool if it's treated as target, but usually lag is fine.
    # Let's ensure we have a broad pool.
    pool = list(set(PRICE_FEATURES + TECHNICAL_FEATURES + SENTIMENT_FEATURES + MACRO_FEATURES + 
                    MOMENTUM_FEATURES + VOLATILITY_FEATURES + SENTIMENT_SMOOTHED))
    
    # Remove duplicates
    pool = list(set(pool))
    
    combinations = {}
    for i in range(n):
        k = random.randint(min_k, min(max_k, len(pool)))
        combo = random.sample(pool, k)
        
        # Randomize selection strictly from pool

        
        combinations[f"RANDOM_SUB_{i+1}"] = combo
        
    if random_state is not None:
        print(f"[sets] Generated {len(combinations)} random subspaces (seed={random_state}): {list(combinations.keys())[:3]}...")

    return combinations
