"""
Feature Set Definitions.

Defines groups of features to be used in experimentation.
"""

from typing import List, Dict

# 1. Price / Volume Only
# Minimalist set, good for establishing if technicals add value
PRICE_FEATURES = [
    "Open", "High", "Low", "Close", "Volume", 
    "Log_Return"
]

# 2. Technical Indicators
# Standard set of momentum and volatility indicators
TECHNICAL_FEATURES = [
    "RSI", 
    "MACD", "MACD_Signal", "MACD_Hist", 
    "ATR", 
    "Bollinger_Upper", "Bollinger_Lower",
    "MA20", "MA50" 
]

# 3. Sentiment Features
# NLP-derived signals
SENTIMENT_FEATURES = [
    "sentiment_mean", 
    "sentiment_std", 
    "news_count", 
    "market_sentiment", # context
    "sentiment_trend", 
    "sentiment_volatility_7d"
]

# 4. Auxiliary / Macro
# Broad market context
MACRO_FEATURES = [
    "VIX_Index", 
    "Treasury_10Y", 
    "Nasdaq_100_Return"
]

# 5. Granular Technicals
MOMENTUM_FEATURES = [
    "RSI", "MACD", "MACD_Signal", "MACD_Hist"
]

VOLATILITY_FEATURES = [
    "ATR", "Bollinger_Upper", "Bollinger_Lower"
]

# 6. Advanced Sentiment
SENTIMENT_SMOOTHED = [
    "sentiment_ma_7d",
    "sentiment_trend", 
    "sentiment_volatility_7d"
]

# 8. Notebook Playground Set (Matches current generation)
NOTEBOOK_FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "Log_Return",
    "MA20", "MA50",
    "MACD", "MACD_Signal", "MACD_Hist",
    "Vol20"
]

# 7. Grandmaster Set (The "Best" Combo)
GRANDMASTER_FEATURES = list(set(
    PRICE_FEATURES + 
    TECHNICAL_FEATURES + 
    SENTIMENT_FEATURES + 
    VOLATILITY_FEATURES
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
    elif name == "VOLATILITY":
        return VOLATILITY_FEATURES
    elif name == "SENTIMENT_SMOOTHED":
        return SENTIMENT_SMOOTHED

    elif name == "GRANDMASTER":
        return GRANDMASTER_FEATURES
    elif name == "NOTEBOOK_FEATURES":
        return NOTEBOOK_FEATURES
    elif name == "ALL":
        # Note: Be careful with huge dimensions
        all_feats = PRICE_FEATURES + TECHNICAL_FEATURES + SENTIMENT_FEATURES + MACRO_FEATURES + SENTIMENT_SMOOTHED
        return list(set(all_feats))
    else:
        # Check if it's a generated random set (handled by caller, but if passed as list, we might need logic)
        raise ValueError(f"Unknown feature set: {name}")

def get_feature_buckets() -> Dict[str, List[str]]:
    """Returns the raw buckets for sampling."""
    return {
        "PRICE": [f for f in PRICE_FEATURES if f != "Log_Return"], # log return is often target or strict feature
        "MOMENTUM": MOMENTUM_FEATURES,
        "VOLATILITY": VOLATILITY_FEATURES,
        "SENTIMENT": SENTIMENT_SMOOTHED,
        "MACRO": MACRO_FEATURES
    }

def generate_diverse_combinations(n: int = 10) -> Dict[str, List[str]]:
    """Generates N random combinations, picking one from each bucket."""
    import random
    buckets = get_feature_buckets()
    combinations = {}
    
    for i in range(n):
        combo = []
        # Always include Log_Return lag as base? Let's stick to buckets.
        # Pick 1 from each
        for category, features in buckets.items():
            if features:
                combo.append(random.choice(features))
        
        # Add Log_Return explicitly if not in bucket (usually standard regressor)
        if "Log_Return" not in combo:
            combo.append("Log_Return")
            
        combinations[f"DIVERSE_RND_{i+1}"] = list(set(combo))
        
    return combinations

def generate_random_subspaces(n: int = 20, min_k: int = 3, max_k: int = 8) -> Dict[str, List[str]]:
    """Generates N random feature subsets of varying size."""
    import random
    
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
        
    return combinations
