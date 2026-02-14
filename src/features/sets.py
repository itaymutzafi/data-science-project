from typing import List, Dict, Optional, Tuple, Iterable
import pandas as pd
import random
from src.config import SENTIMENT_MA_WINDOW, SENTIMENT_MOMENTUM_WINDOW, VOLATILITY_WINDOWS, FEATURE_WINDOWS, TICKERS
from src.utils.feature_names import canonicalize_feature_name

BASIC_FEATURES = ["Close"] # chose only one and not ["Open", "High", "Low", "Close"]
VOLUME_FEATURE = ["Volume"]
DIV_FEATURE = ["Dividends"]
SPLIT_FEATURE = ["Stock Splits"]
TIME_FEATURES = ['Day_sin', 'Day_cos', 'Month_sin', 'Month_cos']
RETURN_FEATURES = ['Return', 'Log_Return']
VOL_FEATURE = [f"Vol{win}" for win in VOLATILITY_WINDOWS]
MA_FEATURES = [f"MA{win}" for win in FEATURE_WINDOWS]
MOMENTUM_FEATURES = ['MACD', 'MACD_Signal', 'MACD_Hist', 'Regime_Bull', 'Regime_Strength']


def _build_peer_features() -> Dict[str, List[str]]:
    by_ticker: Dict[str, List[str]] = {}
    peer_cols = ("Close", "Volume", "Log_Return")
    for ticker in TICKERS:
        features: List[str] = []
        for other in TICKERS:
            if other == ticker:
                continue
            for col in peer_cols:
                features.append(canonicalize_feature_name(f"{other} - {col}"))
        by_ticker[ticker] = features
    return by_ticker


PEER_FEATURES = _build_peer_features()
MACRO_FEATURES = [
    'NVDA_Leader',
    'Nasdaq_100',
    'Treasury_10Y',
    'VIX_Index',
    f'VIX_MA{FEATURE_WINDOWS[0]}',
    'VIX_Gap',
]
REPORT_FEATURE = ["Days_To_Nearest_Report"]
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
RSI_FEATURE = ["RSI"]
MA_DIST_FEATURES = [f"Dist_MA{win}" for win in FEATURE_WINDOWS]
INTERACTION_FEATURES = ['Vol_x_Return', 'MACD_x_RSI', 'Trend_x_RSI']

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
    "RSI": RSI_FEATURE,
    "Trend": MA_DIST_FEATURES,
    "Interaction": INTERACTION_FEATURES,
    # "Peer" is added per ticker
}

# Keep sampled sets relatively lean: this improved LR stability in section 6.1
# while preserving enough diversity across runs.
MIN_RANDOM_BLOCKS = 2
MAX_RANDOM_BLOCKS = 4
MA_MULTI_PICK_PROB = 0.65
MAX_MA_FEATURES_PER_SET = 2


def _sample_features_from_block(block_name: str, block_features: List[str]) -> List[str]:
    """
    Sample representative features while controlling within-block collinearity.

    Default behavior is one feature per block. Moving-average windows are the
    explicit exception: we may sample two windows together to preserve
    short-vs-long horizon information.
    """
    if not block_features:
        return []

    if block_name == "MovingAverage" and len(block_features) > 1 and random.random() < MA_MULTI_PICK_PROB:
        k = min(MAX_MA_FEATURES_PER_SET, len(block_features))
        return random.sample(block_features, k)

    return [random.choice(block_features)]


def generate_diverse_combinations(
    dfs: Dict[str, pd.DataFrame],
    n: int,
    random_state: Optional[int],
    *,
    verbose: bool = False,
) -> Dict[str, Dict[int, List[str]]]:
    ticker_diverse_sets = {}

    for ticker, _ in dfs.items():
        ticker_diverse_sets[ticker] = generate_diverse_combination_per_ticker(
            ticker,
            n,
            random_state,
            verbose=verbose,
        )
    
    return ticker_diverse_sets


def generate_diverse_combination_per_ticker(
    ticker: str,
    n: int = 20,
    random_state: Optional[int] = 42,
    *,
    verbose: bool = False,
) -> Dict[int, List[str]]:
    """Generate N random combinations with block-level collinearity control.

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
        min_blocks = min(MIN_RANDOM_BLOCKS, max(1, len(block_names) - 1))
        max_blocks = min(MAX_RANDOM_BLOCKS, max(1, len(block_names) - 1))
        num_blocks = random.randint(min_blocks, max_blocks)

        # Always have basic feature
        # `random.sample` keeps block names unique, so we never pick two features
        # from the same conceptual block except the moving-average exception.
        other_blocks = [b for b in block_names if b != "Basic"]
        selected_block_names = ["Basic"] + random.sample(other_blocks, num_blocks)

        combo = []
        for name in selected_block_names:
            combo.extend(_sample_features_from_block(name, blocks[name]))

        # Remove potential duplicates while preserving deterministic order.
        combo = list(dict.fromkeys(combo))
        
        subset_id = i + 1

        combinations[subset_id] = combo
        
    if verbose:
        print(f"[sets] Generated {len(combinations)} diverse combos (seed={random_state}) for {ticker}")

    return combinations


def print_feature_sets(ticker_diverse_sets: Dict[str, Dict[int, List[str]]]):
    for ticker, diverse_sets in ticker_diverse_sets.items():
        print(f"\nTicker: {ticker}")
        for k, v, in diverse_sets.items():
            print(f"{k}: {v}")


def feature_sets_to_frame(ticker_diverse_sets: Dict[str, Dict[int, List[str]]]) -> pd.DataFrame:
    """Convert sampled feature sets to a compact tabular preview."""
    rows = []
    for ticker, sets_by_ticker in ticker_diverse_sets.items():
        for feature_set_id, features in sets_by_ticker.items():
            rows.append(
                {
                    "Ticker": ticker,
                    "FeatureSet": feature_set_id,
                    "FeatureCount": len(features),
                    "Features": ", ".join(sorted(features)),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["Ticker", "FeatureSet", "FeatureCount", "Features"])
    return (
        pd.DataFrame(rows)
        .sort_values(["Ticker", "FeatureSet"])
        .reset_index(drop=True)
    )


def build_feature_to_block_map() -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    feature_to_block = {}

    blocks = BLOCKS.copy()
    blocks["Peer"] = sorted({f for features in PEER_FEATURES.values() for f in features})

    for block, features in blocks.items():
        for f in features:
            feature_to_block[f] = block

    return feature_to_block, blocks


def get_defined_features_for_ticker(
    ticker: str,
    *,
    enabled_blocks: Optional[Iterable[str]] = None,
    include_peer: bool = True,
) -> set[str]:
    """Return defined feature names for a ticker, optionally filtered by blocks."""
    blocks = BLOCKS.copy()
    if include_peer:
        blocks["Peer"] = PEER_FEATURES.get(ticker, [])

    if enabled_blocks is not None:
        enabled = set(enabled_blocks)
        unknown = enabled - set(blocks.keys())
        if unknown:
            raise ValueError(f"Unknown block names: {sorted(unknown)}")
        blocks = {k: v for k, v in blocks.items() if k in enabled}

    defined: set[str] = set()
    for features in blocks.values():
        defined.update(features)
    return defined


def _is_ignored_column(name: str, ignored: set[str]) -> bool:
    if name in ignored:
        return True
    return name.startswith("Target_")


def audit_features_vs_sets(
    dfs: Dict[str, pd.DataFrame],
    *,
    enabled_blocks: Optional[Iterable[str]] = None,
    include_peer: bool = True,
    ignore_columns: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Compare dataframe columns against feature definitions in sets.py.

    Returns, per ticker:
    - missing_defined: features defined in sets but missing from dataframe
    - unexpected_present: dataframe columns not defined in sets (excluding ignore list)
    """
    default_ignored = {
        "Open",
        "High",
        "Low",
        "Adj Close",
        "Ticker",
        "Symbol",
        "Day",
        "Month",
        "Target",
        "TargetBinary",
        "TargetRegression",
    }
    ignored = set(ignore_columns or ())
    ignored.update(default_ignored)

    report: Dict[str, Dict[str, List[str]]] = {}
    for ticker, df in dfs.items():
        defined = get_defined_features_for_ticker(
            ticker,
            enabled_blocks=enabled_blocks,
            include_peer=include_peer,
        )
        present = set(df.columns)

        missing_defined = sorted(defined - present)
        unexpected_present = sorted(
            c for c in (present - defined)
            if not _is_ignored_column(c, ignored)
        )

        report[ticker] = {
            "missing_defined": missing_defined,
            "unexpected_present": unexpected_present,
        }

    return report


def feature_audit_to_frame(report: Dict[str, Dict[str, List[str]]]) -> pd.DataFrame:
    """Flatten audit dictionary into a compact dataframe summary."""
    rows = []
    for ticker, item in report.items():
        rows.append(
            {
                "Ticker": ticker,
                "MissingCount": len(item.get("missing_defined", [])),
                "UnexpectedCount": len(item.get("unexpected_present", [])),
                "MissingFeatures": ", ".join(item.get("missing_defined", [])),
                "UnexpectedFeatures": ", ".join(item.get("unexpected_present", [])),
            }
        )
    return pd.DataFrame(rows).sort_values("Ticker").reset_index(drop=True)
