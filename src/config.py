"""Project configuration.

This module defines global constants and project paths.
"""

from pathlib import Path
from datetime import date
from calendar import month_name, day_name

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Unified cache layout (primary)
CACHE_DIR = DATA_DIR / "cache"
PRICE_CACHE_DIR = CACHE_DIR / "prices"
MARKET_CACHE_DIR = CACHE_DIR / "market"
PROPHET_CACHE_DIR = CACHE_DIR / "prophet"
SEC_FILINGS_CACHE_DIR = CACHE_DIR / "sec_filings"
SENTIMENT_CACHE_DIR = CACHE_DIR / "sentiment"
BULL_RESULTS_CACHE_DIR = CACHE_DIR / "bull_results"

# Price & auxiliary caches (primary)
RAW_PRICE_DIR = PRICE_CACHE_DIR
AUX_DATA_PATH = MARKET_CACHE_DIR / "auxiliary_market_data.parquet"

# Legacy cache layout (fallback read-paths)
LEGACY_RAW_PRICE_DIR = RAW_DIR
LEGACY_AUX_DATA_PATH = RAW_DIR / "auxiliary_market_data.parquet"
LEGACY_PROPHET_CACHE_DIR = RAW_DIR / "prophet"
LEGACY_SEC_FILINGS_CACHE_DIR = RAW_DIR / "sec_filings"

# News caches (keep location stable; files stay ignored)
RAW_NEWS_PATH = RAW_DIR / "news_last_5y.parquet"

# Sentiment outputs
PROCESSED_SENTIMENT_PATH = PROCESSED_DATA_DIR / "daily_sentiment.parquet"
SENTIMENT_CACHE = SENTIMENT_CACHE_DIR / "daily_sentiment_features.csv"
LEGACY_SENTIMENT_CACHE = PROCESSED_DATA_DIR / "daily_sentiment_features.csv"

# Tickers / Companys
TICKER_TO_COMPANY_MAP = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOG": "Google",
}
COMPANY_TO_TICKERS_MAP = {v : k for k,v in TICKER_TO_COMPANY_MAP.items()}
TICKERS = list(TICKER_TO_COMPANY_MAP.keys())

# Time Range
DAYNAMES = list(day_name)
MONTHNAMES = list(month_name)[1:]
START_DATE = date(2020, 1, 1)
# Warm-up for Window features in order to avoid Nulls.
WARMUP_START_DATE = START_DATE.replace(year=START_DATE.year - 1)
END_DATE = date(2025, 12, 3)
DAYS_IN_YEAR = 365.25

# Features
AUX_TICKER_MAP = {
    'QQQ': 'Nasdaq_100',
    '^VIX': 'VIX_Index',
    '^TNX': 'Treasury_10Y',
    'NVDA': 'NVDA_Leader',
}
SAMPLES_PER_DAY = 1

# Standard Rolling Windows (Unified across all features)
FEATURE_WINDOWS = [20, 50, 200]  # Monthly, Quarterly, Yearly
VOLATILITY_WINDOWS = [20]        # Standard Bollinger/Vol measure
OUTLIER_THRESHOLD = 3.0          # Z-score threshold for outliers
SENTIMENT_MA_WINDOW = 7          # Weekly sentiment trend
SENTIMENT_MOMENTUM_WINDOW = 3    # Short-term sentiment shift

# Models
SEED = 42
DEF_SPLITS = 5
SPLITS = 2#DEF_SPLITS

# Visualization Colors
COMPANY_COLORS = {
    "Apple": "#34A853",      # Green
    "Amazon": "#FF9900",     # Orange/Yellow
    "Google": "#4285F4",     # Blue
    "Microsoft": "#F25022",  # Red
    # Ticker Fallbacks
    "AAPL": "#34A853",
    "AMZN": "#FF9900",
    "GOOG": "#4285F4",
    "GOOGL": "#4285F4",
    "MSFT": "#F25022"
}
AUX_COLORS = {
    'Nasdaq_100': '#663399',          # RebeccaPurple (Rich, distinct index color)
    'VIX_Index': '#E0115F',           # Ruby (Distinct from Microsoft Red, signaling alert)
    'Treasury_10Y': '#708090',        # SlateGray (Neutral, bond-like)
    'NVDA_Leader': '#20B2AA'          # LightSeaGreen (Distinct from Apple, tech-like)
}
