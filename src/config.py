"""Project configuration.

This module defines global constants and project paths.
"""

from pathlib import Path
from datetime import date
from calendar import month_name, day_name

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
AUX_DATA_PATH = "data/raw/auxiliary_market_data.parquet"
RAW_NEWS_PATH = "data/raw/news_last_5y.parquet"
PROCESSED_SENTIMENT_PATH = "data/processed/daily_sentiment.parquet"
SENTIMENT_CACHE = "data/processed/daily_sentiment_features.csv"

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
END_DATE = date(2025, 12, 3)

# Features
AUX_TICKER_MAP = {
    'QQQ': 'Nasdaq_100',
    '^VIX': 'VIX_Index',
    '^TNX': 'Treasury_10Y',
    'NVDA': 'NVIDIA_Segment_Leader'
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
SPLITS = 2

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
    'NVIDIA_Segment_Leader': '#20B2AA' # LightSeaGreen (Distinct from Apple, tech-like)
}
