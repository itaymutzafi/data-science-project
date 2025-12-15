"""Project configuration.

This module defines global constants and project paths.
"""

from pathlib import Path
from datetime import date
from calendar import month_name, day_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR

# Ticker to Company Name Mapping
# Ensures consistency between stock data (Tickers) and News Data (Company Names)
TICKER_TO_COMPANY_MAP = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOG": "Google",
}

TICKERS = list(TICKER_TO_COMPANY_MAP.keys())

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

# Data Settings
TICKER = "AAPL"
BENCHMARK = "^GSPC"  # S&P 500
LOOKBACK_WINDOW = 30
TEST_SIZE = 0.2

# Date Range (None implies dynamic fetching or full available history)
DAYNAMES = list(day_name)
MONTHNAMES = list(month_name)[1:]
START_DATE = date(2020, 1, 1)
END_DATE = date(2025, 12, 3)

SEED = 42

# --- Data Enrichment Config ---
AUX_TICKER_MAP = {
    'QQQ': 'Nasdaq_100',
    '^VIX': 'VIX_Index',
    '^TNX': 'Treasury_10Y',
    'NVDA': 'NVIDIA_Segment_Leader'
}
AUX_DATA_PATH = "data/raw/auxiliary_market_data.parquet"

# Sentiment Data
RAW_NEWS_PATH = "data/raw/news_last_5y.csv"
PROCESSED_SENTIMENT_PATH = "data/processed/daily_sentiment.parquet"
SENTIMENT_CACHE = "data/processed/daily_sentiment_features.csv"
SAMPLES_PER_DAY = 1


