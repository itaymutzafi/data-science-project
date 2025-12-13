"""Project configuration.

This module defines global constants and project paths.
"""

from pathlib import Path


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
    "GOOGL": "Google",
    "GOOG": "Google",
    "NVDA": "Nvidia",
    "TSLA": "Tesla"
}

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
START_DATE = None
END_DATE = None

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
SAMPLES_PER_DAY = 5
