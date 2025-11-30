"""Project configuration.

This module defines global constants and project paths.
"""

from pathlib import Path


class DataConfig:
    """Global configuration constants."""
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "data"

    # Data Settings
    TICKER = "AAPL"
    BENCHMARK = "^GSPC"  # S&P 500
    LOOKBACK_WINDOW = 30
    TEST_SIZE = 0.2
    
    # Date Range (None implies dynamic fetching or full available history)
    START_DATE = None
    END_DATE = None

    SEED = 42
