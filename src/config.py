"""Project configuration (skeleton).

This module defines global constants and project paths. Implementations
and detailed values are intentionally minimal to let the team decide
final naming and values.
"""

from pathlib import Path


class DataConfig:
    """Global configuration constants.

    Populate these fields when the team finalizes the environment and
    dataset choices.
    """
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "data"

    # Example placeholders (edit before use)
    TICKER = "<TICKER>"
    BENCHMARK = "<BENCHMARK>"
    LOOKBACK_WINDOW = 30
    TEST_SIZE = 0.2
    START_DATE = "YYYY-MM-DD"
    END_DATE = "YYYY-MM-DD"

    SEED = 42
