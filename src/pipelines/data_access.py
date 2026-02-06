from __future__ import annotations

"""Centralized data access helpers to keep notebooks consistent.

These helpers keep a single cached copy of price and sentiment data on first load
and return (by default) defensive copies so downstream code can mutate safely
without accidental cross-talk. This reduces the proliferation of similarly named
variables (e.g., `stocks_data`, `stocks_data_pre`, `master_data`) inside notebooks.
"""

from copy import deepcopy
from typing import Dict, Optional

import pandas as pd

from src.data.loader import fetch_data_for_eda
from src.features.sentiment_analysis import generate_daily_sentiment_features
from src.features.day_month import preprocess_day_feature, preprocess_month_feature
from src.config import START_DATE, END_DATE, RAW_NEWS_PATH

# Module-level caches
_PRICE_CACHE: Optional[Dict[str, pd.DataFrame]] = None
_SENTIMENT_CACHE_DF: Optional[pd.DataFrame] = None


def load_price_data(
    start_date=START_DATE,
    end_date=END_DATE,
    *,
    copy: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Load price data once and reuse it.

    Parameters
    ----------
    start_date : date or str
        Inclusive start date for the fetch. Defaults to global START_DATE.
    end_date : date or str
        Inclusive end date for the fetch. Defaults to global END_DATE.
    copy : bool
        If True (default), returns a deep copy so callers can mutate safely.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping ticker -> price dataframe.
    """
    global _PRICE_CACHE
    if _PRICE_CACHE is None:
        _PRICE_CACHE = fetch_data_for_eda(start_date, end_date)
    return deepcopy(_PRICE_CACHE) if copy else _PRICE_CACHE


def load_sentiment(
    *,
    force_compute: bool = False,
    cutoff_date: str = "2020-01-01",
    copy: bool = True,
) -> pd.DataFrame:
    """Load sentiment features once and reuse them.

    Parameters
    ----------
    force_compute : bool
        Re-run FinBERT scoring even if a cache file exists. Defaults to False.
    cutoff_date : str
        Optional cutoff filter passed to the sentiment pipeline.
    copy : bool
        If True (default), returns a copy.

    Returns
    -------
    pd.DataFrame
        Daily sentiment features indexed by date with a `company` column.
    """
    global _SENTIMENT_CACHE_DF
    if _SENTIMENT_CACHE_DF is None or force_compute:
        _SENTIMENT_CACHE_DF = generate_daily_sentiment_features(
            news_path=RAW_NEWS_PATH,
            output_path=None,  # let pipeline use its default cache handling
            cutoff_date=cutoff_date,
            force_compute=force_compute,
        )
        # Set index to datetime for consistent downstream joins
        if "date" in _SENTIMENT_CACHE_DF.columns:
            _SENTIMENT_CACHE_DF = _SENTIMENT_CACHE_DF.copy()
            _SENTIMENT_CACHE_DF["date"] = pd.to_datetime(_SENTIMENT_CACHE_DF["date"])
            _SENTIMENT_CACHE_DF = _SENTIMENT_CACHE_DF.set_index("date")
    return _SENTIMENT_CACHE_DF.copy() if copy else _SENTIMENT_CACHE_DF


def prepare_model_data(price_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Lightweight helper to create modeling-ready copies.

    Applies day/month cyclic encodings (if present) and drops rows with NaNs to
    keep feature matrices aligned. Assumes feature engineering steps were already
    applied to `price_data`.
    """
    prepared: Dict[str, pd.DataFrame] = {}
    for ticker, df in price_data.items():
        clean_df = preprocess_day_feature(preprocess_month_feature(df.copy()))
        prepared[ticker] = clean_df.dropna()
    return prepared
