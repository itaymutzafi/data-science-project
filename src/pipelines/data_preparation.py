"""Data preparation pipeline.

Build the end-to-end feature matrix by merging price, sentiment, and technical indicators.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from src import config
from src.data import loader
from src.features.indicators import TechnicalIndicators
from src.features.sentiment_analysis import run_sentiment_pipeline_for_report


DEFAULT_CACHE_PATH = config.PROCESSED_DATA_DIR / "final_dataset.parquet"


def _load_price_data(tickers: Iterable[str], start_date, end_date) -> Dict[str, pd.DataFrame]:
    """Fetch price data for each ticker.

    Args:
        tickers: Symbols to fetch.
        start_date: Start date for historical window.
        end_date: End date for historical window.

    Returns:
        Mapping of ticker -> price DataFrame sorted by index.
    """
    data: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        ticker_obj = yf.Ticker(ticker)
        df = loader.fetch_sample_data(ticker_obj, start_time=start_date, end_time=end_date)
        df = df.sort_index()
        data[ticker] = df
    return data


def _apply_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to a single ticker DataFrame.

    Args:
        df: Price DataFrame containing at least Open/High/Low/Close/Volume.

    Returns:
        DataFrame with the original columns plus Log_Return and all indicators.
    """
    indicators = TechnicalIndicators()
    close = df["Close"]

    macd_df = indicators.calculate_macd(close)
    bb_df = indicators.calculate_bollinger_bands(close)
    atr = indicators.calculate_atr(df["High"], df["Low"], df["Close"])
    obv = indicators.calculate_obv(df["Close"], df["Volume"])
    rsi = indicators.calculate_rsi(close)

    enriched = df.copy()
    enriched["Log_Return"] = np.log(enriched["Close"] / enriched["Close"].shift(1))
    enriched = pd.concat([enriched, macd_df, bb_df], axis=1)
    enriched["ATR"] = atr
    enriched["OBV"] = obv
    enriched["RSI"] = rsi
    return enriched


def _collect_sentiment_columns(sentiment_df: pd.DataFrame) -> List[str]:
    """Collect sentiment-related columns, excluding identifiers.

    Args:
        sentiment_df: Full sentiment feature DataFrame.

    Returns:
        List of sentiment feature column names.
    """
    if sentiment_df.empty:
        return []
    return [
        c
        for c in sentiment_df.columns
        if c not in {"company", "date"}
    ]


def _merge_with_sentiment(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    company_name: str,
    sentiment_columns: List[str],
) -> pd.DataFrame:
    """
    Left-join price data with sentiment for a specific company and preserve all sentiment columns.

    Args:
        price_df: Price and technical feature DataFrame for a single ticker.
        sentiment_df: Full sentiment DataFrame covering all companies.
        company_name: Company name that aligns with the sentiment data.
        sentiment_columns: Columns to retain from the sentiment DataFrame.

    Returns:
        DataFrame with price, technical, and sentiment features aligned on the index.
    """
    price_aligned = price_df.copy()
    price_aligned.index = pd.to_datetime(price_aligned.index)

    # Ensure sentiment columns exist even if sentiment data is missing
    if not sentiment_columns:
        sentiment_columns = ["sentiment_mean", "news_count", "market_sentiment", "Sentiment_Score"]

    sent_company = sentiment_df[sentiment_df["company"] == company_name].copy()
    if not sent_company.empty:
        sent_company["date"] = pd.to_datetime(sent_company["date"])
        sent_company = sent_company.set_index("date").sort_index()
        joined = price_aligned.join(sent_company[sentiment_columns], how="left")
    else:
        joined = price_aligned.copy()
        for col in sentiment_columns:
            joined[col] = np.nan

    joined[sentiment_columns] = joined[sentiment_columns].ffill().fillna(0)
    return joined


def build_final_dataset(
    tickers: Iterable[str],
    start_date=None,
    end_date=None,
    use_cache: bool = True,
    cache_path: Path | None = None,
    return_dict: bool = False,
) -> pd.DataFrame | Dict[str, pd.DataFrame]:
    """
    Build the full feature matrix: price + technical indicators + sentiment + target.

    Args:
        tickers: Iterable of ticker symbols.
        start_date: Start date for historical pull.
        end_date: End date for historical pull.
        use_cache: Whether to read/write a cached parquet.
        cache_path: Optional override for cache location.
        return_dict: If True, returns Dict[Ticker, DataFrame] and skips monolithic caching.

    Returns:
        pd.DataFrame or Dict containing OHLCV, technical indicators, sentiment features.
    """
    cache_path = cache_path or DEFAULT_CACHE_PATH
    cache_path = Path(cache_path)
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    if not raw_dir.exists() and not processed_dir.exists():
        print("Warning: Local data directories not found (data/raw, data/processed). Attempting to fetch/generate data...")

    # Cache logic only applies to the monolithic DataFrame
    if use_cache and cache_path.exists() and not return_dict:
        return pd.read_parquet(cache_path)

    start_date = start_date or config.START_DATE
    end_date = end_date or config.END_DATE

    price_data = _load_price_data(tickers, start_date, end_date)
    sentiment_all = run_sentiment_pipeline_for_report(pd.DataFrame(), config)
    sentiment_columns = _collect_sentiment_columns(sentiment_all)

    processed_frames = []
    processed_dict = {}
    
    ticker_to_company = config.TICKER_TO_COMPANY_MAP
    for ticker in tickers:
        company_name = ticker_to_company.get(ticker, ticker)
        df_price = price_data[ticker].copy()
        df_price.index = pd.to_datetime(df_price.index)

        df_with_indicators = _apply_indicators(df_price)
        df_merged = _merge_with_sentiment(df_with_indicators, sentiment_all, company_name, sentiment_columns)

        if "sentiment_mean" in df_merged.columns and "Sentiment_Score" not in df_merged.columns:
            df_merged["Sentiment_Score"] = df_merged["sentiment_mean"]

        df_merged["Target"] = df_merged["Log_Return"].shift(-1)
        df_merged["Ticker"] = ticker

        feature_columns = [c for c in df_merged.columns if c not in {"Ticker"}]
        df_imputed = df_merged.copy()
        df_imputed[feature_columns] = df_imputed[feature_columns].ffill()
        df_clean = df_imputed.dropna(subset=feature_columns + ["Target"])
        
        if return_dict:
            processed_dict[ticker] = df_clean
        else:
            processed_frames.append(df_clean)

    if return_dict:
        return processed_dict

    if not processed_frames:
        raise ValueError("No data available to build the final dataset.")

    full_df = pd.concat(processed_frames, axis=0).sort_index()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_parquet(cache_path)
    return full_df


def get_data_split(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    val_end: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation/test by date.

    Args:
        df: Processed DataFrame with DatetimeIndex.
        train_end: Inclusive end date for training.
        val_end: Inclusive end date for validation (if None, no validation split).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex for date-based splitting.")

    train = df.loc[:train_end]
    if val_end:
        val = df.loc[train_end:val_end].iloc[1:]  # start after train_end to avoid overlap
        test = df.loc[val_end:].iloc[1:]
    else:
        val = pd.DataFrame()
        test = df.loc[train_end:].iloc[1:]

    return train, val, test
