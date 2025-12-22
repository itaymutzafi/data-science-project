"""Data preparation pipeline.

Build the end-to-end feature matrix by merging price, sentiment, and technical indicators.
"""

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from src import config
from src.data import loader
from src.features.indicators import TechnicalIndicators
from src.features.sentiment_analysis import run_sentiment_pipeline_for_report


DEFAULT_CACHE_PATH = config.PROCESSED_DATA_DIR / "final_dataset.parquet"


def _load_price_data(tickers: Iterable[str], start_date, end_date) -> Dict[str, pd.DataFrame]:
    """Fetch price data for each ticker."""
    data: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        ticker_obj = yf.Ticker(ticker)
        df = loader.fetch_sample_data(ticker_obj, start_time=start_date, end_time=end_date)
        df = df.sort_index()
        data[ticker] = df
    return data


def _apply_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to a single ticker dataframe."""
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


def _merge_with_sentiment(
    price_df: pd.DataFrame, sentiment_df: pd.DataFrame, company_name: str
) -> pd.DataFrame:
    """Left-join price data with sentiment for a specific company."""
    if sentiment_df.empty:
        merged = price_df.copy()
        sentiment_cols = [
            "sentiment_mean",
            "sentiment_std",
            "news_count",
            "sentiment_momentum_3d",
            "sentiment_ma_7d",
            "sentiment_trend",
            "sentiment_volatility_7d",
            "market_sentiment",
        ]
        for col in sentiment_cols:
            merged[col] = 0.0
        return merged

    sent_company = sentiment_df[sentiment_df["company"] == company_name].copy()
    if sent_company.empty:
        merged = price_df.copy()
        sentiment_cols = [
            "sentiment_mean",
            "sentiment_std",
            "news_count",
            "sentiment_momentum_3d",
            "sentiment_ma_7d",
            "sentiment_trend",
            "sentiment_volatility_7d",
            "market_sentiment",
        ]
        for col in sentiment_cols:
            merged[col] = 0.0
        return merged

    sent_company["date"] = pd.to_datetime(sent_company["date"])
    sent_company = sent_company.set_index("date").sort_index()
    merged = price_df.join(sent_company, how="left")

    sentiment_cols = [c for c in merged.columns if c.startswith("sentiment") or c in ("news_count", "market_sentiment")]
    merged[sentiment_cols] = merged[sentiment_cols].ffill().fillna(0)
    return merged


def build_final_dataset(
    tickers: Iterable[str],
    start_date,
    end_date,
    use_cache: bool = True,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """
    Build the full feature matrix: price + sentiment + technical indicators + target.
    """
    cache_path = cache_path or DEFAULT_CACHE_PATH
    cache_path = Path(cache_path)
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    if not raw_dir.exists() and not processed_dir.exists():
        print("Warning: Local data directories not found (data/raw, data/processed). Attempting to fetch/generate data...")

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    price_data = _load_price_data(tickers, start_date, end_date)
    sentiment_all = run_sentiment_pipeline_for_report(pd.DataFrame(), config)

    processed_frames = []
    ticker_to_company = config.TICKER_TO_COMPANY_MAP
    for ticker in tickers:
        company_name = ticker_to_company.get(ticker, ticker)
        df_price = price_data[ticker]
        df_price.index = pd.to_datetime(df_price.index)
        df_price = _apply_indicators(df_price)
        df_merged = _merge_with_sentiment(df_price, sentiment_all, company_name)
        # Alias for readability in downstream EDA
        if "sentiment_mean" in df_merged.columns and "Sentiment_Score" not in df_merged.columns:
            df_merged["Sentiment_Score"] = df_merged["sentiment_mean"]
        df_merged["Target"] = df_merged["Log_Return"].shift(-1)
        df_merged["Ticker"] = ticker
        df_merged = df_merged.dropna()
        processed_frames.append(df_merged)

    full_df = pd.concat(processed_frames, axis=0)
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
