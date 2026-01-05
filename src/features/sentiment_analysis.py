"""Sentiment analysis module using FinBERT.

This module handles the loading, processing, and sentiment scoring of financial news data.
It provides a pipeline to:
1. Load news from Parquet or CSV.
2. Clean and extract relevant text (headlines).
3. Score sentiment using FinBERT.
4. Aggregate scores to daily features.
5. Apply time-series transformations (decay, market context) for modeling.
"""

import logging
import os
from typing import List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

from src.config import TICKER_TO_COMPANY_MAP, RAW_NEWS_PATH, SENTIMENT_CACHE, SAMPLES_PER_DAY
from src.data.news_loader import get_google_news_titles, get_news_df_from_file
from src.evaluation import plots

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_sentiment_pipeline = None  # Lazy-loaded FinBERT pipeline


def _resolve_device(device_arg: str | int) -> Any:
    """Resolves the computation device (CPU/GPU/MPS)."""
    if device_arg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if isinstance(device_arg, str):
        return torch.device(device_arg)
    return device_arg


def _get_pipeline(model_name: str, device: Any):
    """Loads the FinBERT pipeline lazily."""
    global _sentiment_pipeline
    if _sentiment_pipeline is not None:
        return _sentiment_pipeline

    print("Checking for FinBERT model...")
    
    # Map device index for pipeline
    device_index = -1
    if isinstance(device, torch.device):
        if device.type == "cuda":
            device_index = 0
        elif device.type == "mps":
            device_index = "mps"
            
    try:
        print(f"   -> Loading model ({model_name}) on {device}...")
        _sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model=model_name, 
            tokenizer=model_name, 
            device=device_index
        )
        print("Model loaded.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e
        
    return _sentiment_pipeline


def clean_and_extract_text(row: pd.Series) -> str:
    """
    Extracts the most relevant text from a news row and cleans it.
    Prioritizes Headline > Title > Body (First Sentence).
    Filters out short texts, URLs, and common spam.
    """
    candidates = []
    
    # Priority 1: Headline
    if 'headline' in row and pd.notna(row['headline']):
        candidates.append(str(row['headline']).strip())
    
    # Priority 2: Title
    if 'title' in row and pd.notna(row['title']):
        candidates.append(str(row['title']).strip())
        
    # Priority 3: Body (First Sentence/Paragraph)
    for col in ['text', 'body']:
        if col in row and pd.notna(row[col]):
            val = str(row[col]).strip()
            if len(val) > 50: 
                # Take first line/sentence
                line = val.split('\n')[0].strip()
                if len(line) > 10:
                    candidates.append(line)
                else:
                    candidates.append(val[:200]) # Fallback
            else:
                candidates.append(val)

    # Selection & filtering
    for text in candidates:
        clean = text.replace('"', '').replace("'", "").strip()
        
        # Check constraints
        if len(clean) < 4: continue
        if clean.startswith('http') or clean.count('/') > 2: continue
        
        lower_text = clean.lower()
        if "subscribe" in lower_text and len(clean) < 20: continue
        if "click here" in lower_text: continue
            
        return clean
        
    return ""

def load_and_preprocess_news(
    news_path: str, 
    cutoff_date: Optional[str] = None, 
    company_filter: Optional[str] = None
) -> pd.DataFrame:
    """
    Loads news data using data loader, filters by date/company, and extracts clean text.

    Args:
        news_path: Path to file.
        cutoff_date: Start date filter.
        company_filter: Company name filter.

    Returns:
        pd.DataFrame: DataFrame with 'date', 'company', 'final_text'.
    """
    logger.info(f"Loading news data from {news_path}...")
    try:
        # Use shared loader
        df = get_news_df_from_file(news_path)
    except Exception as e:
        logger.error(f"Failed to load news data: {e}")
        return pd.DataFrame()

    # Standardization
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    df = df.dropna(subset=['date'])

    # Filtering
    if cutoff_date:
        df = df[df['date'] >= pd.to_datetime(cutoff_date)]
    
    if company_filter:
        df = df[df['company'] == company_filter]

    if df.empty:
        logger.warning("News DataFrame is empty after filtering.")
        return df

    # Text Extraction
    logger.info("Extracting and cleaning text...")
    df['final_text'] = df.apply(clean_and_extract_text, axis=1)
    
    # Remove empty rows
    df = df[df['final_text'].str.len() > 3].copy()
    
    return df


class SentimentAnalyzer:
    """Pipeline for analyzing financial news sentiment."""

    def __init__(self, model_name: str = "ProsusAI/finbert", device: str | int = "auto"):
        self.model_name = model_name
        self.device = _resolve_device(device)

    def analyze_headlines(self, df: pd.DataFrame, text_col: str = 'final_text', batch_size: int = 32) -> pd.DataFrame:
        """
        Runs the sentiment model on the text column.
        Adds 'sentiment_score' (-1 to 1) and 'sentiment_raw_label'.
        """
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found.")
            
        pipeline = _get_pipeline(self.model_name, self.device)
        texts = df[text_col].tolist()
        results = []
        
        logger.info(f"Scoring {len(texts)} headlines...")
        for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment Inference"):
            batch = texts[i : i + batch_size]
            # top_k=None returns all scores for calculating continuous value
            batch_results = pipeline(batch, padding=True, truncation=True, top_k=None)
            results.extend(batch_results)
            
        # Post-processing scores
        scores = []
        labels = []
        
        for res in results:
            # res is list of dicts: [{'label': 'positive', 'score': 0.9}, ...]
            res_dict = {item['label'].lower(): item['score'] for item in res}
            
            p_pos = res_dict.get('positive', 0.0)
            p_neg = res_dict.get('negative', 0.0)
            compound = p_pos - p_neg
            
            scores.append(compound)
            
            if compound > 0.1: labels.append('positive')
            elif compound < -0.1: labels.append('negative')
            else: labels.append('neutral')
            
        df = df.copy()
        df['sentiment_score'] = scores
        df['sentiment_raw_label'] = labels
        return df

    def aggregate_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates sentiment scores to daily level."""
        df['date_only'] = df['date'].dt.date
        agg = df.groupby(['date_only', 'company']).agg(
            sentiment_mean=('sentiment_score', 'mean'),
            sentiment_std=('sentiment_score', 'std'),
            news_count=('sentiment_score', 'count')
        ).reset_index()
        
        agg = agg.rename(columns={'date_only': 'date'})
        agg['date'] = pd.to_datetime(agg['date'])
        return agg


def apply_exponential_decay(df: pd.DataFrame, decay_factor: float = 0.85) -> pd.DataFrame:
    """Fills missing sentiment days using exponential decay."""
    df = df.sort_values('date')
    sentiment_vals = df['sentiment_mean'].values
    has_news = df['news_count'].fillna(0).values > 0
    decayed = np.zeros_like(sentiment_vals)
    
    last_val = 0.0
    for i in range(len(sentiment_vals)):
        if has_news[i]:
            last_val = sentiment_vals[i]
        else:
            last_val *= decay_factor
        decayed[i] = last_val
        
    df['sentiment_mean'] = decayed
    return df


def calculate_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Market Context (avg of others) and Advanced Features (MA, Momentum)."""
    
    # 1. Market Context (Vectorized Leave-One-Out)
    daily_stats = df.groupby('date')['sentiment_mean'].agg(['sum', 'count'])
    
    def get_context(row):
        d, val = row['date'], row['sentiment_mean']
        if d not in daily_stats.index: return 0.0
        total, n = daily_stats.loc[d]
        if n <= 1: return 0.0
        return (total - val) / (n - 1)

    df['market_sentiment'] = df.apply(get_context, axis=1)
    
    # 2. Time Series Features per Company
    features = []
    for _, group in df.groupby('company'):
        group = group.sort_values('date')
        group['sentiment_momentum_3d'] = group['sentiment_mean'].diff(3).fillna(0)
        group['sentiment_ma_7d'] = group['sentiment_mean'].rolling(7, min_periods=1).mean()
        group['sentiment_trend'] = group['sentiment_ma_7d'] # Alias
        group['sentiment_volatility_7d'] = group['sentiment_mean'].rolling(7, min_periods=1).std().fillna(0)
        features.append(group)
        
    return pd.concat(features).reset_index(drop=True)


def process_sentiment_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Reindexes to continuous daily range and engineers features."""
    if df.empty: return df
    
    df['date'] = pd.to_datetime(df['date'])
    all_dates = pd.date_range(df['date'].min(), df['date'].max(), freq='D')
    
    processed = []
    for company, group in df.groupby('company'):
        group = group.set_index('date').reindex(all_dates)
        group['company'] = company
        group['news_count'] = group['news_count'].fillna(0)
        group = group.reset_index().rename(columns={'index': 'date'})
        
        # Decay
        group = apply_exponential_decay(group)
        processed.append(group)
        
    full_df = pd.concat(processed, ignore_index=True)
    full_df = calculate_market_features(full_df)
    return full_df


def generate_daily_sentiment_features(
    news_path: str,
    output_path: Optional[str] = None,
    n_sample_per_day: int = SAMPLES_PER_DAY,
    cutoff_date: str = '2020-01-01',
    use_google_news: bool = False,
    force_compute: bool = False
) -> pd.DataFrame:
    """
    Main pipeline to generate daily sentiment features.
    Handles loading, optional Google News patching (disabled by default), text cleaning, scoring, and aggregation.

    Output (per company/date): sentiment_mean, sentiment_std, news_count, market_sentiment,
    sentiment_ma_7d (trend), sentiment_momentum_3d, sentiment_volatility_7d, sentiment_trend (alias of MA7).
    """
    cache_path = output_path or SENTIMENT_CACHE

    if cache_path and os.path.exists(cache_path) and not force_compute:
        logger.info(f"Loading cached sentiment: {cache_path}")
        df = pd.read_csv(cache_path)
        df['date'] = pd.to_datetime(df['date'])
        return df

    # 1. Load & Clean
    df = load_and_preprocess_news(news_path, cutoff_date=cutoff_date)
    
    # 2. Google News Patch (Simplified)
    if use_google_news:
        logger.info("Fetching recent Google News to fill gaps...")
        gn_frames = []
        for company in ["Apple", "Microsoft", "Amazon", "Google"]:
            try:
                gn = get_google_news_titles(f"{company} stock", days=730)
                if not gn.empty:
                    gn['company'] = company
                    gn['headline'] = gn['title']
                    gn_frames.append(gn[['date', 'company', 'headline']])
            except Exception as e:
                logger.warning(f"GN Error ({company}): {e}")
        
        if gn_frames:
            gn_df = pd.concat(gn_frames)
            # Process GN text same way
            gn_df['final_text'] = gn_df.apply(clean_and_extract_text, axis=1)
            gn_df = gn_df[gn_df['final_text'].str.len() > 3]
            df = pd.concat([df, gn_df], ignore_index=True)

    # 3. Stratified Sampling (optional)
    raw_counts = None
    if n_sample_per_day and n_sample_per_day > 0:
        logger.info(f"Sampling {n_sample_per_day} items per day/company...")
        # Capture raw counts first
        raw_counts = df.groupby(['date', 'company']).size().reset_index(name='news_count_raw')
        df = df.sample(frac=1, random_state=42).groupby(['date', 'company']).head(n_sample_per_day)
    
    # 4. Analysis
    analyzer = SentimentAnalyzer()
    scored = analyzer.analyze_headlines(df)
    
    # 5. Aggregation & Feature Engineering
    agg = analyzer.aggregate_daily(scored)
    
    # Restore true counts if sampled
    if raw_counts is not None:
        agg = agg.drop(columns=['news_count']).merge(
            raw_counts.rename(columns={'news_count_raw': 'news_count'}), 
            on=['date', 'company'], 
            how='left'
        )
        
    final_df = process_sentiment_timeseries(agg)
    
    if cache_path:
        final_df.to_csv(cache_path, index=False)
        
    return final_df


def get_demo_day_data(news_path: str, company: str, date: str, strict_match: bool = False) -> Tuple[pd.DataFrame, float]:
    """Retrieves processed and scored headlines for a specific day/company (Demo)."""
    
    # Use standard loader for consistency
    df = load_and_preprocess_news(news_path, company_filter=company)
    
    # Filter specific date
    target_dt = pd.to_datetime(date).date()
    df = df[df['date'].dt.date == target_dt].copy()
    
    # Apply strict matching (optional)
    if strict_match:
        # Simple heuristic: headline must contain company name
        # Note: company_filter in load_and_preprocess_news filters by 'company' column, 
        # this filters by text content if needed, or we can just ignore it if company_filter is robust enough.
        # For now, we'll keep it as a placeholder or simple text check.
        pass

    if df.empty:
        logger.warning(f"No valid news found for {company} on {date}")
        return pd.DataFrame(), 0.0
        
    # Analyze
    analyzer = SentimentAnalyzer()
    scored = analyzer.analyze_headlines(df)
    
    daily_score = scored['sentiment_score'].mean()
    return scored.rename(columns={'final_text': 'headline'}), daily_score


# --- Reporting Helpers ---

def display_demo_sentiment(news_path: str, company: str, date: str, strict_match: bool = False):
    """Wrapper to display demo plots."""
    scored, score = get_demo_day_data(news_path, company, date, strict_match=strict_match)
    if not scored.empty:
        plots.plot_day_sentiment_breakdown(scored, date, company, score)
    else:
        print("No Data.")

def run_sentiment_pipeline_for_report(stock_df: pd.DataFrame, config_obj: Any) -> pd.DataFrame:
    """Wrapper to run the full pipeline in the report context."""
    return generate_daily_sentiment_features(
        news_path=config_obj.RAW_NEWS_PATH,
        output_path=config_obj.SENTIMENT_CACHE,
        n_sample_per_day=config_obj.SAMPLES_PER_DAY,
        use_google_news=False,  # offline-safe default; enable explicitly if needed
        force_compute=getattr(config_obj, 'force_sentiment_compute', False)
    )

def verify_feature_integration(stock_df: pd.DataFrame):
    """Checks for sentiment columns in the final dataframe."""
    expected = ['sentiment_mean', 'news_count', 'sentiment_trend']
    missing = [c for c in expected if c not in stock_df.columns and f"{c}_lag1" not in stock_df.columns]
    
    if missing:
        print(f" Missing columns: {missing}")
    else:
        print(" Sentiment features integrated successfully.")

def verify_unified_data(stock_df: pd.DataFrame, company_name: str):
    """
    Displays a sample of the unified dataset where news data is present.
    Used for verification in notebooks.
    """
    print(f"\n--- Unified Data Verification: {company_name} ---")
    
    # Check for sentiment columns
    cols = [c for c in stock_df.columns if 'sentiment' in c or 'news_count' in c]
    if not cols:
        print("No sentiment features found.")
        return

    # Filter for days with news (using lag1 if available, else standard)
    mask = pd.Series(False, index=stock_df.index)
    
    if 'news_count_lag1' in stock_df.columns:
        mask = stock_df['news_count_lag1'] > 0
    elif 'news_count' in stock_df.columns:
        mask = stock_df['news_count'] > 0
        
    sample = stock_df[mask]
    
    if sample.empty:
        print("No days with active news found in the dataset.")
    else:
        print(f"Found {len(sample)} days with active news.")
        print(sample[cols + ['Close']].head())

def integrate_sentiment_data(
    stock_df: pd.DataFrame,
    news_data: pd.DataFrame,
    *,
    company_name: str | None = None,
    sentiment_columns: Optional[List[str]] = None,
    lag: int = 1,
    include_advanced: bool = True,
    add_alias_score: bool = True,
) -> pd.DataFrame:
    """
    Merge lagged sentiment features into a stock dataframe while avoiding same-day leakage.
    
    Args:
        stock_df: Price/feature dataframe indexed by date.
        news_data: Output of generate_daily_sentiment_features (aggregated + decayed).
        company_name: Optional explicit company name; otherwise derived from ticker if possible.
        sentiment_columns: Explicit list of sentiment columns to include; defaults to all non-id columns.
        lag: Days to shift sentiment features (default 1).
        include_advanced: If False, keep only basic columns (mean/std/count/context/MA7/trend).
        add_alias_score: If True and sentiment_mean_lag{lag} exists, set Sentiment_Score alias.
    """
    if lag < 0:
        raise ValueError("lag must be non-negative")

    # Copy and standardize
    sent_df = news_data.copy()
    if 'date' in sent_df.columns:
        sent_df['date'] = pd.to_datetime(sent_df['date'])
    else:
        raise ValueError("news_data must contain a 'date' column")

    # Determine company
    target_company = company_name
    if target_company is None and 'Symbol' in stock_df.columns:
        target_company = TICKER_TO_COMPANY_MAP.get(stock_df['Symbol'].iloc[0])

    if target_company:
        sent_df = sent_df[sent_df['company'].str.lower() == target_company.lower()]

    if sent_df.empty:
        # Nothing to join; return with zeros for expected columns if provided
        base = stock_df.copy()
        if sentiment_columns:
            lagged_cols = [f"{c}_lag{lag}" for c in sentiment_columns]
            for c in lagged_cols:
                base[c] = 0
        return base

    # Select columns
    if sentiment_columns is None:
        sentiment_columns = [c for c in sent_df.columns if c not in {'company', 'date'}]
        if not include_advanced:
            basic = {
                'sentiment_mean',
                'sentiment_std',
                'news_count',
                'market_sentiment',
                'sentiment_ma_7d',
                'sentiment_trend',
            }
            sentiment_columns = [c for c in sentiment_columns if c in basic]

    # Time alignment and lag
    sent_df = sent_df.set_index('date').sort_index()
    sent_df = sent_df[sentiment_columns]
    lagged = sent_df.shift(lag).add_suffix(f"_lag{lag}")

    # Align to stock index
    merged = stock_df.copy()

    # Drop overlapping lagged columns if they already exist to avoid pandas join errors
    overlap = [c for c in lagged.columns if c in merged.columns]
    if overlap:
        logger.warning(f"Replacing existing sentiment columns on merge: {overlap}")
        merged = merged.drop(columns=overlap)

    if not isinstance(merged.index, pd.DatetimeIndex):
        merged.index = pd.to_datetime(merged.index)
    lagged = lagged.reindex(merged.index)
    lagged = lagged.ffill()
    merged = merged.join(lagged)
    merged[lagged.columns] = merged[lagged.columns].fillna(0)

    # Optional compatibility alias
    if add_alias_score:
        alias_col = f"sentiment_mean_lag{lag}"
        if alias_col in merged.columns and "Sentiment_Score" not in merged.columns:
            merged["Sentiment_Score"] = merged[alias_col]

    return merged


def get_config():
    print(f""" Sentiment Analysis Configuration:
    - Raw News Path: {RAW_NEWS_PATH}
    - Output Cache: {SENTIMENT_CACHE}
    - Samples Per Day: {SAMPLES_PER_DAY}
    """)
