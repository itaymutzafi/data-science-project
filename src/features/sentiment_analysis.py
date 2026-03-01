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
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    TICKER_TO_COMPANY_MAP,
    SENTIMENT_CACHE,
    LEGACY_SENTIMENT_CACHE,
    SAMPLES_PER_DAY,
    SENTIMENT_MA_WINDOW,
    SENTIMENT_MOMENTUM_WINDOW,
    COMPANY_COLORS,
)
from src.data.news_loader import get_google_news_titles, get_news_df_from_file
from src.utils import set_style, apply_academic_style
from transformers.utils import logging as hf_logging
from pandas.errors import EmptyDataError, ParserError

# Sentiment Analysis Configuration
# Note: Feature options are now handled in src.features.sets


# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
hf_logging.set_verbosity_error()

_sentiment_pipeline = None  # Lazy-loaded FinBERT pipeline


def _resolve_sentiment_sample_size(
    *,
    sentiment_depth: int | str | None,
    fallback_n_sample_per_day: int | None,
) -> int | None:
    """Resolve headline sampling depth into per-day sample size.

    Conventions:
    - Integer N >= 0: keep at most N headlines per (date, company).
      N=0 means no cap (use all headlines).
    - Presets:
      - "low"/"quick" -> 1
      - "medium"/"balanced" -> 3
      - "high" -> 5
      - "deep"/"full"/"all" -> 0 (no cap)
    """
    if sentiment_depth is None:
        return fallback_n_sample_per_day

    if isinstance(sentiment_depth, int):
        if sentiment_depth < 0:
            raise ValueError("sentiment_depth integer must be >= 0.")
        return sentiment_depth

    if isinstance(sentiment_depth, str):
        key = sentiment_depth.strip().lower()
        presets = {
            "low": 1,
            "quick": 1,
            "medium": 3,
            "balanced": 3,
            "high": 5,
            "deep": 0,
            "full": 0,
            "all": 0,
        }
        if key not in presets:
            raise ValueError(
                "Invalid sentiment_depth preset. Use one of: "
                f"{sorted(presets.keys())} or an integer >= 0."
            )
        return presets[key]

    raise TypeError("sentiment_depth must be int, str preset, or None.")


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

    # Map device index for pipeline
    device_index = -1
    if isinstance(device, torch.device):
        if device.type == "cuda":
            device_index = 0
        elif device.type == "mps":
            device_index = "mps"
            
    try:
        _sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model=model_name, 
            tokenizer=model_name, 
            device=device_index
        )
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

    def analyze_headlines(
        self,
        df: pd.DataFrame,
        text_col: str = 'final_text',
        batch_size: int = 32,
        show_progress: bool = False
    ) -> pd.DataFrame:
        """
        Runs the sentiment model on the text column.
        Adds 'sentiment_score' (-1 to 1) and 'sentiment_raw_label'.
        """
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found.")
            
        pipeline = _get_pipeline(self.model_name, self.device)
        texts = df[text_col].tolist()
        results = []
        
        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="Sentiment Inference",
            disable=not show_progress
        ):
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
        group[f'sentiment_momentum_{SENTIMENT_MOMENTUM_WINDOW}d'] = group['sentiment_mean'].diff(SENTIMENT_MOMENTUM_WINDOW).fillna(0)
        group[f'sentiment_ma_{SENTIMENT_MA_WINDOW}d'] = group['sentiment_mean'].rolling(SENTIMENT_MA_WINDOW, min_periods=1).mean()
        group['sentiment_trend'] = group[f'sentiment_ma_{SENTIMENT_MA_WINDOW}d'] # Alias
        group[f'sentiment_volatility_{SENTIMENT_MA_WINDOW}d'] = group['sentiment_mean'].rolling(SENTIMENT_MA_WINDOW, min_periods=1).std().fillna(0)
        features.append(group)
        
    return pd.concat(features).reset_index(drop=True)


def get_sentiment_coverage_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates coverage statistics for sentiment data.
    Returns a DataFrame with columns:
    [company, first_day, last_day, total_days, days_with_news, no_news_days,
     tagged_news_coverage_pct, no_news_pct]

    Notes:
    - "coverage" here means days with at least one tagged news item (news_count > 0).
    - This is not a strict mention-rate metric.
    """
    if df.empty:
        return pd.DataFrame()

    required_cols = {"company", "date", "news_count"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns for coverage stats: {sorted(missing_cols)}")

    stats = []
    for company, group in df.groupby('company'):
        total_days = len(group)
        news_days = group[group['news_count'] > 0].shape[0]
        no_news_days = total_days - news_days
        tagged_news_coverage_pct = (news_days / total_days * 100) if total_days > 0 else 0.0
        no_news_pct = (no_news_days / total_days * 100) if total_days > 0 else 0.0

        stats.append({
            'company': company,
            'first_day': group['date'].min(),
            'last_day': group['date'].max(),
            'total_days': total_days,
            'days_with_news': news_days,
            'no_news_days': no_news_days,
            'tagged_news_coverage_pct': tagged_news_coverage_pct,
            'no_news_pct': no_news_pct,
        })

    ordered_cols = [
        "company",
        "first_day",
        "last_day",
        "total_days",
        "days_with_news",
        "no_news_days",
        "tagged_news_coverage_pct",
        "no_news_pct",
    ]
    return pd.DataFrame(stats).sort_values('company')[ordered_cols]


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
    sentiment_depth: int | str | None = None,
    cutoff_date: str = '2020-01-01',
    use_google_news: bool = False,
    force_compute: bool = False
) -> pd.DataFrame:
    """
    Main pipeline to generate daily sentiment features.
    Handles loading, optional Google News patching (disabled by default), text cleaning, scoring, and aggregation.

    Output (per company/date): sentiment_mean, sentiment_std, news_count, market_sentiment,
    sentiment_ma_7d (trend), sentiment_momentum_3d, sentiment_volatility_7d, sentiment_trend (alias of MA7).

    Sampling control:
    - ``sentiment_depth`` (recommended): preset string or int.
      Presets: ``low``/``quick``=1, ``medium``/``balanced``=3, ``high``=5,
      ``deep``/``full``/``all``=0 (no cap, use all headlines).
    - ``n_sample_per_day`` remains for backward compatibility and is used when
      ``sentiment_depth`` is not provided.
    """
    cache_path = Path(output_path) if output_path else Path(SENTIMENT_CACHE)
    legacy_cache_path = None if output_path else Path(LEGACY_SENTIMENT_CACHE)
    cache_candidates = [cache_path]
    if legacy_cache_path and legacy_cache_path != cache_path:
        cache_candidates.append(legacy_cache_path)

    if not force_compute:
        for candidate in cache_candidates:
            if candidate.exists():
                try:
                    if candidate.stat().st_size == 0:
                        logger.warning(f"Skipping empty sentiment cache: {candidate}")
                        continue
                except OSError as exc:
                    logger.warning(f"Skipping unreadable sentiment cache metadata ({candidate}): {exc}")
                    continue

                logger.info(f"Loading cached sentiment: {candidate}")
                try:
                    df = pd.read_csv(candidate)
                except (EmptyDataError, ParserError, UnicodeDecodeError, OSError, ValueError) as exc:
                    logger.warning(f"Skipping invalid sentiment cache ({candidate}): {exc}")
                    continue

                if 'date' not in df.columns:
                    logger.warning(f"Skipping sentiment cache without 'date' column: {candidate}")
                    continue

                try:
                    df['date'] = pd.to_datetime(df['date'])
                except Exception as exc:
                    logger.warning(f"Skipping sentiment cache with invalid date column ({candidate}): {exc}")
                    continue

                # One-time migration from legacy cache to primary layout.
                if candidate != cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(cache_path, index=False)
                return df

    # 1. Load & Clean
    df = load_and_preprocess_news(news_path, cutoff_date=cutoff_date)
    
    # 2. Google News Patch (Simplified)
    if use_google_news:
        logger.info("Fetching recent Google News to fill gaps...")
        gn_frames = []
        # Use configuration map instead of hardcoded list
        for company in TICKER_TO_COMPANY_MAP.values():
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
    effective_n_sample_per_day = _resolve_sentiment_sample_size(
        sentiment_depth=sentiment_depth,
        fallback_n_sample_per_day=n_sample_per_day,
    )
    raw_counts = None
    if effective_n_sample_per_day and effective_n_sample_per_day > 0:
        logger.info(f"Sampling {effective_n_sample_per_day} items per day/company...")
        # Capture raw counts first
        raw_counts = df.groupby(['date', 'company']).size().reset_index(name='news_count_raw')
        df = (
            df.sample(frac=1, random_state=42)
            .groupby(['date', 'company'])
            .head(effective_n_sample_per_day)
        )
    
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
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_cache_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
        final_df.to_csv(tmp_cache_path, index=False)
        tmp_cache_path.replace(cache_path)
        
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
        df = df[df['final_text'].str.contains(company, case=False, na=False)]

    if df.empty:
        logger.warning(f"No valid news found for {company} on {date}")
        return pd.DataFrame(), 0.0
        
    # Analyze
    analyzer = SentimentAnalyzer()
    scored = analyzer.analyze_headlines(df)
    
    daily_score = scored['sentiment_score'].mean()
    return scored.rename(columns={'final_text': 'headline'}), daily_score


# --- Reporting Helpers ---

def display_demo_sentiment(
    news_path: str,
    company: str,
    date: str,
    strict_match: bool = False,
    verbose: bool = False
):
    """Display a concise sentiment summary and a publication-ready demo chart."""
    scored, score = get_demo_day_data(news_path, company, date, strict_match=strict_match)
    if not scored.empty:
        n_items = len(scored)
        mean_score = float(scored["sentiment_score"].mean())
        median_score = float(scored["sentiment_score"].median())
        std_score = float(scored["sentiment_score"].std(ddof=0))

        if "sentiment_raw_label" in scored.columns:
            label_mix = scored["sentiment_raw_label"].value_counts(normalize=True)
            pos_pct = 100 * label_mix.get("positive", 0.0)
            neu_pct = 100 * label_mix.get("neutral", 0.0)
            neg_pct = 100 * label_mix.get("negative", 0.0)
        else:
            pos_pct = neu_pct = neg_pct = 0.0

        if verbose:
            print(f"Sentiment Demo | {company} | {date}")
            print(
                f"Articles analyzed: {n_items} "
                f"(strict_match={strict_match})"
            )
            print(
                f"Daily score (mean): {mean_score:+.3f} | "
                f"Median: {median_score:+.3f} | Std: {std_score:.3f}"
            )
            print(
                f"Label mix: Positive {pos_pct:.1f}% | "
                f"Neutral {neu_pct:.1f}% | Negative {neg_pct:.1f}%"
            )

        plot_day_sentiment_breakdown(scored, date, company, score)
    else:
        if verbose:
            print(
                f"No valid news rows found for company={company}, date={date}, "
                f"strict_match={strict_match}."
            )


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
    elif isinstance(sent_df.index, pd.DatetimeIndex):
        # Accept date-indexed sentiment frames (e.g., from cached pipeline helpers).
        index_name = sent_df.index.name or "index"
        sent_df = sent_df.reset_index().rename(columns={index_name: "date"})
        sent_df["date"] = pd.to_datetime(sent_df["date"])
    else:
        raise ValueError("news_data must contain a 'date' column or a DatetimeIndex")

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

    # Deduplicate sentiment data if needed
    if sent_df.index.duplicated().any():
        # logger might not be available in this scope if defined globally, assuming it is from line 32
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Found {sent_df.index.duplicated().sum()} duplicate timestamps in sentiment data for {target_company}. Aggregating by mean.")
        # Group by date and take mean of numeric columns
        sent_df = sent_df.groupby(level=0).mean(numeric_only=True)
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
    
    # Handle duplicate indices in stock data if any
    if merged.index.duplicated().any():
        logger.warning(f"Found {merged.index.duplicated().sum()} duplicate timestamps in stock data for {company_name}. Keeping first occurrence.")
        merged = merged[~merged.index.duplicated(keep='first')]

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


# --- plots ---
def plot_sentiment_coverage_heatmap(daily_sentiment_df: pd.DataFrame) -> None:
    """
    Visualizes sentiment data availability (coverage) aggregated by month.
    Darker cells indicate higher coverage (more days with news in that month).
    """
    set_style()
    if daily_sentiment_df.empty:
        print("No sentiment data to plot.")
        return
        
    df = daily_sentiment_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Create a monthly pivot table: % of days with news in that month
    df['has_news'] = (df['news_count'] > 0).astype(int)
    
    # Group by Company and Month
    pivot = df.set_index('date').groupby('company').resample('ME')['has_news'].mean().unstack(level=0)
    
    # Handle NaNs (months with no data context)
    pivot = pivot.fillna(0)
    
    # Setup plot
    plt.figure(figsize=(12, 5))
    
    # Create Heatmap
    # cmap="Greens" gives a nice progression from white (0%) to dark green (100%)
    ax = sns.heatmap(pivot.T, cmap="Greens", cbar_kws={'label': 'Coverage %'}, linewidths=0.5, linecolor='#eaeaea')
    
    # Format X-axis to show understandable dates (e.g., "Jan 2020")
    # We define labels based on the columns (dates)
    
    # Get the timestamps from the columns
    date_cols = pivot.index
    
    # Let seaborn/matplotlib handle the dates if possible, or force tick labels
    # Since pivot.index is DatetimeIndex, we can format it.
    
    # Setting readable tick labels: Show every 6th month to avoid crowding
    n_months = len(date_cols)
    step = max(1, n_months // 10) # Aim for ~10 ticks max
    
    xticks = np.arange(0, n_months, step)
    xlabels = [date_cols[i].strftime('%b %Y') for i in xticks]
    
    plt.xticks(xticks + 0.5, xlabels, rotation=0, ha='center', fontsize=10)
    plt.yticks(rotation=0, fontsize=11, fontweight='bold')
    
    plt.title("Sentiment Data Coverage Intensity (Monthly)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("") # Date is obvious
    plt.ylabel("") 
    
    plt.tight_layout()
    plt.show()


def plot_rolling_sentiment_correlation(
    merged_df: pd.DataFrame | dict, 
    sentiment_col: str = 'sentiment_mean_lag1',
    return_col: str = 'Log_Return',
    window: int = 60
) -> None:
    """
    Plots the rolling correlation between sentiment and returns.
    Supports either a single DataFrame (with 'Ticker' col or single asset)
    or a dictionary of DataFrames {ticker: df}.
    """
    set_style()
    plt.figure(figsize=(12, 6))
    
    # Handle Dictionary Input
    if isinstance(merged_df, dict):
        datas = merged_df
        for ticker, df in datas.items():
            df = df.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    continue
            
            if sentiment_col not in df.columns or return_col not in df.columns:
                continue
                
            subset = df.sort_index()
            if len(subset) < window: continue
            
            rolling_corr = subset[sentiment_col].rolling(window).corr(subset[return_col])
            
            # Map ticker to company name for label if possible
            company = TICKER_TO_COMPANY_MAP.get(ticker, ticker)
            color = COMPANY_COLORS.get(company, COMPANY_COLORS.get(ticker, None))
            
            plt.plot(rolling_corr.index, rolling_corr, label=f"{company}", color=color)
            
    # Handle Single DataFrame Input
    else:
        df = merged_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                pass
                
        if sentiment_col not in df.columns or return_col not in df.columns:
            print(f"Columns {sentiment_col} or {return_col} missing.")
            return
        
        tickers = df['Ticker'].unique() if 'Ticker' in df.columns else ['Portfolio']
        
        for ticker in tickers:
            if 'Ticker' in df.columns:
                subset = df[df['Ticker'] == ticker].sort_index()
                label = ticker
            else:
                subset = df.sort_index()
                label = "Portfolio"
                
            if len(subset) < window:
                continue
                
            rolling_corr = subset[sentiment_col].rolling(window).corr(subset[return_col])
            
            company = TICKER_TO_COMPANY_MAP.get(ticker, ticker)
            color = COMPANY_COLORS.get(company, COMPANY_COLORS.get(ticker, None))
            
            plt.plot(rolling_corr.index, rolling_corr, label=f"{company} (Reg={window}d)", color=color)

    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title(f"Rolling {window}-Day Correlation: Sentiment vs. Log Returns", fontsize=14, fontweight='bold')
    plt.ylabel("Correlation Coefficient")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_sentiment_trends(daily_sentiment_df: pd.DataFrame) -> None:
    """Plot raw and smoothed sentiment trends for the four companies."""
    set_style()
    if daily_sentiment_df.empty:
        print("No sentiment data available to plot.")
        return

    df = daily_sentiment_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "sentiment_ma_7d" not in df.columns:
        df["sentiment_ma_7d"] = df.groupby("company")["sentiment_mean"].transform(lambda x: x.rolling(7, min_periods=1).mean())

    companies = ["Apple", "Amazon", "Google", "Microsoft"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    grid_map = {"Apple": 0, "Amazon": 1, "Google": 2, "Microsoft": 3}

    for company in companies:
        if company not in grid_map:
            continue
        ax = axes_flat[grid_map[company]]
        subset = df[df["company"] == company].sort_values("date")
        if subset.empty:
            ax.text(0.5, 0.5, "No Data Available", ha="center", va="center")
            ax.set_title(company, fontweight="bold")
            continue

        color = COMPANY_COLORS.get(company, "blue")
        raw_days = subset[subset["news_count"] > 0]
        if not raw_days.empty:
            ax.scatter(raw_days["date"], raw_days["sentiment_mean"], color=color, alpha=0.3, s=15, label="Daily Raw Sentiment")

        trend_data = subset.dropna(subset=["sentiment_ma_7d"])
        if not trend_data.empty:
            ax.plot(trend_data["date"], trend_data["sentiment_ma_7d"], color=color, linewidth=2.5, label="7-Day Moving Avg")

        ax.set_title(f"{company}", fontweight="bold", fontsize=12)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
        if grid_map[company] == 0:
            ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.suptitle("Sentiment Trends: Raw Signals vs. Smoothed Trends", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()


def plot_day_sentiment_breakdown(daily_news_df: pd.DataFrame, date: str, company: str, daily_score: float) -> None:
    """Show sentiment distribution and the most influential headlines for a single day."""
    set_style()
    if daily_news_df.empty:
        print(f"No news found for {company} on {date}")
        return

    df = (
        daily_news_df.copy()
        .drop_duplicates(subset=["headline"])
        .sort_values(by="sentiment_score", ascending=False)
    )

    scores = df["sentiment_score"].astype(float)
    company_color = COMPANY_COLORS.get(company, "#1F7A8C")
    pos_color = "#0A9396"
    neg_color = "#9B2226"

    top_k = min(6, len(df))
    fig_height = max(6.2, 3.8 + 0.65 * top_k)
    fig = plt.figure(figsize=(14.5, fig_height), constrained_layout=True)
    grid = plt.GridSpec(1, 2, figure=fig, width_ratios=[1.1, 1.0], wspace=0.34)

    # Left panel: score distribution
    ax1 = fig.add_subplot(grid[0, 0])
    sns.histplot(
        scores,
        bins=12,
        kde=True,
        ax=ax1,
        color=company_color,
        alpha=0.55,
        edgecolor="white",
        linewidth=0.7,
    )
    ax1.axvline(daily_score, color="#1B263B", linestyle="--", linewidth=1.9, label=f"Mean = {daily_score:+.2f}")
    ax1.axvline(scores.median(), color="#6c757d", linestyle=":", linewidth=1.6, label=f"Median = {scores.median():+.2f}")
    ax1.set_xlim(-1.0, 1.0)
    ax1.set_xlabel("FinBERT Sentiment Score")
    ax1.set_ylabel("Headline Count")
    ax1.legend(frameon=False, loc="upper left")
    apply_academic_style(ax1, f"Sentiment Distribution | {company} | {date}")

    # Right panel: strongest headlines by absolute score
    ax2 = fig.add_subplot(grid[0, 1])
    strongest = (
        df.assign(abs_score=df["sentiment_score"].abs())
        .nlargest(top_k, "abs_score")
        .sort_values("sentiment_score")
    )
    labels = [textwrap.shorten(h, width=52, placeholder="...") for h in strongest["headline"]]
    values = strongest["sentiment_score"].astype(float).tolist()
    colors = [pos_color if v >= 0 else neg_color for v in values]
    y = np.arange(len(strongest))

    ax2.barh(y, values, color=colors, alpha=0.9, edgecolor="white", linewidth=0.6)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.axvline(0.0, color="#1B263B", linestyle="--", linewidth=1.2, alpha=0.85)
    ax2.set_xlim(-1.0, 1.0)
    ax2.set_xlabel("Headline-Level Score")
    apply_academic_style(ax2, "Most Influential Headlines (by |score|)")

    for yi, v in zip(y, values):
        x = v + (0.03 if v >= 0 else -0.03)
        ha = "left" if v >= 0 else "right"
        ax2.text(x, yi, f"{v:+.2f}", va="center", ha=ha, fontsize=8.8, color="#1f1f1f")

    fig.suptitle("FinBERT Daily Sentiment Breakdown", fontsize=16, fontweight="bold")
    plt.show()
