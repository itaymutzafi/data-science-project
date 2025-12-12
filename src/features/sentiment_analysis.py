"""
Sentiment analysis module using FinBERT.
"""
import logging
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import torch
from transformers import pipeline
from tqdm import tqdm
from src.data.news_loader import get_google_news_titles

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_exponential_decay(
    df: pd.DataFrame,
    date_col: str = 'date',
    sentiment_col: str = 'sentiment_mean',
    decay_factor: float = 0.85
) -> pd.DataFrame:
    """
    Applies exponential time decay to fill missing sentiment values.
    S_t = S_new if news exists, else S_{t-1} * lambda
    
    Args:
        df: Input DataFrame with continuous date index per company.
        date_col: Date column name.
        sentiment_col: Sentiment column to decay.
        decay_factor: Lambda decay factor (0 < lambda < 1).
        
    Returns:
        pd.DataFrame: DataFrame with filled sentiment values.
    """
    df = df.copy()
    
    # Iterate to apply decay (easiest way to handle sequential dependence)
    # Vectorized approaches exist but this is clearer for simple recursive decay
    # Optimization: Use grouped apply
    
    sentiment_values = df[sentiment_col].values
    has_news = df['news_count'].values > 0
    decayed_values = np.zeros_like(sentiment_values)
    
    last_val = 0.0
    
    for i in range(len(sentiment_values)):
        if has_news[i]:
            last_val = sentiment_values[i]
        else:
            last_val = last_val * decay_factor
        decayed_values[i] = last_val
        
    df[sentiment_col] = decayed_values
    return df

def calculate_market_context(
    df: pd.DataFrame,
    date_col: str = 'date',
    company_col: str = 'company',
    sentiment_col: str = 'sentiment_mean'
) -> pd.DataFrame:
    """
    Calculates the average sentiment of all OTHER companies for each day.
    
    Args:
        df: Input DataFrame containing all companies.
        date_col: Date column.
        company_col: Company column.
        sentiment_col: Sentiment column.
        
    Returns:
        pd.DataFrame: DataFrame with 'market_sentiment' column merged in.
    """
    # Calculate global daily mean
    daily_market = df.groupby(date_col)[sentiment_col].mean().rename('market_mean')
    
    # Merge back
    df = df.merge(daily_market, on=date_col, how='left')
    
    # For each row, the market context is (Sum_all - Self) / (N-1)
    # But approximate with 'market_mean' is usually sufficient if N is large.
    # For N=4 (AAPL, MSFT, GOOG, AMZN), explicit leave-one-out is better.
    
    # Leave-one-out calculation
    # Sum of all sentiments per day
    daily_sum = df.groupby(date_col)[sentiment_col].sum()
    daily_count = df.groupby(date_col)[sentiment_col].count()
    
    def get_context(row):
        d = row[date_col]
        val = row[sentiment_col]
        total = daily_sum.get(d, 0)
        n = daily_count.get(d, 1)
        
        if n <= 1:
            return 0.0 # No context if alone
        
        return (total - val) / (n - 1)
        
    df['market_sentiment'] = df.apply(get_context, axis=1)
    return df.drop(columns=['market_mean'], errors='ignore')

def calculate_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates advanced sentiment features:
    - Momentum (3-day change)
    - Moving Average (7-day)
    - Volatility (7-day rolling std)
    
    Args:
        df: Input DataFrame (expected to be single company, sorted by date).
        
    Returns:
        pd.DataFrame: DataFrame with new columns.
    """
    # Ensure sorted by date
    df = df.sort_values('date')
    
    # 1. Momentum (3-day difference)
    # How much did sentiment change compared to 3 days ago?
    df['sentiment_momentum_3d'] = df['sentiment_mean'].diff(3)
    
    # 2. Moving Average (7-day)
    # Smooth trend
    df['sentiment_ma_7d'] = df['sentiment_mean'].rolling(window=7, min_periods=1).mean()
    
    # 3. Volatility (7-day rolling std)
    # Uncertainty/Noise measure
    df['sentiment_volatility_7d'] = df['sentiment_mean'].rolling(window=7, min_periods=1).std().fillna(0)
    
    return df

def process_sentiment_timeseries(
    df: pd.DataFrame, 
    date_col: str = 'date',
    company_col: str = 'company',
    decay_factor: float = 0.85
) -> pd.DataFrame:
    """
    Reindexes DataFrame to continuous dates and applies exponential decay for missing values.
    Also adds market context features.
    
    Args:
        df: Input sentiment DataFrame.
        date_col: Name of date column.
        company_col: Name of company column.
        decay_factor: Decay factor for missing news days.
        
    Returns:
        pd.DataFrame: Processed DataFrame with continuous daily range per company.
    """
    if df.empty:
        return df

    # Ensure dates
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Get overall min/max dates
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
    
    companies = df[company_col].unique()
    
    processed_dfs = []
    
    for company in companies:
        # Filter for company
        comp_df = df[df[company_col] == company].set_index(date_col)
        
        # Reindex
        comp_df = comp_df.reindex(all_dates)
        
        # Fill Metadata
        comp_df[company_col] = company
        
        # Fill News Count first (0 for missing days)
        comp_df['news_count'] = comp_df['news_count'].fillna(0)
        
        # Apply Exponential Decay to Sentiment Mean
        # We need to treat NaNs as "no news" -> apply decay
        # Existing values remain, NaNs get decayed
        comp_df = comp_df.reset_index().rename(columns={'index': date_col})
        comp_df = apply_exponential_decay(comp_df, date_col=date_col, sentiment_col='sentiment_mean', decay_factor=decay_factor)
        
        # Calculate Advanced Features (Momentum, Volatility, MA)
        comp_df = calculate_advanced_features(comp_df)
        
        processed_dfs.append(comp_df)
        
    full_df = pd.concat(processed_dfs, ignore_index=True)
    
    # Calculate Market Context
    full_df = calculate_market_context(full_df, date_col=date_col, company_col=company_col, sentiment_col='sentiment_mean')
    
    return full_df

class SentimentAnalyzer:
    """
    Analyzes financial news headlines using the FinBERT model.
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert", device: int = -1):
        """
        Initialize the FinBERT pipeline.
        
        Args:
            model_name (str): Hugging Face model identifier.
            device (int): Device to run on. -1 for CPU, 0+ for GPU. 
                          If -1 is passed and MPS (Mac) is available, it will auto-use MPS.
        """
        logger.info(f"Loading sentiment analysis model: {model_name}...")
        
        # Auto-detect MPS if default device (-1) is detected and MPS is available on Mac
        if device == -1 and torch.backends.mps.is_available():
            logger.info("Apple MPS (Metal Performance Shaders) support detected. Using GPU.")
            self.device = "mps"
        else:
            self.device = device

        try:
            self.pipeline = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, device=self.device)
            logger.info(f"Model loaded successfully on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def analyze_headlines(self, df: pd.DataFrame, text_col: str = 'headline', batch_size: int = 32) -> pd.DataFrame:
        """
        process headlines and extract sentiment.
        
        Args:
            df (pd.DataFrame): Input DataFrame containing headlines.
            text_col (str): Name of the column containing the text.
            batch_size (int): Batch size for inference.
            
        Returns:
            pd.DataFrame: DataFrame with added 'sentiment_score' and 'sentiment_label' columns.
                          (Mapping: Positive=1, Neutral=0, Negative=-1 for numeric score).
        """
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in DataFrame.")

        headlines = df[text_col].astype(str).tolist()
        results = []
        
        logger.info(f"Analyzing {len(headlines)} headlines...")
        
        # Use pipeline's built-in batching if possible, or manual loop for progress bar
        for i in tqdm(range(0, len(headlines), batch_size), desc="Sentiment Analysis"):
            batch = headlines[i : i + batch_size]
            # Truncation=True to handle long headlines (though headlines are usually short)
            batch_results = self.pipeline(batch, padding=True, truncation=True)
            results.extend(batch_results)
            
        # Extract labels and scores
        # FinBERT labels: 'positive', 'negative', 'neutral'
        labels = [r['label'].lower() for r in results]
        
        # Create numeric score mapping
        score_map = {
            'positive': 1,
            'neutral': 0,
            'negative': -1
        }
        
        df = df.copy()
        df['sentiment_raw_label'] = labels
        df['sentiment_score'] = [score_map.get(l, 0) for l in labels]
        
        return df

    def aggregate_daily_sentiment(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """
        Aggregates sentiment scores by day.
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment scores.
            date_col (str): Date column name.
            
        Returns:
            pd.DataFrame: Daily aggregated sentiment with index as date.
        """
        if date_col not in df.columns:
             raise ValueError(f"Column '{date_col}' not found.")
             
        # Ensure date format
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
            
        # Normalize to date only (remove time if present)
        df['date_only'] = df[date_col].dt.date
        
        agg_df = df.groupby('date_only').agg(
            sentiment_mean=('sentiment_score', 'mean'),
            news_count=('sentiment_score', 'count')
        )
        
        agg_df.index = pd.to_datetime(agg_df.index)
        agg_df.index.name = 'date'
        
        return agg_df

def integrate_sentiment_data(
    stock_df: pd.DataFrame, 
    news_data: pd.DataFrame, 
    text_col: str = 'headline',
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Full pipeline: Analyzes news (or takes aggregated features), 
    lags by 1 day, and merges with stock data.
    
    Args:
        stock_df (pd.DataFrame): Main stock DataFrame (index should be date).
        news_data (pd.DataFrame): News DataFrame. Can be either:
                                  1. Raw headlines linked to dates.
                                  2. Pre-aggregated daily sentiment (columns: 'sentiment_mean', 'news_count').
        
    Returns:
        pd.DataFrame: Merged DataFrame with lagged sentiment features.
    """
    
    # Check if input is already aggregated
    required_agg_cols = {'sentiment_mean', 'news_count'}
    is_aggregated = required_agg_cols.issubset(news_data.columns)
    
    if is_aggregated:
        logger.info("Input data appears to be pre-aggregated. Skipping analysis step.")
        daily_sentiment = news_data
        # Ensure index is date
        if not isinstance(daily_sentiment.index, pd.DatetimeIndex):
             # Try to find a date column if one exists and set it
             if date_col in daily_sentiment.columns:
                 daily_sentiment[date_col] = pd.to_datetime(daily_sentiment[date_col])
                 daily_sentiment = daily_sentiment.set_index(date_col)
    else:
        analyzer = SentimentAnalyzer()
        # 1. Analyze
        logger.info("Running sentiment analysis on news data...")
        scored_news = analyzer.analyze_headlines(news_data, text_col=text_col)
        
        # 2. Aggregate
        daily_sentiment = analyzer.aggregate_daily_sentiment(scored_news, date_col=date_col)
    
    # 3. Process Time Series (Decay + Market Context) instead of just Fillna
    # This replaces the fill_missing_sentiment_dates logic if it was called before, but here we do it integrally.
    # Ideally, integrate_sentiment_data should receive raw-ish data and do the processing.
    # However, if 'news_data' is just one company, market context might be partial.
    # Assuming 'news_data' contains all relevant companies.
    
    logger.info("Processing sentiment time series (Exponential Decay + Market Context)...")
    
    # Ensure date is a column, as process_sentiment_timeseries expects it
    if date_col not in daily_sentiment.columns and isinstance(daily_sentiment.index, pd.DatetimeIndex):
         daily_sentiment = daily_sentiment.reset_index()
         
    if 'company' in daily_sentiment.columns:
        # Full multi-company processing
        daily_sentiment = process_sentiment_timeseries(daily_sentiment, date_col=date_col)
    else:
        # Single company fallback (just reindex and decay)
        # Add dummy company col if missing
        daily_sentiment['company'] = 'UNKNOWN'
        daily_sentiment = process_sentiment_timeseries(daily_sentiment, date_col=date_col)
    
    # 4. Lag (Shift by 1 day)
    # We want features from day T to predict day T+1. 
    if not isinstance(stock_df.index, pd.DatetimeIndex):
        stock_df.index = pd.to_datetime(stock_df.index)
        
    # Filter for specific stock if needed (assuming stock_df is for one company)
    # But daily_sentiment might have multiple. We need to merge on Date.
    # If stock_df represents ONE company (e.g. AAPL), we filter daily_sentiment for that company.
    # Usually stock_df is single-asset here.
    
    # Basic check: does stock_df have a company column?
    target_company = None
    if 'Symbol' in stock_df.columns:
        target_company = stock_df['Symbol'].iloc[0]
    elif 'company' in stock_df.columns:
        target_company = stock_df['company'].iloc[0]
    
    # If we found a target company, filter sentiment for it
    if target_company and target_company in daily_sentiment['company'].unique():
        daily_sentiment = daily_sentiment[daily_sentiment['company'] == target_company].copy()
        
    # Set index to date for shifting
    daily_sentiment = daily_sentiment.set_index(date_col)
    
    # Reindex to stock days
    daily_sentiment = daily_sentiment.reindex(stock_df.index)
    
    logger.info("Applying 1-day lag to sentiment features...")
    # Shift features
    features_to_lag = [
        'sentiment_mean', 'news_count', 'market_sentiment',
        'sentiment_momentum_3d', 'sentiment_ma_7d', 'sentiment_volatility_7d'
    ]
    daily_sentiment_lagged = daily_sentiment[features_to_lag].shift(1)
    
    # Rename columns to indicate lag
    daily_sentiment_lagged.columns = [f"{col}_lag1" for col in daily_sentiment_lagged.columns]
    
    # 5. Merge
    logger.info("Merging with stock data...")
    if not isinstance(stock_df.index, pd.DatetimeIndex):
        stock_df.index = pd.to_datetime(stock_df.index)
        
    # Check for overlapping columns
    overlap_cols = stock_df.columns.intersection(daily_sentiment_lagged.columns)
    if not overlap_cols.empty:
        stock_df = stock_df.drop(columns=overlap_cols)

    # Left join
    merged_df = stock_df.join(daily_sentiment_lagged, how='left')
    
    # 6. Fill NaNs (Final Safety)
    # News count -> 0
    # Sentiment -> 0 (Neutral) if start of series has no data
    merged_df['news_count_lag1'] = merged_df['news_count_lag1'].fillna(0)
    merged_df['sentiment_mean_lag1'] = merged_df['sentiment_mean_lag1'].fillna(0)
    merged_df['market_sentiment_lag1'] = merged_df['market_sentiment_lag1'].fillna(0)
    
    # Fill new features
    # Momentum: 0 (no change)
    # MA: 0 (neutral)
    # Volatility: 0 (stable)
    merged_df['sentiment_momentum_3d_lag1'] = merged_df['sentiment_momentum_3d_lag1'].fillna(0)
    merged_df['sentiment_ma_7d_lag1'] = merged_df['sentiment_ma_7d_lag1'].fillna(0)
    merged_df['sentiment_volatility_7d_lag1'] = merged_df['sentiment_volatility_7d_lag1'].fillna(0)
    
    return merged_df

def generate_daily_sentiment_features(
    news_path: str,
    output_path: str = None,
    n_sample_per_day: int = 5,
    cutoff_date: str = None,
    company_filter: str = None,
    use_google_news: bool = False
) -> pd.DataFrame:
    """
    Efficiently processes news data to generate daily sentiment features using stratified sampling.
    Optionally fetches recent Google News to fill data gaps (e.g., 2024-2025).
    
    1. Loads news data.
    2. Fetches Google News (if enabled).
    3. Merges datasets.
    4. Implements intelligent text extraction:
       - Uses 'headline' or 'title' if available.
       - Fallback: Extracts first sentence from 'text' or 'body'.
    5. Stratified Sampling: Ensures coverage for every (date, company) group by taking up to N random samples.
    6. Runs FinBERT sentiment analysis.
    7. Aggregates to daily level.
    
    Args:
        news_path: Path to the raw news CSV.
        output_path: Path to save the aggregated sentiment CSV (cache).
        n_sample_per_day: Number of news items to sample per company-day.
        cutoff_date: Optional filter for start date.
        company_filter: Optional filter for specific company.
        use_google_news: Whether to fetch recent news from Google News RSS.
        
    Returns:
        pd.DataFrame: Daily sentiment features (date, company, sentiment_mean, news_count).
    """
    logger.info(f"Loading news data from {news_path}...")
    
    try:
        # Load all columns first to check for headline/title presence
        # Optimization: use iterator or just load, assuming memory is sufficient for 5y news (usually is)
        df = pd.read_csv(news_path, dtype=str)
    except Exception as e:
        logger.error(f"Failed to load news data: {e}")
        return pd.DataFrame()

    # Google News Fetching
    if use_google_news:
        logger.info("Fetching recent Google News data to fill gaps...")
        google_news_frames = []
        target_companies = ["Apple", "Microsoft", "Amazon", "Google"]
        
        # If filtering for specific company, only fetch for that one
        if company_filter:
            target_companies = [c for c in target_companies if c.lower() == company_filter.lower()] or [company_filter]

        for company in target_companies:
            try:
                logger.info(f"Fetching Google News for {company}...")
                # Fetch last 2 years (approx 730 days) to cover 2024-2025 gap
                gn_df = get_google_news_titles(f"{company} stock", days=730)
                if not gn_df.empty:
                    # Normalize columns to match main df
                    # Google News returns: ['published', 'date', 'title', 'link', 'source']
                    # Main df expects: ['date', 'company', 'text'/'headline']
                    gn_df['company'] = company
                    gn_df['headline'] = gn_df['title'] # Use title as headline
                    # Select relevant columns
                    gn_df = gn_df[['date', 'company', 'headline']]
                    google_news_frames.append(gn_df)
            except Exception as e:
                logger.warning(f"Failed to fetch Google News for {company}: {e}")
        
        if google_news_frames:
            combined_gn = pd.concat(google_news_frames, ignore_index=True)
            # Ensure main df has compatible columns before concat
            # The main df might filter cols earlier, but we read 'dtype=str' so it has all.
            # We strictly need date, company, text/headline.
            logger.info(f"Merging {len(combined_gn)} Google News articles...")
            df = pd.concat([df, combined_gn], ignore_index=True)
        else:
            logger.warning("No Google News data fetched.")

    # Preprocessing Dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    # Filter by date
    if cutoff_date:
        df = df[df['date'] >= pd.to_datetime(cutoff_date)]
        
    # Filter by company
    if company_filter:
        df = df[df['company'] == company_filter]
        
    # Text Extraction Logic
    logger.info("Extracting best available text (Headline -> Title -> Text First Sentence)...")
    
    def extract_text(row):
        # Priority 1: Headline
        if 'headline' in row and pd.notna(row['headline']) and len(str(row['headline']).strip()) > 3:
            return str(row['headline']).strip()
        
        # Priority 2: Title
        if 'title' in row and pd.notna(row['title']) and len(str(row['title']).strip()) > 3:
            return str(row['title']).strip()
            
        # Priority 3: Text / Body (First Sentence)
        candidates = ['text', 'body']
        for col in candidates:
            if col in row and pd.notna(row[col]):
                content = str(row[col]).strip()
                if len(content) > 3:
                    # Extract first sentence (split by newline or period)
                    # Simple heuristic: Split by \n first (common for headlines in body), then by period
                    first_line = content.split('\n')[0]
                    if len(first_line) > 10:
                        return first_line
                    # If first line is short, try splitting by period
                    first_sentence = content.split('.')[0]
                    if len(first_sentence) > 3:
                        return first_sentence + "."
                    return content[:200] # Fallback to first 200 chars
        return ""

    # Vectorized approach is harder with conditional column checks, but we can do a fillna chain
    # Create a 'final_text' column
    df['final_text'] = ""
    
    # 1. Headline
    if 'headline' in df.columns:
        df['final_text'] = df['headline'].fillna("").astype(str).str.strip()
        
    # 2. Title (fill gaps)
    if 'title' in df.columns:
        mask = df['final_text'].str.len() <= 3
        df.loc[mask, 'final_text'] = df.loc[mask, 'title'].fillna("").astype(str).str.strip()
        
    # 3. Text/Body (fill gaps with first sentence)
    for col in ['text', 'body']:
        if col in df.columns:
            mask = df['final_text'].str.len() <= 3
            # Extract first line/sentence efficiently
            extracted = df.loc[mask, col].fillna("").astype(str).str.split('\n').str[0].str.strip()
            # If still empty/short, try split by dot? 
            # (Note: split by dot needs regex to be robust, usually split('\n')[0] is main headline fallback)
            df.loc[mask, 'final_text'] = extracted

    # Filter out empty text
    df = df[df['final_text'].str.len() > 3].copy()
    
    logger.info(f"Total valid headlines before sampling: {len(df)}")
    if len(df) == 0:
        logger.warning("No valid text found after preprocessing.")
        return pd.DataFrame()

    # Stratified Sampling Strategy
    if n_sample_per_day > 0:
        logger.info(f"Applying Stratified Sampling: Up to {n_sample_per_day} items per (Date, Company)...")
        # Shuffle to ensure random selection within groups
        df = df.sample(frac=1, random_state=42)
        # Group by date/company and take top N
        df = df.groupby(['date', 'company']).head(n_sample_per_day)
    else:
        logger.info("No sampling limit set. Processing all available news.")

    logger.info(f"Final Count for Analysis: {len(df)}")

    # Run Analysis
    analyzer = SentimentAnalyzer()
    # Pass 'final_text' as the column to analyze
    scored_df = analyzer.analyze_headlines(df, text_col='final_text')
    
    # Aggregate
    agg_features = scored_df.groupby(['date', 'company']).agg(
        sentiment_mean=('sentiment_score', 'mean'),
        news_count=('sentiment_score', 'count'),
        sentiment_std=('sentiment_score', 'std')
    ).reset_index()
    
    # Gap Filling and Advanced Feature Engineering
    logger.info("Processing sentiment time series (Exponential Decay + Market Context)...")
    agg_features = process_sentiment_timeseries(agg_features)
    
    if output_path:
        logger.info(f"Saving sentiment features to {output_path}...")
        agg_features.to_csv(output_path, index=False)
        
    return agg_features
