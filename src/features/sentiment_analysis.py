"""
Sentiment analysis module using FinBERT.
"""
import logging
from typing import Optional, Tuple
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    
    # 3. Lag (Shift by 1 day)
    # We want features from day T to predict day T+1. 
    # Must reindex to match stock structure FIRST to ensure shift aligns with trading days.
    # Note: This simple approach effectively maps Previous Trading Day -> Current Trading Day.
    # Weekend news might be lost if not aggregated into Monday, but matches "1-day lag" request.
    if not isinstance(stock_df.index, pd.DatetimeIndex):
        stock_df.index = pd.to_datetime(stock_df.index)
        
    daily_sentiment = daily_sentiment.reindex(stock_df.index)
    
    logger.info("Applying 1-day lag to sentiment features...")
    daily_sentiment_lagged = daily_sentiment.shift(1)
    
    # Rename columns to indicate lag
    daily_sentiment_lagged.columns = [f"{col}_lag1" for col in daily_sentiment_lagged.columns]
    
    # 4. Merge
    logger.info("Merging with stock data...")
    if not isinstance(stock_df.index, pd.DatetimeIndex):
        stock_df.index = pd.to_datetime(stock_df.index)
        
    # Check for overlapping columns and drop them from stock_df to allow clean merge (idempotency)
    overlap_cols = stock_df.columns.intersection(daily_sentiment_lagged.columns)
    if not overlap_cols.empty:
        logger.info(f"Dropping existing sentiment columns from stock data to avoid duplicates: {overlap_cols.tolist()}")
        stock_df = stock_df.drop(columns=overlap_cols)

    # Left join to keep all stock rows
    merged_df = stock_df.join(daily_sentiment_lagged, how='left')
    
    # 5. Fill NaNs
    # For news count, fill with 0 (no news). For sentiment, maybe 0 (neutral) or forward fill.
    # Let's simple fill 0 for now as 'no news' implies neutral signal or no signal.
    merged_df['news_count_lag1'] = merged_df['news_count_lag1'].fillna(0)
    merged_df['sentiment_mean_lag1'] = merged_df['sentiment_mean_lag1'].fillna(0)
    
    return merged_df

def generate_daily_sentiment_features(
    news_path: str,
    output_path: str = None,
    n_sample_per_day: int = 5,
    sampling_frac: float = None,
    cutoff_date: str = None,
    company_filter: str = None
) -> pd.DataFrame:
    """
    Efficiently processes news data to generate daily sentiment features using sampling.
    
    1. Loads news data (CSV).
    2. Filters by date/company.
    3. Groups by Date + Company and samples N headlines.
    4. Runs FinBERT.
    5. Aggregates to daily level.
    6. Saves to cache (optional).
    
    Args:
        news_path: Path to the raw news CSV.
        output_path: Path to save the aggregated sentiment CSV (cache).
        n_sample_per_day: Number of news items to sample per company-day.
        sampling_frac: Fraction of total headlines to sample (0.0 to 1.0). Applied before n_sample_per_day.
        cutoff_date: Optional filter for start date.
        company_filter: Optional filter for specific company.
        
    Returns:
        pd.DataFrame: Daily sentiment features (date, company, sentiment_mean, news_count).
    """
    logger.info(f"Loading news data from {news_path}...")
    # Load only necessary columns to save memory
    try:
        df = pd.read_csv(news_path, usecols=['date', 'company', 'text'], dtype=str)
    except Exception as e:
        logger.error(f"Failed to load news data: {e}")
        return pd.DataFrame()

    # Preprocessing
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'text'])
    
    # Filter by date
    if cutoff_date:
        df = df[df['date'] >= pd.to_datetime(cutoff_date)]
        
    # Filter by company
    if company_filter:
        df = df[df['company'] == company_filter]
        
    # Preprocessing: Extract headline from text (first sentence/line)
    # This ensures we handle article bodies by taking the title/first sentence
    logger.info("Extracting headlines (first sentence of article)...")
    df['text'] = df['text'].astype(str).str.split('\n').str[0].str.strip()
    # Drop empty texts
    df = df[df['text'].str.len() > 3] 

    logger.info(f"Total headlines before sampling: {len(df)}")
    
    if len(df) == 0:
        logger.warning(f"No headlines found after filtering! Check news_path or cutoff_date. News Path: {news_path}")
        return pd.DataFrame(columns=['date', 'company', 'sentiment_mean', 'news_count', 'sentiment_std'])

    # Sampling Strategy
    # Case A: Hybrid Sampling (Percentage but ensure coverage)
    if n_sample_per_day == 0 and sampling_frac and 0 < sampling_frac < 1.0:
         logger.info(f"Applying Hybrid Sampling: Ensure at least 1 per day + ~{sampling_frac*100}% of valid news...")
         
         # Shuffle first to be random
         df = df.sample(frac=1, random_state=42)
         
         # 1. Mandatory Coverage: Take first item from every (date, company) group
         # This fixes the issue where random sampling misses entire days/companies
         mandatory = df.groupby(['date', 'company']).head(1)
         
         # 2. Variable Sampling: Sample 'frac' from the REST
         remainder = df.drop(mandatory.index)
         if len(remainder) > 0:
             # We sample 'frac' percent of the *remainder* to add to the mandatory set
             # Or 'frac' of total? User said "randomly sample percent". 
             # Let's simple sample 'frac' from remainder.
             sampled_remainder = remainder.sample(frac=sampling_frac, random_state=42)
             df = pd.concat([mandatory, sampled_remainder])
         else:
             df = mandatory
             
         logger.info(f"Headlines after hybrid sampling: {len(df)}")

    # Case B: Fixed Cap (Original Logic)
    elif n_sample_per_day > 0:
        logger.info(f"Sampling up to {n_sample_per_day} headlines per company per day...")
        # Shuffle deterministically
        df = df.sample(frac=1, random_state=42)
        # Take top N per group
        df = df.groupby(['date', 'company']).head(n_sample_per_day)
        
        logger.info(f"Total headlines after sampling: {len(df)}")
    
    # Case C: Percentage Only (without coverage guarantee - deprecated by Hybrid, but logic remains if needed)
    elif sampling_frac and 0 < sampling_frac < 1.0:
        df = df.sample(frac=sampling_frac, random_state=42)
        
    # Run analysis
    analyzer = SentimentAnalyzer()
    scored_df = analyzer.analyze_headlines(df, text_col='text')
    
    # Aggregate
    # We want features per company per day
    agg_features = scored_df.groupby(['date', 'company']).agg(
        sentiment_mean=('sentiment_score', 'mean'),
        news_count=('sentiment_score', 'count'),
        sentiment_std=('sentiment_score', 'std') # Extra feature
    ).reset_index()
    
    if output_path:
        logger.info(f"Saving aggregated sentiment features to {output_path}...")
        agg_features.to_csv(output_path, index=False)
        
    return agg_features
