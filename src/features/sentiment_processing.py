"""
Module for processing large news datasets for sentiment analysis.
Refactored from standalone script to comply with project standards.
"""
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from src.features.sentiment_analysis import SentimentAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_news_file(
        input_path: str, 
        output_path: str, 
        batch_size: int = 32, 
        chunk_size: int = 10000, 
        start_date: str = None, 
        sample_frac: float = 1.0
    ):
    """
    Process a large CSV file in chunks and save the aggregated results.
    
    Args:
        input_path (str): Path to raw news CSV.
        output_path (str): Path to save processed parquet file.
        batch_size (int): Batch size for model inference.
        chunk_size (int): Number of rows to read from CSV at a time.
        start_date (str): Optional. Filter data from this date (YYYY-MM-DD).
        sample_frac (float): Optional. Fraction of data to sample (0.0 - 1.0).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    start_date_ts = pd.to_datetime(start_date) if start_date else None
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Initializing Sentiment Analyzer...")
    analyzer = SentimentAnalyzer() 
    
    logger.info(f"Processing {input_path} in chunks of {chunk_size}...")
    if start_date:
        logger.info(f"Filtering data from: {start_date}")
    if sample_frac < 1.0:
        logger.info(f"Random sampling: {sample_frac*100}% of data")
    
    daily_aggs = []
    
    # Read CSV in chunks
    try:
        reader = pd.read_csv(input_path, chunksize=chunk_size, on_bad_lines='skip', parse_dates=['date'])
    except TypeError:
         # Fallback for older pandas
         reader = pd.read_csv(input_path, chunksize=chunk_size, error_bad_lines=False, parse_dates=['date'])

    for i, chunk in enumerate(tqdm(reader, desc="Processing Chunks")):
        if chunk.empty:
            continue
            
        # Ensure date parsing worked
        if not pd.api.types.is_datetime64_any_dtype(chunk['date']):
            chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
            chunk = chunk.dropna(subset=['date'])

        # 1. Date Filtering
        if start_date_ts:
            chunk = chunk[chunk['date'] >= start_date_ts]
            
        if chunk.empty:
            continue

        # 2. Random Sampling
        if sample_frac < 1.0:
            chunk = chunk.sample(frac=sample_frac)
            
        if chunk.empty:
            continue

        # Analyze sentiment
        # Check text columns dynamically
        text_cols = [c for c in chunk.columns if 'text' in c.lower() or 'headline' in c.lower()]
        target_text_col = 'text' if 'text' in chunk.columns else (text_cols[0] if text_cols else 'text')
        
        scored_chunk = analyzer.analyze_headlines(chunk, text_col=target_text_col, batch_size=batch_size)
        
        # Aggregate immediately to save memory
        agg_chunk = analyzer.aggregate_daily_sentiment(scored_chunk, date_col='date')
        daily_aggs.append(agg_chunk)
        
    logger.info("Combining partial aggregations...")
    if not daily_aggs:
        logger.warning("No data processed (check date filter or input file).")
        return

    # Combine all daily chunks
    full_df = pd.concat(daily_aggs)
    
    # Reset index to make 'date' a column for grouping
    full_df = full_df.reset_index()
    
    # Calculate weighted sums for the mean
    full_df['weighted_sum'] = full_df['sentiment_mean'] * full_df['news_count']
    
    final_agg = full_df.groupby('date').agg(
        total_score=('weighted_sum', 'sum'),
        total_count=('news_count', 'sum')
    )
    
    final_agg['sentiment_mean'] = final_agg['total_score'] / final_agg['total_count']
    final_agg = final_agg.rename(columns={'total_count': 'news_count'})
    final_agg = final_agg[['sentiment_mean', 'news_count']]
    
    logger.info(f"Saving processed data to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_agg.to_parquet(output_path)
    logger.info("Done.")
