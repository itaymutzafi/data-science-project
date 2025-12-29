
import logging
import os
import pandas as pd
import numpy as np
from typing import Dict, List
from src import config

logger = logging.getLogger(__name__)

def check_dividends(df_s: Dict[str, pd.DataFrame]):
    """Checks for dividend data in the provided stocks dictionary."""
    for ticker, df in df_s.items():
        logger.info(f"\n--- {ticker} ---")
        if 'Dividends' in df.columns:
            dividends = df['Dividends']
            non_zero = dividends[dividends > 0]
            logger.info(f"Total Rows: {len(df)}")
            logger.info(f"Dividends Column Exists: YES")
            logger.info(f"Non-zero Dividends count: {len(non_zero)}")
            if not non_zero.empty:
                logger.info("Sample Dividends:")
                logger.info(f"\n{non_zero.head()}")
        else:
            logger.info(f"Dividends Column Exists: NO")

def check_lookahead_correlation(df: pd.DataFrame, target_col: str, feature_cols: List[str], threshold: float = 0.95) -> List[str]:
    """
    Checks if any features are suspiciously highly correlated with the target,
    which might indicate leakage.
    """
    suspicious = []
    if target_col not in df.columns:
        return suspicious
        
    for feat in feature_cols:
        if feat not in df.columns:
            continue
        try:
            corr = df[feat].corr(df[target_col])
            if abs(corr) > threshold:
                suspicious.append(feat)
                logger.warning(f"Feature '{feat}' has correlation {corr:.4f} with target '{target_col}'. Possible Leakage!")
        except Exception:
            pass
            
    return suspicious

def audit_timestamps(df: pd.DataFrame, time_col: str = 'date'):
    """Checks if timestamps are monotonic increasing."""
    if time_col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            try:
                df[time_col] = pd.to_datetime(df[time_col])
            except Exception:
                logger.error(f"Could not convert {time_col} to datetime.")
                return False
        
        is_mono = df[time_col].is_monotonic_increasing
        if not is_mono:
            logger.error(f"{time_col} is NOT monotonic increasing!")
        return is_mono
    return True

def run_leakage_audit(sentiment_path=None, aux_path=None):
    """Runs a full leakage audit on available data sources."""
    logger.info("Starting Leakage Audit...")
    
    # 1. Audit Sentiment Data
    path = sentiment_path or config.SENTIMENT_CACHE
    if os.path.exists(path):
        logger.info(f"Auditing Sentiment Data: {path}")
        sent_df = pd.read_csv(path)
        if 'date' in sent_df.columns:
            audit_timestamps(sent_df, 'date')
        
        # Check basic stats
        if 'sentiment_mean' in sent_df.columns:
             logger.info(f"Sentiment Mean stats:\n{sent_df['sentiment_mean'].describe()}")
    else:
        logger.warning(f"Sentiment cache not found at {path}")

    # 2. Audit Auxiliary Data
    path = aux_path or config.AUX_DATA_PATH
    if os.path.exists(path):
        logger.info(f"Auditing Aux Data: {path}")
        try:
            aux_df = pd.read_parquet(path)
            if isinstance(aux_df.index, pd.DatetimeIndex):
                if not aux_df.index.is_monotonic_increasing:
                    logger.error("Aux Data index is not monotonic!")
                else:
                    logger.info("Aux Data index is monotonic.")
        except Exception as e:
            logger.error(f"Failed to load Aux data: {e}")

