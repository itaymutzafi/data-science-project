"""
Leakage detection and data integrity suite.

This module provides utilities to audit datasets and model pipelines for common
data leakage patterns in financial time-series.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Any
import logging

logger = logging.getLogger(__name__)

def check_timestamp_order(df: pd.DataFrame, date_col: str = 'date') -> bool:
    """
    Verifies that data is strictly chronological per ticker/group.
    """
    if date_col not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("No date column or index found for timestamp check.")
        return False
        
    # Standardize to column
    data = df.copy()
    if date_col not in data.columns:
        data[date_col] = data.index
        
    # Sort just in case, though we expect it to be sorted
    # Check if sorted
    for company, group in data.groupby('company') if 'company' in data.columns else [('All', data)]:
        if not group[date_col].is_monotonic_increasing:
            logger.error(f"Data for {company} is not monotonically increasing in time.")
            return False
            
    return True

def check_lookahead_correlation(
    df: pd.DataFrame, 
    target_col: str, 
    feature_cols: List[str], 
    threshold: float = 0.95
) -> List[str]:
    """
    Checks if any feature is suspiciously correlated with the target.
    A correlation of 1.0 (or close) often indicates the target (or a proxy) 
    was accidentally included in features.
    
    Args:
        df: DataFrame with features and target.
        target_col: Name of target column.
        feature_cols: List of features to check.
        threshold: Correlation threshold to flag.
        
    Returns:
        List of suspicious feature names.
    """
    if target_col not in df.columns:
        return []
        
    suspicious = []
    correlations = df[feature_cols].corrwith(df[target_col])
    
    for feat, corr in correlations.items():
        if abs(corr) >= threshold:
            logger.warning(f"Potential Leakage: Feature '{feat}' has correlation {corr:.4f} with target '{target_col}'.")
            suspicious.append(feat)
            
    return suspicious

def check_overlap_leakage(
    train_idx: np.ndarray, 
    val_idx: np.ndarray, 
    df: pd.DataFrame
) -> bool:
    """
    Verifies that training data strictly precedes validation data (no overlap).
    Assumes df matches the indices.
    """
    train_dates = df.iloc[train_idx].index if isinstance(df.index, pd.DatetimeIndex) else df.iloc[train_idx]['date']
    val_dates = df.iloc[val_idx].index if isinstance(df.index, pd.DatetimeIndex) else df.iloc[val_idx]['date']
    
    max_train = train_dates.max()
    min_val = val_dates.min()
    
    if max_train >= min_val:
        logger.error(f"Time Travel Leakage: Max Train Date ({max_train}) >= Min Validation Date ({min_val})")
        return True # Leakage found
        
    return False # No leakage

def run_full_leakage_audit(
    df: pd.DataFrame, 
    target_col: str, 
    feature_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Runs all static checks on a dataframe.
    """
    results = {}
    
    # 1. Timestamp Order
    results['monotonic_timestamps'] = check_timestamp_order(df)
    
    # 2. Null Checks
    null_target_rows = df[target_col].isnull().sum()
    results['null_targets'] = null_target_rows
    
    # 3. Correlation Leakage
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col and df[c].dtype in [np.float64, np.float32, np.int64]]
        
    suspicious_feats = check_lookahead_correlation(df, target_col, feature_cols)
    results['suspicious_features'] = suspicious_feats
    
    return results
