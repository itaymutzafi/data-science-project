"""
Target generation module.

Defines logic for creating continuous (Regression) and discrete (Classification) targets
from price data.
"""

import pandas as pd
import numpy as np
from typing import Tuple

def experiment_create_target_variable(
    df: pd.DataFrame, 
    price_col: str = "Close", 
    horizon: int = 1, 
    target_type: str = "continuous",
    threshold: float = 0.0
) -> Tuple[pd.DataFrame, str]:
    """
    Adds a target column to the dataframe.
    
    Args:
        df: Input DataFrame with price column.
        price_col: Column to calculate returns from.
        horizon: Prediction horizon (days).
        target_type: 'continuous' (log return), 'binary' (up/down), 'multiclass' (bear/neutral/bull).
        threshold: Threshold for multiclass labeling (e.g. 0.005 for 0.5% buffer).
        
    Returns:
        DataFrame with new target column, name of target column.
    """
    df = df.copy()
    
    # Base: Log Return over horizon, SHIFTED BACK by horizon
    # Target at time t is Return from t to t+h
    # So we compute Return(t+h) and align it to t.
    # Return(t+h) = ln(P_{t+h} / P_t)
    # in pandas: shift(-horizon) brings future value to current row
    
    future_price = df[price_col].shift(-horizon)
    current_price = df[price_col]
    
    # Raw forward log return
    raw_target = np.log(future_price / current_price)
    
    target_name = f"Target_{horizon}D"
    
    if target_type == "continuous":
        df[target_name] = raw_target
        
    elif target_type == "binary":
        # 1 if return > 0, else 0
        df[target_name] = (raw_target > 0).astype(int)
        # NaN where raw_target is NaN
        df.loc[raw_target.isna(), target_name] = np.nan
        
    elif target_type == "multiclass":
        # 0: Bearish, 1: Neutral, 2: Bullish
        conditions = [
            (raw_target < -threshold),
            (raw_target >= -threshold) & (raw_target <= threshold),
            (raw_target > threshold)
        ]
        choices = [0, 1, 2]
        # Use select from numpy or map
        # But we need to handle NaNs safely
        labels = np.select(conditions, choices, default=np.nan)
        df[target_name] = labels
        
    else:
        raise ValueError(f"Unknown target_type: {target_type}")
        
    return df, target_name
