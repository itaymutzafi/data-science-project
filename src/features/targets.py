"""Target generation utilities."""

import numpy as np
import pandas as pd
from typing import Tuple


def experiment_create_target_variable(
    df: pd.DataFrame,
    price_col: str = "Close",
    horizon: int = 1,
    target_type: str = "continuous",
    threshold: float = 0.0,
) -> Tuple[pd.DataFrame, str]:
    """Create a forward target column and return its name.

    The target is aligned to timestamp `t` by using future price `t+horizon`,
    so each row keeps only information available at prediction time.
    """
    df = df.copy()

    future_price = df[price_col].shift(-horizon)
    current_price = df[price_col]

    raw_target = np.log(future_price / current_price)

    target_name = f"Target_{horizon}D"

    if target_type == "continuous":
        df[target_name] = raw_target

    elif target_type == "binary":
        df[target_name] = (raw_target > 0).astype(int)
        df.loc[raw_target.isna(), target_name] = np.nan

    elif target_type == "multiclass":
        conditions = [
            (raw_target < -threshold),
            (raw_target >= -threshold) & (raw_target <= threshold),
            (raw_target > threshold),
        ]
        choices = [0, 1, 2]
        labels = np.select(conditions, choices, default=np.nan)
        df[target_name] = labels

    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    return df, target_name
