"""Calendar-based feature engineering utilities."""

import pandas as pd
import numpy as np
from typing import Dict


def add_day_month_features(dfs: Dict[str, pd.DataFrame]) -> None:
    """Add day-name and month features from dataframe indices."""
    added = []

    for name, df in dfs.items():
        df["Day"] = df.index.day_name()

        df["Month"] = df.index.month

        added.append(name)

    print(f"{added}: Added Day and Month features")


def preprocess_day_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the Day column as cyclical sine and cosine features."""
    if "Day" not in df.columns:
        return df

    df = df.copy()
    day_num = pd.to_datetime(df["Day"], format="%A").dt.dayofweek

    df["Day_sin"] = np.sin(2 * np.pi * day_num / 7)
    df["Day_cos"] = np.cos(2 * np.pi * day_num / 7)

    df = df.drop(columns=["Day"])

    return df


def preprocess_month_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the Month column as cyclical sine and cosine features."""
    if "Month" not in df.columns:
        return df

    df = df.copy()
    month_num = df["Month"].astype(int)

    df["Month_sin"] = np.sin(2 * np.pi * month_num / 12)
    df["Month_cos"] = np.cos(2 * np.pi * month_num / 12)

    df = df.drop(columns=["Month"])

    return df
