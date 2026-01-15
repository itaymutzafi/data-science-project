import pandas as pd
import numpy as np
from typing import Dict


def add_day_month_features(dfs: Dict[str, pd.DataFrame]) -> None:
    added = []

    for name, df in dfs.items():
        df['Day'] = df.index.day_name()

        df['Month'] = df.index.month

        added.append(name)
    
    print(f"{added}: Added Day and Month features")

def preprocess_day_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode day-of-week cyclically using sine and cosine.
    Requires a 'Day' column with values like 'Monday', 'Tuesday', etc.
    """

    if "Day" not in df.columns:
        return df

    df = df.copy()
    day_num = pd.to_datetime(df["Day"], format="%A").dt.dayofweek

    df["Day_sin"] = np.sin(2 * np.pi * day_num / 7)
    df["Day_cos"] = np.cos(2 * np.pi * day_num / 7)

    df = df.drop(columns=["Day"])

    return df
