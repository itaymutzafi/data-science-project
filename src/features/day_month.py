import pandas as pd
from typing import Dict

TIME_FEATURES = []

def add_day_month_features(dfs: Dict[str, pd.DataFrame]) -> None:
    added = []

    for name, df in dfs.items():
        df['Day'] = df.index.day_name()
        TIME_FEATURES.append("Day")

        df['Month'] = df.index.month
        TIME_FEATURES.append("Month")

        added.append(name)
    
    print(f"{added}: Added Day and Month features")
