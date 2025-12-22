import pandas as pd
from typing import Dict

def add_day_month_features(dfs: Dict[str, pd.DataFrame]) -> None:
    added = []

    for name, df in dfs.items():
        df['Day'] = df.index.day_name()
        df['Month'] = df.index.month
        added.append(name)
    
    print(f"{added}: Added Day and Month features")
