import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

from src.config import *

def avg_attr_by_time_plot(dfs: Dict[str, pd.DataFrame], column: str, time_precision: str) -> None:
    if all(column in df.columns for df in dfs.values()):
        plt.figure(figsize=(12, 6))
        width = 0.2
        multiplier = 0

        if time_precision == 'Day':
            time_names = DAYNAMES
            time_range = time_names
        else:
            time_names = MONTHNAMES
            time_range = range(1, 13)
        x = np.arange(len(time_names))
        
        for name, df in dfs.items():
            y = df.groupby(time_precision)[column].mean()
            y = y.reindex(time_range)
            offset = width * multiplier
            plt.bar(x + offset, y.values, width, label=name, color=COMPANY_COLORS[name], alpha=0.8)
            multiplier += 1
        
        plt.xlabel(time_precision)
        plt.ylabel(f'Avg {column}')
        plt.title(f'Average {column} by {time_precision}')
        plt.xticks(x + width * (len(TICKERS) - 1) / 2, time_names, rotation=45)
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
