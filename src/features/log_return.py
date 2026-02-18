import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from matplotlib.patches import Patch

from src.utils import statistic_tests as st
from src.config import COMPANY_COLORS, DAYNAMES, TICKERS


def add_return_features(dfs: Dict[str, pd.DataFrame]) -> None:
    added = []

    for name, df in dfs.items():
        if "Close" in df.columns:
            df['Return'] = df['Close'].pct_change()

            df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

            added.append(name)

    print(f"{added}: Added Return and Log Return features")

def return_day_boxplot(dfs: Dict[str, pd.DataFrame]) -> None:  
    if all('Return' in df.columns for df in dfs.values()):    
        _, ax = plt.subplots(figsize=(14, 6))
        positions = []
        data_to_plot = []
        labels_to_plot = []
        
        for i, day in enumerate(DAYNAMES):
            for j, name in enumerate(TICKERS):
                if name in dfs:
                    day_data = dfs[name][dfs[name]['Day'] == day]['Return'].dropna()
                    if len(day_data) == 0:
                        continue

                    positions.append(i * (len(TICKERS) + 1) + j)
                    data_to_plot.append(day_data)
                    labels_to_plot.append(name)
        
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True)
        for patch, name in zip(bp['boxes'], labels_to_plot):
            patch.set_facecolor(COMPANY_COLORS[name])
            patch.set_alpha(0.7)
        
        ax.set_xticks([i * (len(TICKERS) + 1) + (len(TICKERS) - 1) / 2 for i in range(len(DAYNAMES))])
        ax.set_xticklabels(DAYNAMES, rotation=45)
        ax.set_ylabel('Return')
        ax.set_title('Return Distribution by Day of Week')
        ax.grid(True, axis='y', alpha=0.3)
        
        legend_elements = [Patch(facecolor=COMPANY_COLORS[name], alpha=0.7, label=name) for name in TICKERS]
        ax.legend(handles=legend_elements, loc='upper right')
        plt.tight_layout()
        plt.show()

def test_return_seasonality(dfs: Dict[str, pd.DataFrame]) -> None:
    day_results = {}
    month_results = {}

    for name, df in dfs.items():
        if 'Return' in df.columns:
            day_results[name] = st.test_seasonality(df, 'Return', 'Day')
            month_results[name] = st.test_seasonality(df, 'Return', 'Month')

    if day_results:
        st.plot_all_tickers_seasonality(day_results, "Day", COMPANY_COLORS)
    if month_results:
        st.plot_all_tickers_seasonality(month_results, "Month", COMPANY_COLORS)
    if day_results or month_results:
        st.seasonality_summary_table(day_results, month_results)
