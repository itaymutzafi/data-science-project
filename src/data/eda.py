import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from src.config import COMPANY_COLORS, DAYNAMES, MONTHNAMES, TICKERS

def eda_attr_comparative_plot(dfs: Dict[str, pd.DataFrame], attr: str, title: str) -> None:
    for name, df in dfs.items():
        if attr in df.columns:
            plt.plot(df.index, df[attr], label=f"{name} "+attr, color=COMPANY_COLORS[name], alpha=0.8, linewidth=1.5)
    plt.title(title)
    plt.ylabel(attr)
    plt.xlabel("Year")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def eda_volume_seasonality_plot(dfs: Dict[str, pd.DataFrame], time_presicion: str) -> None:
    if time_presicion == 'Day':
        time_names = DAYNAMES
        time_range = time_names
    else:
        time_names = MONTHNAMES
        time_range = range(1, 13)

    x = np.arange(len(time_names))
    width = 0.2
    multiplier = 0
    
    for name, df in dfs.items():
        vol = df.groupby(time_presicion)['Volume'].mean()
        vol = vol.reindex(time_range)
        offset = width * multiplier
        plt.bar(x + offset, vol.values, width, label=name, color=COMPANY_COLORS[name], alpha=0.8)
        multiplier += 1
    
    plt.xlabel(time_presicion)
    plt.ylabel('Avg Volume')
    plt.title(f'Average Volume by {time_presicion} - Comparison')
    plt.xticks(x + width * (len(TICKERS) - 1) / 2, time_names, rotation=45)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def eda_correlation(ticker_name: str, df, corr_method: str='pearson') -> None:
    corr_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'] 
    corr_cols = [c for c in corr_cols if c in df.columns] 
    if len(corr_cols) > 1: 
        corr = df[corr_cols].corr(method=corr_method)
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f") 
        plt.title(f"{corr_method} Correlation: {ticker_name}")
        plt.tight_layout() 
        plt.show()
