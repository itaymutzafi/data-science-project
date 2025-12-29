import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

from src.config import COMPANY_COLORS
from src.evaluation.plots import set_style

def eda_attr_comparative_plot(dfs: Dict[str, pd.DataFrame], attr: str, title: str, is_scatter: bool) -> None:
    set_style()
    for name, df in dfs.items():
        if attr in df.columns:
            if is_scatter:
                s = df[attr].copy()

                # Keep only actual events
                s = s.dropna()
                s = s[s > 0]
                if s.empty:
                    continue

                plt.scatter(s.index, s.values, label=f"{name} {attr}", color=COMPANY_COLORS[name], alpha=0.8, s=20)
            else:
                plt.plot(df.index, df[attr], label=f"{name} {attr}", color=COMPANY_COLORS[name], alpha=0.8, linewidth=1.5)
    plt.title(title)
    plt.ylabel(attr)
    plt.xlabel("Year")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def eda_correlation(ticker_name: str, df, corr_method: str='pearson') -> None:
    set_style()
    corr_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'] 
    corr_cols = [c for c in corr_cols if c in df.columns] 
    if len(corr_cols) > 1: 
        corr = df[corr_cols].corr(method=corr_method)
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f") 
        plt.title(f"{corr_method} Correlation: {ticker_name}")
        plt.tight_layout() 
        plt.show()

def stock_split_table(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    set_style()
    rows = []
    for ticker, df in dfs.items():
        if "Stock Splits" in df.columns:
            s = df["Stock Splits"].copy()

            s = s.dropna()
            s = s[s != 0]

            for dt, val in s.items():
                rows.append({"Ticker": ticker, "Date": dt, "Split Factor": val})

    if not rows:
        return pd.DataFrame(columns=["Ticker", "Date", "Split Factor"])

    out = pd.DataFrame(rows).sort_values(["Date", "Ticker"])
    return out.reset_index(drop=True)
