from pathlib import Path
from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import TICKERS

def add_peer_stock_features(dfs: Dict[str, pd.DataFrame], columns_to_merge: List[str]) -> None:
    added = []

    for ticker, df in dfs.items():
        other_tickers = {key: other_df for key, other_df in dfs.items() if key != ticker}
        dfs[ticker] = merge_df_by_date(ticker, df, other_tickers, columns_to_merge)
        added.append(ticker)

    print(f"{added}: Added Peer Stock feature")

def save_merged_df_to_file(df: pd.DataFrame, ticker_name: str) -> None:
    project_root = Path(__file__).resolve().parents[2]
    cache_dir = project_root / "data" / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"merged_{ticker_name}"
    cache_path = cache_dir / output_filename
    
    df.to_parquet(cache_path)
    print(f"Merged data saved to: {cache_path}")

def merge_df_by_date(ticker_name: str, main_df: pd.DataFrame, other_companies: Dict[str, pd.DataFrame], columns: List[str]) -> pd.DataFrame:
    res_df = main_df.copy()

    for feature_name, feature_df in other_companies.items():
        available_cols = [col for col in columns if col in feature_df.columns]
        if not available_cols:
            print(f"No matching columns {columns} found in {feature_name}")
            continue
        
        feature_df_selected = feature_df[available_cols].copy()
        feature_df_selected.columns = [f"{feature_name} - {col}" for col in feature_df_selected.columns]

        new_cols = [c for c in feature_df_selected.columns if c not in res_df.columns]
        if not new_cols:
            print(f"Columns of {feature_name} already exist in {ticker_name}")
            continue
        
        feature_df_selected = feature_df_selected[new_cols]

        res_df = res_df.join(feature_df_selected, how="left")
        print(f"Merged {feature_name} - Added {len(feature_df_selected.columns)} columns: {list(feature_df_selected.columns)}")
       
    save_merged_df_to_file(res_df, ticker_name)
    return res_df

def peer_stock_correlation(dfs: Dict[str, pd.DataFrame], ticker: str, column: str) -> None:
    other_tickers = [t for t in TICKERS if t != ticker]
    others_cols = [f"{t} - {column}" for t in other_tickers]
    cols = [column] + others_cols
    corr_df = dfs[ticker][cols].corr()

    sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Correlation Between {ticker} and peer stock - {column}")
    plt.show()

# def plot_corrletion_companies(df_s):
#     """
#     Correlation heatmap across companies' Close/Volume.
#     Works directly on df_s (dict of dataframes) without assuming Apple-specific columns.
#     """
#     if not df_s:
#         print("No data provided.")
#         return

#     # Build a combined dataframe with standardized column names per ticker
#     combined = pd.DataFrame()
#     for ticker, df in df_s.items():
#         temp = df.copy()
#         if not isinstance(temp.index, pd.DatetimeIndex):
#             temp.index = pd.to_datetime(temp.index)
#         if temp.index.tz is not None:
#             temp.index = temp.index.tz_localize(None)

#         cols = {}
#         if "Close" in temp.columns:
#             cols[f"{ticker} - Close"] = temp["Close"]
#         if "Volume" in temp.columns:
#             cols[f"{ticker} - Volume"] = temp["Volume"]
#         if not cols:
#             continue

#         temp_df = pd.DataFrame(cols)
#         combined = temp_df if combined.empty else combined.join(temp_df, how="outer")

#     if combined.empty:
#         print("No overlapping Close/Volume data to correlate.")
#         return

#     corr_df = combined.corr().round(2)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
#     plt.title("Correlation Between Stocks (Close & Volume)")
#     plt.tight_layout()
#     plt.show()
