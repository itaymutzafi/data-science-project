"""External market and peer-feature integration utilities."""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
import seaborn as sns

from src.config import TICKERS, TICKER_TO_COMPANY_MAP, COMPANY_COLORS, AUX_COLORS, FEATURE_WINDOWS
from src.utils.feature_names import canonicalize_feature_name, canonicalize_feature_columns


def add_macro_features(
    dfs: Dict[str, pd.DataFrame],
    aux_data: pd.DataFrame,
    *,
    verbose: bool = False,
    plot: bool = False,
) -> None:
    """Add auxiliary macro features to all ticker dataframes."""
    add_auxiliary_features(dfs, canonicalize_feature_columns(aux_data), verbose=verbose, plot=plot)


def add_peer_stock_features(dfs: Dict[str, pd.DataFrame], columns_to_merge: List[str]) -> None:
    """Merge selected peer-stock columns into each ticker dataframe."""
    added = []

    for ticker, df in dfs.items():
        other_tickers = {key: other_df for key, other_df in dfs.items() if key != ticker}
        dfs[ticker] = merge_df_by_date(ticker, df, other_tickers, columns_to_merge)
        added.append(ticker)

    print(f"{added}: Added Peer Stock feature")


def save_merged_df_to_file(df: pd.DataFrame, ticker_name: str) -> None:
    """Persist merged dataframe to the raw data cache directory."""
    project_root = Path(__file__).resolve().parents[2]
    cache_dir = project_root / "data" / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"merged_{ticker_name}.parquet"
    cache_path = cache_dir / output_filename
    
    df.to_parquet(cache_path)
    print(f"Merged data saved to: {cache_path}")


def merge_df_by_date(
    ticker_name: str,
    main_df: pd.DataFrame,
    other_companies: Dict[str, pd.DataFrame],
    columns: List[str],
) -> pd.DataFrame:
    """Join selected columns from peer companies by index."""
    res_df = main_df.copy()

    for feature_name, feature_df in other_companies.items():
        available_cols = [col for col in columns if col in feature_df.columns]
        if not available_cols:
            print(f"No matching columns {columns} found in {feature_name}")
            continue
        
        feature_df_selected = feature_df[available_cols].copy()
        feature_df_selected.columns = [
            canonicalize_feature_name(f"{feature_name} - {col}")
            for col in feature_df_selected.columns
        ]

        new_cols = [c for c in feature_df_selected.columns if c not in res_df.columns]
        if not new_cols:
            print(f"Columns of {feature_name} already exist in {ticker_name}")
            continue
        
        feature_df_selected = feature_df_selected[new_cols]

        res_df = res_df.join(feature_df_selected, how="left")
        print(f"Merged {feature_name} - Added {len(feature_df_selected.columns)} columns: {list(feature_df_selected.columns)}")

    return res_df


def peer_stock_correlation(dfs: Dict[str, pd.DataFrame], ticker: str, column: str) -> None:
    """Plot correlation between a ticker column and peer equivalents."""
    other_tickers = [t for t in TICKERS if t != ticker]
    others_cols = [canonicalize_feature_name(f"{t} - {column}") for t in other_tickers]
    cols = [column] + others_cols
    corr_df = dfs[ticker][cols].corr()

    sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Correlation Between {ticker} and peer stock - {column}")
    plt.show()


def add_auxiliary_features(
    dfs: Dict[str, pd.DataFrame],
    aux_data: pd.DataFrame,
    *,
    verbose: bool = False,
    plot: bool = False,
) -> None:
    """Merge auxiliary market features into each ticker dataframe."""
    if aux_data.empty:
        if verbose:
            print("Auxiliary data is empty. Skipping feature integration.")
        return

    if aux_data.index.tz is not None:
        aux_data = aux_data.copy()
        aux_data.index = aux_data.index.tz_localize(None)

    features_to_add = aux_data.columns.tolist()
    if verbose:
        print(f"Integrating Auxiliary Features: {features_to_add}")

    for name, df in dfs.items():
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        new_cols = [c for c in features_to_add if c not in df.columns]

        if new_cols:
            dfs[name] = df.join(aux_data[new_cols], how="left")
            dfs[name][new_cols] = dfs[name][new_cols].ffill()
            dfs[name] = canonicalize_feature_columns(dfs[name])

        df = dfs[name]
        if "VIX_Index" in df.columns:
            vix_win = FEATURE_WINDOWS[0]
            df[f"VIX_MA{vix_win}"] = df["VIX_Index"].rolling(window=vix_win).mean()
            df["VIX_Gap"] = df["VIX_Index"] - df[f"VIX_MA{vix_win}"]

    if plot:
        plt.figure(figsize=(16, 8))

        for col in features_to_add:
            series = aux_data[col].dropna()
            common_start = min(df.index.min() for df in dfs.values())
            common_end = max(df.index.max() for df in dfs.values())
            if common_start.tz is not None:
                common_start = common_start.tz_localize(None)
            if common_end.tz is not None:
                common_end = common_end.tz_localize(None)
            mask = (series.index >= common_start) & (series.index <= common_end)
            series = series.loc[mask]
            if not series.empty:
                norm = (series - series.mean()) / series.std()
                color = AUX_COLORS.get(col, "grey")
                plt.plot(series.index, norm, label=f"{col} (Macro)", color=color, linestyle="--", alpha=0.6, linewidth=1.5)

        for ticker, df in dfs.items():
            if "Close" not in df.columns:
                continue
            series = df["Close"].dropna()
            if not series.empty:
                norm = (series - series.mean()) / series.std()
                color = COMPANY_COLORS.get(ticker, COMPANY_COLORS.get(TICKER_TO_COMPANY_MAP.get(ticker), "black"))
                plt.plot(series.index, norm, label=f"{ticker}", color=color, linewidth=2.5, alpha=0.9)

        plt.title("Market Context: Stocks vs. Macro Indicators (Normalized Z-Score)", fontsize=14)
        plt.xlabel("Date")
        plt.ylabel("Z-Score (Std Dev from Mean)")
        plt.legend(ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        corr_data = {}
        for ticker, df in dfs.items():
            if "Return" not in df.columns:
                continue
            valid_cols = [c for c in features_to_add if c in df.columns]
            if not valid_cols:
                continue
            corrs = df[["Return"] + valid_cols].corr()["Return"].drop("Return")
            corr_data[ticker] = corrs

        if corr_data:
            corr_df = pd.DataFrame(corr_data)
            corr_df.T.plot(
                kind="bar",
                figsize=(14, 6),
                width=0.8,
                color=[AUX_COLORS.get(col, "grey") for col in corr_df.index],
            )
            plt.title("Correlation: Daily Returns vs. Macro Indicators", fontsize=14)
            plt.ylabel("Pearson Correlation")
            plt.xlabel("Company")
            plt.axhline(0, color="black", linewidth=0.8)
            plt.grid(True, axis="y", alpha=0.3)
            plt.legend(title="Macro Indicator")
            plt.tight_layout()
            plt.show()
