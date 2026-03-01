"""Volatility feature engineering and diagnostics."""

import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
import seaborn as sns

from src.config import COMPANY_COLORS, VOLATILITY_WINDOWS, OUTLIER_THRESHOLD
from src.utils import statistic_tests as st


def add_volatility_features(dfs: Dict[str, pd.DataFrame], windows: List[int] = VOLATILITY_WINDOWS) -> List[str]:
    """Add rolling volatility features and cap extreme outliers."""
    added = []
    col_names = [f"Vol{win}" for win in windows]

    for name, df in dfs.items():
        if "Return" in df.columns:
            for idx, win in enumerate(windows):
                col_name = col_names[idx]

                vol_series = df["Return"].rolling(window=win).std()
                mean_vol = vol_series.mean()
                std_vol = vol_series.std()

                if std_vol > 0:
                    upper_limit = mean_vol + (OUTLIER_THRESHOLD * std_vol)
                    vol_series = vol_series.clip(upper=upper_limit)

                df[col_name] = vol_series

            added.append(name)

    print(f"Added Volatility features: {windows} for {len(added)} stocks.")
    print(f"Note: Outliers capped at {OUTLIER_THRESHOLD} sigma.")

    return col_names


def volatility_comparison_plot(dfs: Dict[str, pd.DataFrame], window_sizes: List[int] = VOLATILITY_WINDOWS) -> None:
    """Plot rolling volatility comparison across companies."""
    for win in window_sizes:
        vol_col = f"Vol{win}"

        plt.figure(figsize=(14, 6))

        has_data = False
        for name, df in dfs.items():
            if vol_col in df.columns:
                has_data = True
                plt.plot(
                    df.index,
                    df[vol_col],
                    label=f"{name}",
                    color=COMPANY_COLORS.get(name, "grey"),
                    alpha=0.8,
                    linewidth=1.5,
                )

        if not has_data:
            plt.close()
            continue

        plt.title(f"Market Volatility ({win}-day Rolling Std Dev)", fontsize=14, fontweight="bold")
        plt.ylabel("Volatility (Std Dev of Returns)", fontsize=12)
        plt.xlabel("Date", fontsize=12)
        plt.legend(frameon=True, shadow=True)
        plt.grid(True, alpha=0.2, linestyle="--")
        sns.despine()
        plt.tight_layout()
        plt.show()


def test_volatility_seasonality(dfs: Dict[str, pd.DataFrame], vol_col: str = "Vol20") -> None:
    """Seasonality significance test for volatility, same approach as return seasonality."""
    day_results = {}
    month_results = {}

    for name, df in dfs.items():
        if vol_col in df.columns:
            day_results[name] = st.test_seasonality(df, vol_col, "Day")
            month_results[name] = st.test_seasonality(df, vol_col, "Month")

    if day_results:
        st.plot_all_tickers_seasonality(day_results, "Day", COMPANY_COLORS, feature_name="Volatility")
    if month_results:
        st.plot_all_tickers_seasonality(month_results, "Month", COMPANY_COLORS, feature_name="Volatility")
    if day_results or month_results:
        st.seasonality_summary_table(day_results, month_results)
