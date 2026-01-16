import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

from src.config import DAYNAMES, MONTHNAMES, COMPANY_COLORS, TICKERS
from src.utils import set_style

# --- Generic helpers ---
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


# --- News ---
def date_groupby_line_plot(df: pd.DataFrame, yname: str, title: str) -> None:
    set_style()
    per_day = df.groupby(df["date"].dt.date).size()
    per_day.plot(kind="line")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(yname)
    plt.grid(True)
    plt.show()


def articles_over_time_by_dataset_plot(df: pd.DataFrame, is_log: bool, specific_years=None) -> None:
    set_style()
    grouped = (
        df.set_index("date")
        .groupby("dataset")
        .resample("ME", include_groups=False)
        .size()
        .unstack(level=0)
    )

    plt.figure()
    for column in grouped.columns:
        plt.plot(grouped.index, grouped[column], label=column)

    plt.xlabel("Time")
    plt.ylabel("Log Number of Articles" if is_log else "Number of Articles")
    if is_log:
        plt.yscale("log")

    title = "Articles Over Time by Dataset"
    if specific_years is not None:
        title += f" {specific_years}"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def article_volume_per_company_plot(df: pd.DataFrame) -> None:
    set_style()
    for company, group in df.groupby("company"):
        monthly_counts = group.set_index("date").resample("ME").size()
        plt.plot(monthly_counts.index, monthly_counts.values, label=company)

    plt.xlabel("Time")
    plt.ylabel("Number of Articles")
    plt.title("Monthly News Volume per Company")
    plt.legend(title="Company")
    plt.tight_layout()
    plt.show()


def pie_plot(counts: pd.Series, subject: str) -> None:
    set_style()
    plt.figure()
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%")
    plt.title(f"Distribution of {subject}")
    plt.show()


def table_visualize(df: pd.DataFrame, groupby) -> pd.DataFrame:
    return df.groupby(groupby).size().unstack(fill_value=0)
