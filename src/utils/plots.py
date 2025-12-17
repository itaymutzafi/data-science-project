import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
import pandas as pd
import src.config as config

def date_groupby_line_plot(df, yname, title):
    per_day = df.groupby(df["date"].dt.date).size()

    per_day.plot(kind="line")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(yname)
    plt.grid(True)
    plt.show()

def articles_over_time_by_dataset_plot(df, is_log, specific_years):
    grouped = (
        df
        .set_index("date")
        .groupby("dataset")
        .resample("ME", include_groups=False)
        .size()
        .unstack(level=0)
    )

    plt.figure()

    for column in grouped.columns:
        plt.plot(grouped.index, grouped[column], label=column)

    plt.xlabel("Time")
    if is_log:
        plt.ylabel("Log Number of Articles")
        plt.yscale("log")
    else:
        plt.ylabel("Number of Articles")

    title = "Articles Over Time by Dataset"
    if specific_years is not None:
        title += f" {specific_years}"
    plt.title(title)
    
    plt.legend()
    plt.show()

def article_volume_per_company_plot(df):
    for company, group in df.groupby("company"):
        monthly_counts = (
            group.set_index("date")
                .resample("ME")
                .size()
        )
        
        plt.plot(
            monthly_counts.index, 
            monthly_counts.values, 
            label=company
        )

    plt.xlabel("Time")
    plt.ylabel("Number of Articles")
    plt.title("Monthly News Volume per Company")
    plt.legend(title="Company")
    plt.tight_layout()
    plt.show()

def pie_plot(counts, subject):
    plt.figure()
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%")
    plt.title(f"Distribution of {subject}")
    plt.show()

def table_visualize(df, groupby):
    return (
        df
        .groupby(groupby)
        .size()
        .unstack(fill_value=0)
    )

def plot_sec_fillings(reports_by_company):
    import pandas as pd
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, ticker in enumerate(config.TICKERS):
        filings_list = reports_by_company[ticker]          # your list of dicts
        filings_df = pd.DataFrame(filings_list)

        # Ensure datetime
        filings_df['date'] = pd.to_datetime(filings_df['date'])

        # Optional: use full company name + color from config
        company_name = config.TICKER_TO_COMPANY_MAP.get(ticker, ticker)
        color = config.COMPANY_COLORS.get(company_name, config.COMPANY_COLORS.get(ticker, None))

        # One horizontal row of dots per company
        ax.scatter(filings_df['date'], [i] * len(filings_df), 
                label=company_name, color=color, alpha=0.7, s=20)

    ax.set_yticks(range(len(config.TICKERS)))
    ax.set_yticklabels([config.TICKER_TO_COMPANY_MAP[t] for t in config.TICKERS])
    ax.set_xlabel("Filing Date")
    ax.set_title("SEC Filing Dates per Company")

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.show()