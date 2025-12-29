import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List
import yfinance as yf

from src.config import *

def plot_sec_fiilings_dates(reports_by_company):
    _, ax = plt.subplots(figsize=(12, 6))
    has_data = False
    for i, ticker in enumerate(TICKERS):
        filings_list = reports_by_company[ticker]
        filings_df = pd.DataFrame(filings_list)

        if filings_df.empty or 'date' not in filings_df.columns:
            print(f"No filing dates available for {ticker}. Skipping.")
            continue

        has_data = True
        filings_df['date'] = pd.to_datetime(filings_df['date'])

        company_name = TICKER_TO_COMPANY_MAP.get(ticker, ticker)
        color = COMPANY_COLORS.get(company_name, COMPANY_COLORS.get(ticker, None))

        ax.scatter(filings_df['date'], [i] * len(filings_df), 
                label=company_name, color=color, alpha=0.7, s=20)

    if not has_data:
        print("No filing dates available for any ticker. Skipping filings plot.")
        return

    ax.set_yticks(range(len(TICKERS)))
    ax.set_yticklabels([TICKER_TO_COMPANY_MAP[t] for t in TICKERS])
    ax.set_xlabel("Filing Date")
    ax.set_title("SEC Filing Dates per Company")

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.show()

def create_days_to_report(df:pd.DataFrame, report_dates:List) -> pd.DataFrame:
    df = df.copy()
    # Ensure index is datetime and sorted
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    report_dates = pd.to_datetime(report_dates)
    filings_df = pd.DataFrame({'filing_date': report_dates}).sort_values('filing_date')

    nearest = pd.merge_asof(
        df,
        filings_df,
        left_index=True,
        right_on='filing_date',
        direction='nearest',
    )
    nearest.index = df.index
    df['Days To Nearest Report'] = abs((nearest['filing_date'] - nearest.index)).dt.days
    df['Days To Nearest Report'] = df['Days To Nearest Report'].fillna(np.inf)

    return df

def create_reports_dic() -> Dict:
    reports_by_company = {}

    for company_name in TICKERS:
        ticker = yf.Ticker(company_name)
        filings_fn = getattr(ticker, "get_sec_filings", None)
        filings = []
        if callable(filings_fn):
            filings = filings_fn()
        else:
            # Fallback: skip gracefully when API not available in current yfinance
            print(f"Warning: SEC filings API not available for {company_name}. Skipping.")
        reports_by_company[company_name] = filings

    return reports_by_company

def reports(dfs: Dict[str, pd.DataFrame]) -> None:
    reports_by_company = create_reports_dic()
    plot_sec_fiilings_dates(reports_by_company)
    
    for company, df in dfs.items():
        if company not in TICKERS:
            company = COMPANY_TO_TICKERS_MAP[company]
        reports_list = reports_by_company[company]
        reports_df = pd.DataFrame(reports_list)
        if "date" not in reports_df.columns or reports_df.empty:
            print(f"No filing dates available for {company}. Filling 'Days To Nearest Report' with NaNs.")
            df['Days To Nearest Report'] = np.nan
            continue
        report_dates = reports_df["date"]
        dfs[company] = create_days_to_report(df, report_dates)
