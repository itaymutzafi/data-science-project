import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List
import yfinance as yf
from pathlib import Path

from src.config import *
from src.utils.feature_names import canonicalize_feature_columns


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
    df['Days_To_Nearest_Report'] = abs((nearest['filing_date'] - nearest.index)).dt.days
    df['Days_To_Nearest_Report'] = df['Days_To_Nearest_Report'].fillna(np.inf)
    df = canonicalize_feature_columns(df)

    return df

def _get_reports_cache_path(ticker: str) -> Path:
    cache_dir = PROJECT_ROOT / "data" / "raw" / "sec_filings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{ticker}_sec_filings.parquet"

def _load_reports_cache(ticker: str) -> List[dict]:
    cache_path = _get_reports_cache_path(ticker)
    if not cache_path.exists():
        return []
    try:
        df = pd.read_parquet(cache_path)
        if df.empty:
            return []
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df.to_dict("records")
    except Exception as exc:
        print(f"Warning: failed to load filings cache for {ticker}: {exc}")
        return []

def _save_reports_cache(ticker: str, filings: List[dict]) -> None:
    cache_path = _get_reports_cache_path(ticker)
    df = pd.DataFrame(filings)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(cache_path, index=False)

def create_reports_dic(force_refresh: bool = False) -> Dict:
    reports_by_company = {}

    for company_name in TICKERS:
        if not force_refresh:
            cached = _load_reports_cache(company_name)
            if cached:
                reports_by_company[company_name] = cached
                continue

        ticker = yf.Ticker(company_name)
        filings_fn = getattr(ticker, "get_sec_filings", None)
        filings = []
        if callable(filings_fn):
            try:
                filings = filings_fn()
            except Exception as exc:
                print(f"Warning: failed to fetch filings for {company_name}: {exc}")
                filings = _load_reports_cache(company_name)
        else:
            # Fallback: skip gracefully when API not available in current yfinance
            print(f"Warning: SEC filings API not available for {company_name}. Skipping.")
            filings = _load_reports_cache(company_name)
        
        if isinstance(filings, pd.DataFrame):
            filings = filings.to_dict("records")
        
        if filings:
            _save_reports_cache(company_name, filings)
        
        reports_by_company[company_name] = filings        

    return reports_by_company

def reports(dfs: Dict[str, pd.DataFrame], force_refresh: bool = False) -> None:
    reports_by_company = create_reports_dic(force_refresh=force_refresh)
    plot_sec_fiilings_dates(reports_by_company)    
    
    for company, df in dfs.items():
        if company not in TICKERS:
            company = COMPANY_TO_TICKERS_MAP[company]
        reports_list = reports_by_company[company]
        reports_df = pd.DataFrame(reports_list)
        if "date" not in reports_df.columns or reports_df.empty:
            print(f"No filing dates available for {company}. Filling 'Days_To_Nearest_Report' with NaNs.")
            df['Days_To_Nearest_Report'] = np.nan
            continue
        report_dates = reports_df["date"]
        dfs[company] = create_days_to_report(df, report_dates)
