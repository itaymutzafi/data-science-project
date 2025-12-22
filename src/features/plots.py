import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List
import seaborn as sns

from src.utils import statistic_tests as st
import yfinance as yf
from src.config import *

def return_plot(dfs: Dict[str, pd.DataFrame]) -> None:
    # 3. Return Seasonality - Daily (All companies)
    if all('Return' in df.columns for df in dfs.values()):
        print("\n3. Return Seasonality - Daily (All Companies)")
        plt.figure(figsize=(12, 6))
        x = np.arange(len(DAYNAMES))
        width = 0.2
        multiplier = 0
        
        for name, df in dfs.items():
            daily_ret = df.groupby('Day')['Return'].mean()
            daily_ret = daily_ret.reindex(DAYNAMES)
            offset = width * multiplier
            plt.bar(x + offset, daily_ret.values, width, label=name, color=COMPANY_COLORS[name], alpha=0.8)
            multiplier += 1
        
        plt.xlabel('Day of Week')
        plt.ylabel('Avg Return')
        plt.title('Average Return by Day of Week - Comparison')
        plt.xticks(x + width * (len(TICKERS) - 1) / 2, DAYNAMES, rotation=45)
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Box plot comparison
        plt.figure(figsize=(14, 6))
        combined_data = []
        combined_labels = []
        for name, df in dfs.items():
            for day in DAYNAMES:
                day_data = df[df['Day'] == day]['Return'].dropna()
                combined_data.append(day_data)
                combined_labels.append(f"{name}\n{day}")
        
        # Create a more readable box plot
        plot_data = []
        plot_labels = []
        for day in DAYNAMES:
            for name in TICKERS:
                if name in dfs:
                    day_data = dfs[name][dfs[name]['Day'] == day]['Return'].dropna()
                    if len(day_data) > 0:
                        plot_data.append(day_data)
                        plot_labels.append(f"{name}")
        
        # Alternative: grouped box plot
        fig, ax = plt.subplots(figsize=(14, 6))
        positions = []
        data_to_plot = []
        labels_to_plot = []
        
        for i, day in enumerate(DAYNAMES):
            for j, name in enumerate(TICKERS):
                if name in dfs:
                    day_data = dfs[name][dfs[name]['Day'] == day]['Return'].dropna()
                    if len(day_data) > 0:
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
        ax.set_title('Return Distribution by Day of Week - Comparison')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=COMPANY_COLORS[name], alpha=0.7, label=name) for name in TICKERS]
        ax.legend(handles=legend_elements, loc='upper right')
        plt.tight_layout()
        plt.show()
    
    # 4. Return Seasonality - Monthly (All companies)
    if all('Return' in df.columns for df in dfs.values()):
        print("4. Return Seasonality - Monthly (All Companies)")
        plt.figure(figsize=(14, 6))
        x = np.arange(len(MONTHNAMES))
        width = 0.2
        multiplier = 0
        
        for name, df in dfs.items():
            monthly_ret = df.groupby('Month')['Return'].mean()
            monthly_ret = monthly_ret.reindex(range(1, 13))
            offset = width * multiplier
            plt.bar(x + offset, monthly_ret.values, width, label=name, color=COMPANY_COLORS[name], alpha=0.8)
            multiplier += 1
        
        plt.xlabel('Month')
        plt.ylabel('Avg Return')
        plt.title('Average Return by Month - Comparison')
        plt.xticks(x + width * (len(TICKERS) - 1) / 2, MONTHNAMES, rotation=45)
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # 9. Statistical tests for each company
    print("\n9. Statistical Seasonality Tests")
    for name, df in dfs.items():
        if 'Return' in df.columns:
            print(f"\n{name}:")
            st.display_seasonality_results(st.test_seasonality(df, 'Return', 'Day'))
            st.display_seasonality_results(st.test_seasonality(df, 'Return', 'Month'))

def volatility(dfs: Dict[str, pd.DataFrame]) -> None:
    # 7. Volatility Analysis (All companies)
    if all('Return' in df.columns for df in dfs.values()):
        print("7. Volatility Analysis (All Companies)")
        # Calculate volatility for each company
        for name, df in dfs.items():
            df['Vol20'] = df['Return'].rolling(20).std().bfill()
        
        plt.figure(figsize=(14, 6))
        for name, df in dfs.items():
            if 'Vol20' in df.columns:
                plt.plot(df.index, df['Vol20'], label=f"{name} (20-day)", color=COMPANY_COLORS[name], alpha=0.8, linewidth=1.5)
        plt.title("Rolling Volatility (20-day) - Comparison")
        plt.ylabel("Volatility (Std Dev)")
        plt.xlabel("Year")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Volatility by day
        print("Volatility by Day of Week (All Companies)")
        plt.figure(figsize=(12, 6))
        x = np.arange(len(DAYNAMES))
        width = 0.2
        multiplier = 0
        
        for name, df in dfs.items():
            if 'Vol20' in df.columns:
                daily_vol20 = df.groupby('Day')['Vol20'].mean()
                daily_vol20 = daily_vol20.reindex(DAYNAMES)
                offset = width * multiplier
                plt.bar(x + offset, daily_vol20.values, width, label=name, color=COMPANY_COLORS[name], alpha=0.8)
                multiplier += 1
        
        plt.xlabel('Day of Week')
        plt.ylabel('Avg Volatility (20-day)')
        plt.title('Average Volatility by Day of Week - Comparison')
        plt.xticks(x + width * (len(TICKERS) - 1) / 2, DAYNAMES, rotation=45)
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Volatility by month
        print("Volatility by Month (All Companies)")
        plt.figure(figsize=(14, 6))
        x = np.arange(len(MONTHNAMES))
        width = 0.2
        multiplier = 0
        
        for name, df in dfs.items():
            if 'Vol20' in df.columns:
                monthly_vol20 = df.groupby('Month')['Vol20'].mean()
                monthly_vol20 = monthly_vol20.reindex(range(1, 13))
                offset = width * multiplier
                plt.bar(x + offset, monthly_vol20.values, width, label=name, color=COMPANY_COLORS[name], alpha=0.8)
                multiplier += 1
        
        plt.xlabel('Month')
        plt.ylabel('Avg Volatility (20-day)')
        plt.title('Average Volatility by Month - Comparison')
        plt.xticks(x + width * (len(TICKERS) - 1) / 2, MONTHNAMES, rotation=45)
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

def moving_average(dfs: Dict[str, pd.DataFrame]) -> None:
    if all('Close' in df.columns for df in dfs.values()):
        for name, df in dfs.items():
            # for the first 20 / 50 days - we fill the values with the value of the first valid day
            df['MA20'] = df['Close'].rolling(20).mean().bfill()
            df['MA50'] = df['Close'].rolling(50).mean().bfill()
        
        plt.figure(figsize=(14, 6))
        for name, df in dfs.items():
            plt.plot(df.index, df['Close'], label=f"{name} Close", color=COMPANY_COLORS[name], alpha=0.5, linewidth=1)
            if 'MA20' in df.columns:
                plt.plot(df.index, df['MA20'], label=f"{name} MA20", color=COMPANY_COLORS[name], linestyle='--', alpha=0.7, linewidth=1)
        plt.title("Close Price and 20-day Moving Average - Comparison")
        plt.ylabel("Price")
        plt.xlabel("Year")
        plt.legend(ncol=2, fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def plot_sec_fiilings_dates(reports_by_company):
    fig, ax = plt.subplots(figsize=(12, 6))
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
        if "date" not in reports_df.columns:
            print(f"No filing dates available for {company}. Skipping date merge.")
            continue
        report_dates = reports_df["date"]
        dfs[company] = create_days_to_report(df, report_dates)
    
def plot_corrletion_companies(df_s):
    """
    Correlation heatmap across companies' Close/Volume.
    Works directly on df_s (dict of dataframes) without assuming Apple-specific columns.
    """
    if not df_s:
        print("No data provided.")
        return

    # Build a combined dataframe with standardized column names per ticker
    combined = pd.DataFrame()
    for ticker, df in df_s.items():
        temp = df.copy()
        if not isinstance(temp.index, pd.DatetimeIndex):
            temp.index = pd.to_datetime(temp.index)
        if temp.index.tz is not None:
            temp.index = temp.index.tz_localize(None)

        cols = {}
        if "Close" in temp.columns:
            cols[f"{ticker} - Close"] = temp["Close"]
        if "Volume" in temp.columns:
            cols[f"{ticker} - Volume"] = temp["Volume"]
        if not cols:
            continue

        temp_df = pd.DataFrame(cols)
        combined = temp_df if combined.empty else combined.join(temp_df, how="outer")

    if combined.empty:
        print("No overlapping Close/Volume data to correlate.")
        return

    corr_df = combined.corr().round(2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Between Stocks (Close & Volume)")
    plt.tight_layout()
    plt.show()

def add_auxiliary_features(dfs: Dict[str, pd.DataFrame], aux_data: pd.DataFrame) -> None:
    """
    Merges selected auxiliary features (e.g., Nasdaq, VIX) into each company's DataFrame 
    and plots them for visual inspection.
    """
    if aux_data.empty:
        print("Auxiliary data is empty. Skipping feature integration.")
        return

    # Ensure aux_data index is tz-naive for merging
    if aux_data.index.tz is not None:
        aux_data = aux_data.copy()
        aux_data.index = aux_data.index.tz_localize(None)

    features_to_add = aux_data.columns.tolist()
    print(f"Integrating Auxiliary Features: {features_to_add}")

    for name, df in dfs.items():
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Merge left on index
        # We use join to keep the original index
        # Check if columns already exist to avoid duplication/suffixes
        new_cols = [c for c in features_to_add if c not in df.columns]
        
        if new_cols:
            dfs[name] = df.join(aux_data[new_cols], how='left')
            # Forward fill to handle missing daily data if aux matches higher timeframe or gaps
            dfs[name][new_cols] = dfs[name][new_cols].ffill()
        
            # Forward fill to handle missing daily data if aux matches higher timeframe or gaps
            dfs[name][new_cols] = dfs[name][new_cols].ffill()
        
    # --- Visualization 1: Combined Trend Plot (All Stocks + Aux Features) ---
    print(f"\nVisualizing Combined Context for {len(dfs)} companies...")
    plt.figure(figsize=(16, 8))
    
    # 1. Plot Aux Features (Background context)
    for col in features_to_add:
        # We can take the data from aux_data directly for the full range, 
        # or from the first df to ensure alignment. Using aux_data with alignment:
        series = aux_data[col].dropna()
        # Align to the date range of the stocks roughly
        common_start = min(df.index.min() for df in dfs.values())
        common_end = max(df.index.max() for df in dfs.values())
        
        # Ensure timezone naive comparisons
        if common_start.tz is not None: common_start = common_start.tz_localize(None)
        if common_end.tz is not None: common_end = common_end.tz_localize(None)
        
        mask = (series.index >= common_start) & (series.index <= common_end)
        series = series.loc[mask]
        
        if not series.empty:
            norm = (series - series.mean()) / series.std()
            color = AUX_COLORS.get(col, 'grey')
            plt.plot(series.index, norm, label=f"{col} (Macro)", 
                     color=color, linestyle='--', alpha=0.6, linewidth=1.5)

    # 2. Plot Stocks (Foreground focus)
    for ticker, df in dfs.items():
        if 'Close' not in df.columns:
            continue
        series = df['Close'].dropna()
        if not series.empty:
            norm = (series - series.mean()) / series.std()
            color = COMPANY_COLORS.get(ticker, COMPANY_COLORS.get(TICKER_TO_COMPANY_MAP.get(ticker), 'black'))
            plt.plot(series.index, norm, label=f"{ticker}", 
                     color=color, linewidth=2.5, alpha=0.9)

    plt.title("Market Context: Stocks vs. Macro Indicators (Normalized Z-Score)", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Z-Score (Std Dev from Mean)")
    plt.legend(ncol=2) # Two columns for legend to save space
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Visualization 2: Combined Correlation Matrix (Grouped Bar) ---
    corr_data = {}
    
    for ticker, df in dfs.items():
        if 'Return' not in df.columns:
            continue
            
        valid_cols = [c for c in features_to_add if c in df.columns]
        if not valid_cols:
            continue
            
        # correlation between Stock Return and Aux features
        corrs = df[['Return'] + valid_cols].corr()['Return'].drop('Return')
        corr_data[ticker] = corrs
    
    if corr_data:
        corr_df = pd.DataFrame(corr_data) # Rows = Aux features, Cols = Tickers
        
        # Transpose for easier grouping by ticker if preferred, 
        # OR keep as is to compare how 'VIX' affects AAPL vs AMZN.
        # Let's group by Stock (Ticker) on X-axis, and bars for each Aux feature.
        
        # We want: X-axis = Tickers, Bars = Aux Features
        ax = corr_df.T.plot(kind='bar', figsize=(14, 6), width=0.8, 
                            color=[AUX_COLORS.get(col, 'grey') for col in corr_df.index])
        
        plt.title("Correlation: Daily Returns vs. Macro Indicators", fontsize=14)
        plt.ylabel("Pearson Correlation")
        plt.xlabel("Company")
        plt.axhline(0, color='black', linewidth=0.8)
        plt.grid(True, axis='y', alpha=0.3)
        plt.legend(title="Macro Indicator")
        plt.tight_layout()
        plt.show()
    

    
    
    
    
