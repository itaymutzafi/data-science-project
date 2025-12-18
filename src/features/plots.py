import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List
import seaborn as sns

from sympy import true
from src.utils import statistic_tests as st
import yfinance as yf
from src.config import COMPANY_COLORS, DAYNAMES, MONTHNAMES, TICKERS, TICKER_TO_COMPANY_MAP

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
            df['Vol20'] = df['Return'].rolling(20).std()
        
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
            df['MA20'] = df['Close'].rolling(20).mean()
            df['MA50'] = df['Close'].rolling(50).mean()
        
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
    for i, ticker in enumerate(TICKERS):
        filings_list = reports_by_company[ticker]
        filings_df = pd.DataFrame(filings_list)

        filings_df['date'] = pd.to_datetime(filings_df['date'])

        company_name = TICKER_TO_COMPANY_MAP.get(ticker, ticker)
        color = COMPANY_COLORS.get(company_name, COMPANY_COLORS.get(ticker, None))

        ax.scatter(filings_df['date'], [i] * len(filings_df), 
                label=company_name, color=color, alpha=0.7, s=20)

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
    df['Days To Nearest Report'] = (nearest['filing_date'] - nearest.index).dt.days
    df['Days To Nearest Report'] = df['Days To Nearest Report'].fillna(np.inf)

    return df

def create_reports_dic() -> Dict:
    reports_by_company = {}

    for company_name in TICKERS:
        ticker = yf.Ticker(company_name)
        reports_by_company[company_name] = ticker.get_sec_filings()

    return reports_by_company

def reports(dfs: Dict[str, pd.DataFrame]) -> None:
    reports_by_company = create_reports_dic()
    plot_sec_fiilings_dates(reports_by_company)
    
    for company, df in dfs.items():
        reports_list = reports_by_company[company]
        reports_df = pd.DataFrame(reports_list)
        report_dates = reports_df["date"]
        dfs[company] = create_days_to_report(df, report_dates)
    
def plot_corrletion_companies(df_s):
    cols = ['Close', 'Volume', 'MSFT - Close', 'MSFT - Volume', 'GOOG - Close', 'GOOG - Volume', 'AMZN - Close', 'AMZN - Volume']
    corr_df = df_s["AAPL"][cols].corr()
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Between APPLE and other stocks")
    plt.show()
    

    
    
    
    



