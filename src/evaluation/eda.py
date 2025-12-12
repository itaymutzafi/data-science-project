import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

def run_eda(
    df: pd.DataFrame,
    ticker_name: str,
    daynames: List[str],
    monthnames: List[str],
    features_to_analyze: List[str] = None
) -> None:
    """
    Generic EDA pipeline for any stock DataFrame.
    
    Args:
        df: DataFrame with DatetimeIndex and OHLCV columns
        ticker_name: Stock ticker (e.g., "AAPL", "MSFT")
        daynames: List of day names (e.g., list(day_name))
        monthnames: List of month names (e.g., list(month_name)[1:])
        features_to_analyze: List of column names to analyze (e.g., ['Return', 'Volume'])
    """
    if features_to_analyze is None:
        features_to_analyze = ['Return', 'Volume', 'Close']
    
    # Add temporal columns if not present
    if 'Day' not in df.columns:
        df['Day'] = df.index.day_name()
    if 'Month' not in df.columns:
        df['Month'] = df.index.month
    
    print(f"\n{'='*60}")
    print(f"EDA for {ticker_name}")
    print(f"{'='*60}\n")
    
    # 1. Price & Volume Trends
    print(f"1. Price Trends ({ticker_name})")
    cols = ['Close', 'Open']
    df[cols].plot(figsize=(12, 5))
    plt.title(f"{ticker_name} Open and Close Price Over Time")
    plt.ylabel("Price ($)")
    plt.xlabel("Year")
    plt.grid(True)
    plt.show()
    
    print(f"Volume Behavior ({ticker_name})")
    df['Volume'].plot(figsize=(12, 5))
    plt.title(f"{ticker_name}'s Volume Over Time")
    plt.ylabel("Volume")
    plt.xlabel("Year")
    plt.grid(True)
    plt.show()
    
    # 2. Dividends & Stock Splits (if present)
    if 'Dividends' in df.columns:
        print(f"\nDividends ({ticker_name})")
        df['Dividends'].plot(figsize=(12, 5))
        plt.title(f"{ticker_name}'s Dividends Over Time")
        plt.ylabel("Paid per share ($)")
        plt.xlabel("Year")
        plt.grid(True)
        plt.show()
    
    if 'Stock Splits' in df.columns:
        print(f"Stock Splits ({ticker_name})")
        splits = df[df['Stock Splits'] != 0][['Stock Splits']].copy()
        if len(splits) > 0:
            print(splits)
        else:
            print(f"  → No stock splits in period")
    
    # 3. Correlation heatmap
    print(f"\nCorrelation Matrix ({ticker_name})")
    corr_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    corr_cols = [c for c in corr_cols if c in df.columns]
    if len(corr_cols) > 1:
        corr = df[corr_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f"Correlation: {ticker_name} Price Features")
        plt.tight_layout()
        plt.show()
    
    # 4. Return seasonality (daily)
    if 'Return' in df.columns:
        print(f"\nReturn Seasonality - Daily ({ticker_name})")
        daily_ret = df.groupby('Day')['Return'].mean()
        daily_ret = daily_ret.reindex(daynames)
        daily_ret.plot(kind='bar', figsize=(10, 5))
        plt.title(f"{ticker_name} Avg Return by Day of Week")
        plt.ylabel("Avg Return")
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.show()
        
        # Box plot for returns
        sns.boxplot(x='Day', y='Return', data=df, order=daynames)
        plt.title(f"{ticker_name} Return Distribution by Day of Week")
        plt.xticks(rotation=45)
        plt.show()
    
    # 5. Return seasonality (monthly)
    print(f"Return Seasonality - Monthly ({ticker_name})")
    monthly_ret = df.groupby('Month')['Return'].mean()
    monthly_ret = monthly_ret.reindex(range(1, 13))
    ax = monthly_ret.plot(kind='bar', figsize=(10, 5))
    for i, v in enumerate(monthly_ret):
        ax.text(i, v, f"{v:.5f}", ha='center', va='bottom', fontsize=8)
    plt.title(f"{ticker_name} Avg Return by Month")
    plt.ylabel("Avg Return")
    plt.xlabel("Month")
    ax.set_xticklabels(monthnames, rotation=45)
    plt.grid(True, axis='y')
    plt.show()
    
    # 6. Volume seasonality
    print(f"Volume Seasonality ({ticker_name})")
    monthly_vol = df.groupby('Month')['Volume'].mean()
    monthly_vol = monthly_vol.reindex(range(1, 13))
    ax = monthly_vol.plot(kind='bar', figsize=(10, 5), color='steelblue')
    plt.title(f"{ticker_name} Avg Volume by Month")
    plt.ylabel("Avg Volume")
    plt.xlabel("Month")
    ax.set_xticklabels(monthnames, rotation=45)
    plt.grid(True, axis='y')
    plt.show()
    
    daily_vol = df.groupby('Day')['Volume'].mean()
    daily_vol = daily_vol.reindex(daynames)
    daily_vol.plot(kind='bar', figsize=(10, 5), color='steelblue')
    plt.title(f"{ticker_name} Avg Volume by Day of Week")
    plt.ylabel("Avg Volume")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.show()
    
    # 7. Volatility (rolling std)
    if 'Return' in df.columns:
        print(f"Volatility Analysis ({ticker_name})")
        df['Vol20'] = df['Return'].rolling(20).std()
        df['Vol252'] = df['Return'].rolling(252).std()
        
        df[['Vol20', 'Vol252']].plot(figsize=(12, 5), color=['b', 'r'])
        plt.title(f"{ticker_name} Rolling Volatility (20-day & 252-day)")
        plt.ylabel("Volatility (Std Dev)")
        plt.grid(True)
        plt.show()
        
        # Volatility by day
        daily_vol20 = df.groupby('Day')['Vol20'].mean()
        daily_vol20 = daily_vol20.reindex(daynames)
        daily_vol20.plot(kind='bar', figsize=(10, 5), color='coral')
        plt.title(f"{ticker_name} Avg Volatility by Day of Week")
        plt.ylabel("Avg Vol (20-day)")
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.show()
        
        # Volatility by month
        monthly_vol20 = df.groupby('Month')['Vol20'].mean()
        monthly_vol20 = monthly_vol20.reindex(range(1, 13))
        ax = monthly_vol20.plot(kind='bar', figsize=(10, 5), color='coral')
        plt.title(f"{ticker_name} Avg Volatility by Month")
        plt.ylabel("Avg Vol (20-day)")
        plt.xlabel("Month")
        ax.set_xticklabels(monthnames, rotation=45)
        plt.grid(True, axis='y')
        plt.show()
    
    # 8. Moving averages
    if 'Close' in df.columns:
        print(f"Moving Averages ({ticker_name})")
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        
        plt.figure(figsize=(12, 5))
        plt.plot(df['Close'], label='Close', alpha=0.7)
        plt.plot(df['MA20'], label='20-day MA', color='orange')
        plt.plot(df['MA50'], label='50-day MA', color='red')
        plt.title(f"{ticker_name} Moving Averages")
        plt.legend()
        plt.grid(True)
        plt.show()