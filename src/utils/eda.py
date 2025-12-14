import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
from src.utils import statistic_tests as st

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
    
    st.display_seasonality_results(st.test_seasonality(df, 'Return', 'Day'))
    
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

    st.display_seasonality_results(st.test_seasonality(df, 'Return', 'Month'))
    
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


def run_eda_comparative(
    dfs: Dict[str, pd.DataFrame],
    daynames: List[str],
    monthnames: List[str],
    features_to_analyze: List[str] = None,
    color_map: Dict[str, str] = None
) -> None:
    """
    Comparative EDA pipeline for multiple stock DataFrames.
    Plots all companies together for easy comparison.
    
    Args:
        dfs: Dictionary mapping company names to DataFrames (e.g., {"Apple": apple_df, "Microsoft": msft_df})
        daynames: List of day names (e.g., list(day_name))
        monthnames: List of month names (e.g., list(month_name)[1:])
        features_to_analyze: List of column names to analyze (e.g., ['Return', 'Volume'])
        color_map: Optional dictionary mapping company names to colors (e.g., {"Apple": "green", "Amazon": "yellow"})
                   If not provided, uses default matplotlib color scheme
    """
    if features_to_analyze is None:
        features_to_analyze = ['Return', 'Volume', 'Close']
    
    # Add temporal columns if not present
    for name, df in dfs.items():
        if 'Day' not in df.columns:
            df['Day'] = df.index.day_name()
        if 'Month' not in df.columns:
            df['Month'] = df.index.month
    
    company_names = list(dfs.keys())
    
    # Use custom color_map if provided, otherwise use default
    if color_map is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(company_names)))
        color_map = {name: colors[i] for i, name in enumerate(company_names)}
    else:
        # Ensure all companies have colors, use default for missing ones
        default_colors = plt.cm.tab10(np.linspace(0, 1, len(company_names)))
        for i, name in enumerate(company_names):
            if name not in color_map:
                color_map[name] = default_colors[i]
    
    print(f"\n{'='*60}")
    print(f"Comparative EDA for {', '.join(company_names)}")
    print(f"{'='*60}\n")
    
    # 1. Price Trends - All companies together
    print("1. Price Trends (All Companies)")
    plt.figure(figsize=(14, 6))
    for name, df in dfs.items():
        if 'Close' in df.columns:
            plt.plot(df.index, df['Close'], label=f"{name} Close", color=color_map[name], alpha=0.8, linewidth=1.5)
    plt.title("Close Price Comparison Over Time")
    plt.ylabel("Price ($)")
    plt.xlabel("Year")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 2. Volume Trends - All companies together
    print("2. Volume Behavior (All Companies)")
    plt.figure(figsize=(14, 6))
    for name, df in dfs.items():
        if 'Volume' in df.columns:
            plt.plot(df.index, df['Volume'], label=f"{name}", color=color_map[name], alpha=0.7, linewidth=1)
    plt.title("Volume Comparison Over Time")
    plt.ylabel("Volume")
    plt.xlabel("Year")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 3. Return Seasonality - Daily (All companies)
    if all('Return' in df.columns for df in dfs.values()):
        print("\n3. Return Seasonality - Daily (All Companies)")
        plt.figure(figsize=(12, 6))
        x = np.arange(len(daynames))
        width = 0.2
        multiplier = 0
        
        for name, df in dfs.items():
            daily_ret = df.groupby('Day')['Return'].mean()
            daily_ret = daily_ret.reindex(daynames)
            offset = width * multiplier
            plt.bar(x + offset, daily_ret.values, width, label=name, color=color_map[name], alpha=0.8)
            multiplier += 1
        
        plt.xlabel('Day of Week')
        plt.ylabel('Avg Return')
        plt.title('Average Return by Day of Week - Comparison')
        plt.xticks(x + width * (len(company_names) - 1) / 2, daynames, rotation=45)
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Box plot comparison
        plt.figure(figsize=(14, 6))
        combined_data = []
        combined_labels = []
        for name, df in dfs.items():
            for day in daynames:
                day_data = df[df['Day'] == day]['Return'].dropna()
                combined_data.append(day_data)
                combined_labels.append(f"{name}\n{day}")
        
        # Create a more readable box plot
        plot_data = []
        plot_labels = []
        for day in daynames:
            for name in company_names:
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
        
        for i, day in enumerate(daynames):
            for j, name in enumerate(company_names):
                if name in dfs:
                    day_data = dfs[name][dfs[name]['Day'] == day]['Return'].dropna()
                    if len(day_data) > 0:
                        positions.append(i * (len(company_names) + 1) + j)
                        data_to_plot.append(day_data)
                        labels_to_plot.append(name)
        
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True)
        for patch, name in zip(bp['boxes'], labels_to_plot):
            patch.set_facecolor(color_map[name])
            patch.set_alpha(0.7)
        
        ax.set_xticks([i * (len(company_names) + 1) + (len(company_names) - 1) / 2 for i in range(len(daynames))])
        ax.set_xticklabels(daynames, rotation=45)
        ax.set_ylabel('Return')
        ax.set_title('Return Distribution by Day of Week - Comparison')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[name], alpha=0.7, label=name) for name in company_names]
        ax.legend(handles=legend_elements, loc='upper right')
        plt.tight_layout()
        plt.show()
    
    # 4. Return Seasonality - Monthly (All companies)
    if all('Return' in df.columns for df in dfs.values()):
        print("4. Return Seasonality - Monthly (All Companies)")
        plt.figure(figsize=(14, 6))
        x = np.arange(len(monthnames))
        width = 0.2
        multiplier = 0
        
        for name, df in dfs.items():
            monthly_ret = df.groupby('Month')['Return'].mean()
            monthly_ret = monthly_ret.reindex(range(1, 13))
            offset = width * multiplier
            plt.bar(x + offset, monthly_ret.values, width, label=name, color=color_map[name], alpha=0.8)
            multiplier += 1
        
        plt.xlabel('Month')
        plt.ylabel('Avg Return')
        plt.title('Average Return by Month - Comparison')
        plt.xticks(x + width * (len(company_names) - 1) / 2, monthnames, rotation=45)
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # 5. Volume Seasonality - Daily (All companies)
    print("5. Volume Seasonality - Daily (All Companies)")
    plt.figure(figsize=(12, 6))
    x = np.arange(len(daynames))
    width = 0.2
    multiplier = 0
    
    for name, df in dfs.items():
        daily_vol = df.groupby('Day')['Volume'].mean()
        daily_vol = daily_vol.reindex(daynames)
        offset = width * multiplier
        plt.bar(x + offset, daily_vol.values, width, label=name, color=color_map[name], alpha=0.8)
        multiplier += 1
    
    plt.xlabel('Day of Week')
    plt.ylabel('Avg Volume')
    plt.title('Average Volume by Day of Week - Comparison')
    plt.xticks(x + width * (len(company_names) - 1) / 2, daynames, rotation=45)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 6. Volume Seasonality - Monthly (All companies)
    print("6. Volume Seasonality - Monthly (All Companies)")
    plt.figure(figsize=(14, 6))
    x = np.arange(len(monthnames))
    width = 0.2
    multiplier = 0
    
    for name, df in dfs.items():
        monthly_vol = df.groupby('Month')['Volume'].mean()
        monthly_vol = monthly_vol.reindex(range(1, 13))
        offset = width * multiplier
        plt.bar(x + offset, monthly_vol.values, width, label=name, color=color_map[name], alpha=0.8)
        multiplier += 1
    
    plt.xlabel('Month')
    plt.ylabel('Avg Volume')
    plt.title('Average Volume by Month - Comparison')
    plt.xticks(x + width * (len(company_names) - 1) / 2, monthnames, rotation=45)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 7. Volatility Analysis (All companies)
    if all('Return' in df.columns for df in dfs.values()):
        print("7. Volatility Analysis (All Companies)")
        # Calculate volatility for each company
        for name, df in dfs.items():
            df['Vol20'] = df['Return'].rolling(20).std()
        
        plt.figure(figsize=(14, 6))
        for name, df in dfs.items():
            if 'Vol20' in df.columns:
                plt.plot(df.index, df['Vol20'], label=f"{name} (20-day)", color=color_map[name], alpha=0.8, linewidth=1.5)
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
        x = np.arange(len(daynames))
        width = 0.2
        multiplier = 0
        
        for name, df in dfs.items():
            if 'Vol20' in df.columns:
                daily_vol20 = df.groupby('Day')['Vol20'].mean()
                daily_vol20 = daily_vol20.reindex(daynames)
                offset = width * multiplier
                plt.bar(x + offset, daily_vol20.values, width, label=name, color=color_map[name], alpha=0.8)
                multiplier += 1
        
        plt.xlabel('Day of Week')
        plt.ylabel('Avg Volatility (20-day)')
        plt.title('Average Volatility by Day of Week - Comparison')
        plt.xticks(x + width * (len(company_names) - 1) / 2, daynames, rotation=45)
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Volatility by month
        print("Volatility by Month (All Companies)")
        plt.figure(figsize=(14, 6))
        x = np.arange(len(monthnames))
        width = 0.2
        multiplier = 0
        
        for name, df in dfs.items():
            if 'Vol20' in df.columns:
                monthly_vol20 = df.groupby('Month')['Vol20'].mean()
                monthly_vol20 = monthly_vol20.reindex(range(1, 13))
                offset = width * multiplier
                plt.bar(x + offset, monthly_vol20.values, width, label=name, color=color_map[name], alpha=0.8)
                multiplier += 1
        
        plt.xlabel('Month')
        plt.ylabel('Avg Volatility (20-day)')
        plt.title('Average Volatility by Month - Comparison')
        plt.xticks(x + width * (len(company_names) - 1) / 2, monthnames, rotation=45)
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # 8. Moving Averages (All companies)
    if all('Close' in df.columns for df in dfs.values()):
        print("8. Moving Averages (All Companies)")
        for name, df in dfs.items():
            df['MA20'] = df['Close'].rolling(20).mean()
            df['MA50'] = df['Close'].rolling(50).mean()
        
        plt.figure(figsize=(14, 6))
        for name, df in dfs.items():
            plt.plot(df.index, df['Close'], label=f"{name} Close", color=color_map[name], alpha=0.5, linewidth=1)
            if 'MA20' in df.columns:
                plt.plot(df.index, df['MA20'], label=f"{name} MA20", color=color_map[name], linestyle='--', alpha=0.7, linewidth=1)
        plt.title("Close Price and 20-day Moving Average - Comparison")
        plt.ylabel("Price ($)")
        plt.xlabel("Year")
        plt.legend(ncol=2, fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # 9. Statistical tests for each company
    print("\n9. Statistical Seasonality Tests")
    for name, df in dfs.items():
        if 'Return' in df.columns:
            print(f"\n{name}:")
            st.display_seasonality_results(st.test_seasonality(df, 'Return', 'Day'))
            st.display_seasonality_results(st.test_seasonality(df, 'Return', 'Month'))


