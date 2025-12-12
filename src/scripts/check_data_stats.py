import pandas as pd
import os

NEWS_PATH = 'data/raw/news_last_5y.csv'

def check_stats():
    if not os.path.exists(NEWS_PATH):
        print(f"File not found: {NEWS_PATH}")
        return

    try:
        # Load date and source only for speed
        df = pd.read_csv(NEWS_PATH, usecols=['date', 'source', 'company'], dtype=str)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        print(f"Total Rows: {len(df)}")
        print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
        
        print("\n--- Source Distribution ---")
        print(df['source'].value_counts())
        
        print("\n--- Date Range per Source ---")
        print(df.groupby('source')['date'].agg(['min', 'max', 'count']))

        print("\n--- Date Range per Company ---")
        print(df.groupby('company')['date'].agg(['min', 'max', 'count']))
        
        # Check specifically for Reddit
        reddit_df = df[df['source'].str.contains('reddit', case=False, na=False)]
        if not reddit_df.empty:
            print(f"\nReddit Data Range: {reddit_df['date'].min()} to {reddit_df['date'].max()}")
        else:
            print("\nNo explicited 'reddit' source found (check distinct sources above).")

    except Exception as e:
        print(f"Error checking stats: {e}")

if __name__ == "__main__":
    check_stats()
