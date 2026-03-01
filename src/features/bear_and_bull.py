import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict


def bull_and_bear_check_window_size(feature_data: Dict[str, pd.DataFrame]):
    for ticker, df in feature_data.items():
        df = df.copy()

        if 'MA50' not in df.columns:
            df['MA50'] = df['Close'].rolling(50).mean()
        if 'MA200' not in df.columns:
            df['MA200'] = df['Close'].rolling(200).mean()

        plot_df = df.dropna(subset=['Close', 'MA50', 'MA200']).copy()
        if plot_df.empty:
            print(f"{ticker}: not enough history for MA50/MA200 regime plot")
            continue

        _, axes = plt.subplots(1, 2, figsize=(18, 4), sharey=True)

        # ---- Left: MA200 ----
        ax = axes[0]
        ax.plot(plot_df.index, plot_df['Close'], color='black', linewidth=1, label='Close')
        ax.plot(plot_df.index, plot_df['MA200'], color='blue', linewidth=1, label='MA200')
        bull200 = plot_df['Close'] > plot_df['MA200']
        ax.fill_between(plot_df.index, plot_df['Close'].min(), plot_df['Close'].max(),
                        where=bull200, color='green', alpha=0.15, label='Bull (MA200)')
        ax.fill_between(plot_df.index, plot_df['Close'].min(), plot_df['Close'].max(),
                        where=~bull200, color='red', alpha=0.10, label='Bear (MA200)')
        ax.set_title(f'{ticker}: MA200 Regime')
        ax.legend(loc='upper left')

        # ---- Right: MA50 ----
        ax = axes[1]
        ax.plot(plot_df.index, plot_df['Close'], color='black', linewidth=1, label='Close')
        ax.plot(plot_df.index, plot_df['MA50'], color='orange', linewidth=1, label='MA50')
        bull50 = plot_df['Close'] > plot_df['MA50']
        ax.fill_between(plot_df.index, plot_df['Close'].min(), plot_df['Close'].max(),
                        where=bull50, color='green', alpha=0.15, label='Bull (MA50)')
        ax.fill_between(plot_df.index, plot_df['Close'].min(), plot_df['Close'].max(),
                        where=~bull50, color='red', alpha=0.10, label='Bear (MA50)')
        ax.set_title(f'{ticker}: MA50 Regime')
        ax.legend(loc='upper left')

        plt.tight_layout()
        plt.show()


def smooth_regime_causal(regime_bool: pd.Series, min_len: int) -> pd.Series:
    """
    Applies a causal smoothing filter to binary signals.
    A regime change is only accepted if it persists for at least min_len days.
    """
    s = regime_bool.astype(int).copy()
    out = s.copy()

    if len(s) == 0:
        return out

    current = s.iloc[0]
    count = 0

    for i in range(len(s)):
        if s.iloc[i] == current:
            count = 0
        else:
            count += 1
            if count >= min_len:
                current = s.iloc[i]
                count = 0
        out.iloc[i] = current

    return out


def evaluate_regime_thresholds(feature_data: Dict[str, pd.DataFrame], 
                               min_day: int = 1, 
                               max_day: int = 30, 
                               penalty: int = 100) -> int:
    """
    EDA function to find the optimal smoothing threshold across all tickers.
    Returns the maximum optimal threshold (proj_threshold) found.
    """
    all_ticker_thresholds = []

    for ticker, df in feature_data.items():
        # Ensure MA200 exists
        ma200 = df['MA200'] if 'MA200' in df.columns else df['Close'].rolling(200).mean()
        valid = ma200.notna()
        raw_regime = (df.loc[valid, 'Close'] > ma200.loc[valid])

        if raw_regime.empty:
            continue

        rows = []
        for t in range(min_day, max_day + 1):
            sm = smooth_regime_causal(raw_regime, t)
            switches = (sm != sm.shift()).sum()
            flipped = (sm != raw_regime).mean()
            score = switches + penalty * flipped
            rows.append({'threshold': t, 'score': score})

        results = pd.DataFrame(rows)
        best_t = results.loc[results['score'].idxmin(), 'threshold']
        all_ticker_thresholds.append(best_t)
        print(f"Ticker {ticker}: Optimal smoothing threshold = {int(best_t)}")

    proj_threshold = int(max(all_ticker_thresholds)) if all_ticker_thresholds else 13
    print(f"Global Project Threshold determined: {proj_threshold} days")
    return proj_threshold


def make_regime_features(df: pd.DataFrame, bull_and_bear_threshold: int) -> pd.DataFrame:
    """
    Generates Bull/Bear binary features and cleaned Regime Strength.
    Zeroes out Strength during 'islands' (short-term noise) to reduce model error.
    """
    df = df.copy()

    # 1. Calculate base MA200 if missing
    if 'MA200' not in df.columns:
        df['MA200'] = df['Close'].rolling(200).mean()

    valid = df['MA200'].notna()
    
    # 2. Raw Signal: Is price above MA200?
    raw_regime = (df.loc[valid, 'Close'] > df.loc[valid, 'MA200'])

    # 3. Smoothed Regime: Apply global threshold to remove noise
    df['Regime_Bull'] = 0
    smoothed = smooth_regime_causal(raw_regime, bull_and_bear_threshold)
    df.loc[valid, 'Regime_Bull'] = smoothed.astype(int)
    
    # Forward fill to handle any gaps, then ensure integer type
    df['Regime_Bull'] = df['Regime_Bull'].ffill().fillna(0).astype(int)

    # 4. Regime Strength: Calculate distance from MA200
    raw_strength = (df.loc[valid, 'Close'] - df.loc[valid, 'MA200']) / df.loc[valid, 'MA200']
    
    # 5. Noise Filtering: If Raw Signal != Smoothed Regime (Island detected), zero out Strength
    df['Regime_Strength'] = 0.0
    is_consistent = (raw_regime == df.loc[valid, 'Regime_Bull'])
    
    # Use .where: Keep strength value only where consistent, else 0.0
    df.loc[valid, 'Regime_Strength'] = raw_strength.where(is_consistent, 0.0)

    return df
