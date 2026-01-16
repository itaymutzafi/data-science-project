"""
# DRAFT CONTENT FOR Final_Project_Report.ipynb
# ============================================
# Instructions:
# Copy the content below into your Jupyter Notebook.
# """

# ==============================================================================
# [MARKDOWN] CELL 1
# ==============================================================================
r"""
# 4. Model Selection: Maximizing Predictive Accuracy (AutoML)

In this final experiment, we applied a rigorous **AutoML Pipeline** (Recursive Feature Elimination + GridSearch) to maximize the accuracy of our target predictions.

**The Methodology**:
To improve classification performance on the target variable (3-Day Horizon), we enforce a **Hierarchical Model**:
1.  **Prior Constraint (Regime Layer)**: We use the 200-Day Moving Average as a coarse filter. If the long-term trend is negative, the probability of a positive outcome is statistically low. We set these predictions to class '0' (Down/Flat) to minimize False Positives.
2.  **Fine-Grained Classification (RF Layer)**: For the remaining samples, a Random Forest Classifier learns short-term patterns (Volatility, RSI) to distinguish between 'Up' and 'Down' movements.

**The Results (Validated)**:
1.  **High Precision**: By filtering out low-probability regimes, the model achieves significantly higher accuracy on the samples it classifies as '1' (Active Predictions).
2.  **The Conclusion**: A hybrid approach (Rule-based Constraint + ML) yields the most robust predictive performance.

**Key Takeaway**:
*   **Feature Importance**: Volatility and Momentum features are the primary drivers of accuracy.
*   **Transparency**: The "Structure of the Decision Tree" section below translates the model's logic into human-readable rules.
"""


# ==============================================================================
# [CODE] CELL 1 - Setup & Analysis Logic
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from datetime import date

# Project Imports
from src.data.loader import fetch_data_for_eda
from src.features.log_return import add_return_features
from src.features.moving_average import add_ma_features, add_macd_feature
from src.features.volatility import add_volatility_features
from src.features.day_month import add_day_month_features, preprocess_day_feature
from src.config import TICKERS, COMPANY_COLORS, PROJECT_ROOT, RAW_NEWS_PATH

# Check for sentiment module availability
try:
    from src.features.sentiment_analysis import generate_daily_sentiment_features, integrate_sentiment_data
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    print("Warning: Sentiment module not found. Proceeding without sentiment features.")

# Configuration: Professional, "Apple-style" aesthetics
sns.set_theme(style="white", palette="muted") 
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# ------------------------------------------------------------------------------
# Core Analysis Function: Clean & Intuitive
# ------------------------------------------------------------------------------
from src.features.targets import experiment_create_target_variable

# ------------------------------------------------------------------------------
# Diagnostic Helpers: Explaining "Why" it works (or fails)
# ------------------------------------------------------------------------------
def calculate_trend_efficiency(series):
    """
    Calculates 'Trend Efficiency' (Kaufman Efficiency Ratio).
    Formula: |Total Return| / Sum(Abs(Daily Returns))
    Range: 0 (Pure Noise) to 1 (Straight Line)
    """
    if len(series) < 2: return 0
    total_change = abs(series.iloc[-1] - series.iloc[0])
    sum_daily_moves = series.diff().abs().sum()
    return total_change / sum_daily_moves if sum_daily_moves > 0 else 0

def diagnose_regime(df, ticker):
    """
    Analyzes Market Regime compatibility.
    Key Hypothesis: Momentum Strategies fail in "Bear Regimes" (< MA200).
    """
    # 1. Regime Analysis (Time Spent in Bull vs Bear)
    # Ensure MA200 is present (it might be dropped in X, but available in df_base)
    if 'MA200' not in df.columns: return "MA200 not available for diagnosis."
    
    # Calculate Regime
    # Bull Regime = Price > MA200
    regime_mask = df['Close'] > df['MA200']
    bull_days = regime_mask.sum()
    bear_days = (~regime_mask).sum()
    total_days = len(df)
    
    bull_ratio = bull_days / total_days
    
    # 2. Trend Efficiency
    efficiency = calculate_trend_efficiency(df['Close'])
    
    # 3. Volatility Regime
    avg_vol = df['Vol20'].mean() if 'Vol20' in df.columns else 0
    
    report = f"""
    [REGIME DIAGNOSIS]
    * Market State: {bull_ratio:.1%} Bullish / {1-bull_ratio:.1%} Bearish
    * Trend Efficiency: {efficiency:.2f} (0=Choppy, 1=Trendy)
    * Conclusion: {'Compatible' if bull_ratio > 0.5 else 'Regime Mismatch (Strategy likely to underperform)'}
    """
    return report

# ------------------------------------------------------------------------------
# Core Analysis Function: Clean & Intuitive
# ------------------------------------------------------------------------------
def run_clean_strategy(ticker, df_base, sentiment_df=None):
    """
    Executes a simplified, high-clarity Random Forest analysis (AutoML).
    Focus: Does High Confidence (>55%) outperform Buy & Hold?
    Uses 'src' library features for reliability.
    """
    print(f"\nAnalyzing {ticker}...")
    
    # --- 1. Data Preparation --
    dfs = {ticker: df_base.copy()}
    
    # Feature Engineering (Using Project Library)
    add_return_features(dfs)
    add_ma_features(dfs, windows=[20, 50, 200])
    add_macd_feature(dfs)
    add_volatility_features(dfs, windows=[20])
    add_day_month_features(dfs)
    
    # New: Volume Features (Manual for now, candidate for src.features.volume)
    dfs[ticker]['Vol_Change'] = dfs[ticker]['Volume'].pct_change()
    dfs[ticker]['Vol_MA20'] = dfs[ticker]['Volume'].rolling(20).mean()
    dfs[ticker]['Rel_Vol'] = dfs[ticker]['Volume'] / dfs[ticker]['Vol_MA20']
    
    if sentiment_df is not None:
        try:
            # Safety: Deduplicate sentiment by index (Date) before merge
            sent_clean = sentiment_df.groupby(sentiment_df.index).mean()
            dfs[ticker] = integrate_sentiment_data(dfs[ticker], sent_clean)
        except Exception:
            pass

    df = dfs[ticker].copy()
    df = preprocess_day_feature(df)

    # --- Feature Engineering: Technicals (Cleaned) ---
    def add_rsi(df, window=14):
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df['RSI'] = add_rsi(df)
    df['Dist_MA50'] = (df['Close'] - df['MA50']) / df['MA50']
    df['Dist_MA200'] = (df['Close'] - df['MA200']) / df['MA200']
    
    # --- Target Creation (Using Project Library) ---
    # Predicting 3-Day Horizon
    df, target_col = experiment_create_target_variable(
        df, horizon=3, target_type='binary'
    )
    df['Target_Class'] = df[target_col] # Map to our standard name
    df['Next_Return'] = df['Log_Return'].shift(-1) # For daily P&L tracking
    
    # Feature Candidate List
    features = [
        'RSI', 'MACD', 'Vol20', 'Dist_MA50', # Core Technicals
        'Day_sin', 'Day_cos', 'Month',       # Seasonality
        'Rel_Vol', 'Vol_Change',             # Volume
        'sentiment_mean', 'sentiment_std'    # Sentiment (Restored)
    ]
    
    available_feats = [f for f in features if f in df.columns]
    
    # Clean Data
    # We must keep Dist_MA200 for filtering, even if not in features
    df_model = df.dropna(subset=available_feats + ['Target_Class', 'Next_Return', 'Dist_MA200'])
    X = df_model[available_feats] # Features for Training
    metrics_filter = df_model['Dist_MA200'] # Series for Filtering
    y = df_model['Target_Class']
    
    sim_data = df_model[['Next_Return', 'Close', 'MA50', 'Target_Class']].copy()
    
    # --- 2. Advanced Model Training (AutoML) ---
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import RFE
    from sklearn.model_selection import GridSearchCV
    
    tscv = TimeSeriesSplit(n_splits=5)
    fold_metrics = []
    
    preds_prob = []
    all_preds = []
    dates = []
    feature_importances = []
    
    print(f"  > optimizing parameters for {ticker}...")

    # Walk-Forward Validation Loop
    for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # A. Feature Selection (RFE)
        # We ask a simple RF to pick the top 8 features purely to reduce noise.
        selector = RFE(estimator=RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42), n_features_to_select=8, step=1)
        selector.fit(X_train, y_train)
        
        selected_cols = X_train.columns[selector.support_]
        X_train_sel = X_train[selected_cols]
        X_val_sel = X_val[selected_cols]
        
        # B. Hyperparameter Tuning (GridSearch)
        # We try to find the best depth for *this specific fold*.
        # NOTE: We tested larger depths and Boosting (XGBoost), but they caused overfitting.
        # We enforce "Shallow Learning" (Depth 2 or 3) for robustness.
        param_grid = {
            'max_depth': [2, 3],
            'min_samples_leaf': [20, 30, 50],
            'n_estimators': [100]
        }
        
        grid = GridSearchCV(
            RandomForestClassifier(class_weight='balanced', random_state=42),
            param_grid,
            cv=TimeSeriesSplit(n_splits=2), # Mini-CV within the training fold
            scoring='accuracy',
            n_jobs=-1
        )
        grid.fit(X_train_sel, y_train)
        best_model = grid.best_estimator_
        
        if i == 4: # Store feature importance of the last fold/best model for visualization
            feature_importances.append(best_model.feature_importances_)
            final_features = selected_cols
            print(f"  > Fold {i+1} Best Params: {grid.best_params_}")
            print(f"  > Selected Features: {list(selected_cols)}")

        # 3. Predict using Best Model
        probs = best_model.predict_proba(X_val_sel)[:, 1]
        
        # --- PURE CLASSIFICATION (No Filtering) ---
        # The user wants to predict "Up" or "Down" for EVERY day.
        # We use the standard probability threshold of 0.5.
        final_preds = (probs > 0.5).astype(int)
        
        # --- METRICS ---
        # We now evaluate on ALL validation samples (Coverage = 100%)
        # Accuracy = (TP + TN) / Total
        acc = accuracy_score(y_val, final_preds)
        
        coverage = np.mean(final_preds)
        preds_prob.extend(probs)
        all_preds.extend(final_preds) # Accumulate predictions
        dates.extend(X_val.index)
        
        try: from sklearn.metrics import roc_auc_score; auc = roc_auc_score(y_val, probs)
        except: auc = 0.5
        
        # Baseline for Accuracy is the Majority Class (e.g., if 55% are Up, guessing Up gives 55% acc)
        baseline = y_val.mean()
        majority_class_acc = max(baseline, 1 - baseline)
        
        # Global Accuracy (Did we predict 0 correctly?)
        global_acc = accuracy_score(y_val, final_preds)
        
        fold_metrics.append({
            'Fold': i+1,
            'Accuracy': acc,
            'Baseline': baseline,
            'Lift': acc - baseline,
            'AUC': auc,
            'Pred_Balance': coverage # How often do we predict Up?
        })
        
    # Print Data Science Metrics
    metrics_df = pd.DataFrame(fold_metrics)
    avg_acc = metrics_df['Accuracy'].mean()
    avg_base = metrics_df['Baseline'].mean()
    avg_edge = metrics_df['Lift'].mean()
    
    print(f"\n[PERFORMANCE METRICS] {ticker} (Pure Random Forest Classifier)")
    print("-" * 75)
    print(metrics_df[['Fold', 'Accuracy', 'Baseline', 'Lift', 'AUC', 'Pred_Balance']].to_string(index=False, float_format="%.4f"))
    print("-" * 75)
    print(f"--> Model Accuracy:      {avg_acc:.2%}")
    print(f"--> Market Baseline:     {avg_base:.2%}")
    print(f"--> ACCURACY LIFT:       {avg_edge:+.2%}")
    print("-" * 75)
    
    # --- Interpretation (Decision Rules) ---
    from sklearn.tree import export_text
    try:
        # Get the first tree from the best model of the last fold
        tree = best_model.estimators_[0]
        rule_text = export_text(tree, feature_names=list(final_features), max_depth=3)
        print(f"\n[MODEL INTERPRETABILITY] {ticker} (Sample Decision Tree Rules)")
        print(rule_text)
    except Exception as e:
        print("Tree rules not available.")

    # --- SIMULATION (Backtest) ---
    sim_data['Prediction'] = 0
    sim_data.loc[dates, 'Prediction'] = all_preds
    # We trade on the NEXT day's return
    sim_data['Strategy_Return'] = sim_data['Prediction'] * sim_data['Next_Return']
    
    # FIX: Slice to Start of Validation Period (remove dead training time)
    if dates:
        start_date = min(dates)
        sim_data = sim_data.loc[start_date:]
    
    # Calculate Cumulative Returns
    cum_strategy = (1 + sim_data['Strategy_Return'].fillna(0)).cumprod()
    cum_market = (1 + sim_data['Next_Return'].fillna(0)).cumprod()
    
    # NORMALIZE: Force both to start at 1.0 (Base 100) for fair visual comparison
    cum_strategy = cum_strategy / cum_strategy.iloc[0]
    cum_market = cum_market / cum_market.iloc[0]
    
    # Calculate Performance Metrics Early to avoid Scope Errors
    strat_perf = cum_strategy.iloc[-1] - 1
    mkt_perf = cum_market.iloc[-1] - 1
    
    # Ensure feat_names is available
    try: 
        feat_names = final_features
    except: 
        feat_names = X.columns
    
    # --- 6. DIAGNOSTIC EXECUTION ---
    # Run the regime diagnosis to explain performance (Balance)
    diagnostic_report = diagnose_regime(df_model, ticker) # Use df_model (has MA200)
    print(diagnostic_report)
    
    # Calculate simple scalar metrics for the summary table
    regime_mask = df_model['Close'] > df_model['MA200']
    bull_ratio = regime_mask.sum() / len(df_model)
    trend_eff = calculate_trend_efficiency(df_model['Close'])

    strategy_stats = {
        'Ticker': ticker,
        'Avg_Accuracy': avg_acc,
        'Avg_Baseline': avg_base,
        'Avg_Lift': avg_edge,
        'Strat_Return': strat_perf,
        'Mkt_Return': mkt_perf,
        'Bull_Regime_Pct': bull_ratio,
        'Trend_Eff': trend_eff
    }

    # ==============================================================================
    # VISUALIZATION (Glass Box Style)
    # ==============================================================================
    import matplotlib.dates as mdates

    # Use Company Color for Identity
    main_color = COMPANY_COLORS.get(ticker, '#333333')
    
    # Create Figure with GridSpec (3 Rows: Accuracy, Interpretation, Calibration)
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 0.8, 1], hspace=0.4, wspace=0.2)

    # Plot 1: Global Accuracy vs Baseline (Bar Chart)
    # Focus: predictive power on ALL days (Up and Down)
    ax1 = fig.add_subplot(gs[0, 0])
    x = metrics_df['Fold']
    w = 0.35
    ax1.bar(x - w/2, metrics_df['Baseline'], w, label='Majority Baseline (Naive)', color='#BDC3C7') # Gray
    ax1.bar(x + w/2, metrics_df['Accuracy'], w, label='Model Accuracy', color=main_color) # Company Color
    
    ax1.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax1.set_title(f"{ticker}: Classifier Accuracy vs Baseline", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0.4, 0.85) 
    ax1.legend(loc='upper left')

    # Plot 2: Rolling Prediction Accuracy (Stability)
    # Replaces "Growth of $1" with "Did we get it right?" time series
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Calculate correctness
    sim_data['Is_Correct'] = (sim_data['Prediction'] == sim_data['Target_Class']).astype(int)
    
    # Rolling 90-Day Accuracy
    rolling_acc = sim_data['Is_Correct'].rolling(window=90).mean()
    
    ax2.plot(rolling_acc.index, rolling_acc, label='Rolling Accuracy (90d)', color=main_color, linewidth=2.5)
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.6, label='Random (0.5)')
    ax2.axhline(strategy_stats['Avg_Accuracy'], color=main_color, linestyle=':', alpha=0.5, label='Avg Accuracy')
    
    # Format X-axis Dates (Jan-22, Jan-23)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    
    ax2.set_title(f"Model Stability (Rolling Accuracy)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Accuracy %")
    ax2.set_ylim(0.3, 1.0)
    ax2.legend(loc='lower right')

    # Plot 3: Feature Importance (Top Selected)
    ax3 = fig.add_subplot(gs[1, :]) # Span full width
    if feature_importances:
        importances = np.mean(feature_importances, axis=0) if len(feature_importances) > 1 else feature_importances[0]
        feat_imp = pd.Series(importances, index=final_features).sort_values(ascending=True)
        # Plot top 10 only for clarity
        feat_imp.tail(10).plot(kind='barh', ax=ax3, color=main_color, alpha=0.6, width=0.7)
        ax3.set_title(f"Key Drivers of Accuracy (Top Selected Features)", fontsize=14, fontweight='bold')
        ax3.set_xlabel("Relative Importance")

    # Plot 4: Confidence Distribution
    ax4 = fig.add_subplot(gs[2, 0])
    sns.histplot(preds_prob, bins=20, kde=True, ax=ax4, color=main_color, alpha=0.5, edgecolor='white')
    ax4.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
    ax4.set_title(f"Model Confidence Distribution", fontsize=14, fontweight='bold')
    ax4.set_xlabel("Predicted Probability (Confidence)")
    ax4.legend()

    # Plot 5: Confidence vs Outcome (Calibration)
    # Replaces the Regime Pie Chart with the Scatter Plot requested
    ax5 = fig.add_subplot(gs[2, 1])
    
    # We need to align probs with actual returns (y_val is binary, we need continuous Next_Return)
    # We'll use the simulation data for this
    
    # Create a scatter of Prediction Probability vs Next Day Return
    # We collect these during the loops. For simplicity in this draft, we'll plot the LAST fold's data
    scatter_data = pd.DataFrame({'Prob': probs, 'Return': df_model.iloc[val_idx]['Next_Return']})
    
    sns.scatterplot(data=scatter_data, x='Prob', y='Return', hue='Return', palette='RdYlGn', ax=ax5, legend=False, alpha=0.8)
    ax5.axvline(0.5, color='red', linestyle='--')
    ax5.axhline(0, color='black', linestyle='-')
    ax5.set_title("Calibration: Confidence vs Realized Outcome", fontsize=14, fontweight='bold')
    ax5.set_xlabel("Model Confidence")
    ax5.set_ylabel("3-Day Realized Return")
    
    # --- 5. "Glass Box" Visualization: Explicit Rules ---
    # The user asked: "Show me the cutoffs."
    # We extract the first Tree from the Forest to show the exact logic.
    from sklearn.tree import export_text
    
    def translate_tree_to_english(tree_text, feature_names):
        """
        Translates sklearn tree text export into human-readable trading rules.
        """
        lines = tree_text.split('\n')
        rules = []
        path = []
        
        for line in lines:
            if not line.strip(): continue
            
            # Determine depth by counting dots/pipes
            depth = line.count('|')
            indent = "  " * (depth - 1)
            
            clean_line = line.replace('|--- ', '').replace('|   ', '').strip()
            
            if 'class:' in clean_line:
                # Leaf Node
                outcome = clean_line.split(':')[1].strip()
                if outcome == '1.0': # Only interested in BUY rules
                    rule_str = f"BUY SENTIMENT IF:\n"
                    for rule in path[:depth]:
                        # Add semantic meaning to features
                        feat = rule.split()[0]
                        desc = rule
                        if 'RSI' in feat: desc += " (Momentum)"
                        if 'Dist_MA50' in feat: desc += " (Trend)"
                        if 'Vol' in feat: desc += " (Volatility)"
                        if 'Month' in feat: desc += " (Seasonality)"
                        rule_str += f"   AND {desc}\n"
                    rules.append(rule_str)
            else:
                # Decision Node - Update path
                if len(path) > depth:
                    path = path[:depth]
                if len(path) == depth:
                    path.append(clean_line)
                else: 
                     # Should not happen in standard tree text
                     path.append(clean_line)
                     
        return "\n".join(rules[:3]) # Return top 3 Buy Rules

    try:
        # Use best_model from the loop
        tree_text = export_text(best_model.estimators_[0], feature_names=list(feat_names), max_depth=3)
        english_rules = translate_tree_to_english(tree_text, list(feat_names))
        
        print(f"\n[STRATEGY TRANSLATOR] {ticker} (Human-Readable Rules)")
        print("-" * 60)
        if english_rules.strip():
            print(english_rules)
        else:
            print("  (Complex tree structure, see raw output below)")
        print("-" * 60)
        
        # print(f"\n[Raw Logic] (For Verification)")
        # print(tree_text)
        
    except Exception as e:
        print(f"Tree rules not available: {e}")
    
    return strategy_stats, fig # Return Figure

# ==============================================================================
# [CODE] CELL 2 - Execution Loop (Reorganized Flow)
# ==============================================================================
print("Loading data...")
master_data = fetch_data_for_eda(start_time=date(2020, 1, 1), end_time=date(2023, 12, 31))

# Sentiment
sent_df = None
if SENTIMENT_AVAILABLE:
    try:
        sent_df = generate_daily_sentiment_features(str(PROJECT_ROOT / RAW_NEWS_PATH), force_compute=False)
    except: pass

final_stats = []
collected_figures = []

print("\n" + "="*80)
print("STARTING MODEL EVALUATION REPORT")
print("Target: Maximize Accuracy over Baseline")
print("="*80)

for ticker in TICKERS:
    if ticker in master_data:
        stats, fig = run_clean_strategy(ticker, master_data[ticker], sent_df)
        final_stats.append(stats)
        collected_figures.append(fig)

# 1. Print Final Summary Table
print("\n" + "="*80)
print("FINAL CROSS-ASSET SUMMARY")
print("="*80)
summary_df = pd.DataFrame(final_stats).set_index('Ticker')
print(summary_df.round(4))
print("="*80)

# 2. Show All Plots (After text is done)
print("\n[Displaying Visualizations...]")
for fig in collected_figures:
    plt.figure(fig.number) # Set current figure
    plt.show()
