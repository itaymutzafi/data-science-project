
import pandas as pd
import logging
from pathlib import Path
from typing import Optional
from src.evaluation import comparisons

logger = logging.getLogger(__name__)

def analyze_results_df(df: pd.DataFrame):
    """Prints a leaderboard summary of the provided results DataFrame."""
    print("\n" + "="*40)
    print(f"RESULTS SUMMARY (Total Rows: {len(df)})")
    print("="*40)
    
    # 1. Best Regression Models
    reg_df = df[df['TargetType'] == 'continuous']
    if not reg_df.empty:
        print("\n--- Regression Leaderboard (Sharpe) ---")
        print(reg_df.groupby(['Model'])['Strategy Sharpe'].mean().sort_values(ascending=False))
        
        print("\nTop 5 Model+FeatureSet combinations:")
        print(reg_df.groupby(['Model', 'FeatureSet'])['Strategy Sharpe'].mean().sort_values(ascending=False).head(5))

    # 2. Best Classification Models
    cls_df = df[df['TargetType'] == 'binary']
    if not cls_df.empty:
        print("\n--- Classification Leaderboard (F1) ---")
        print(cls_df.groupby(['Model'])['F1'].mean().sort_values(ascending=False))
        
        print("\nTop 5 Model+FeatureSet combinations:")
        print(cls_df.groupby(['Model', 'FeatureSet'])['F1'].mean().sort_values(ascending=False).head(5))

    # 3. Feature Set Analysis
    print("\n--- Best Feature Sets (Across all models) ---")
    if 'Strategy Sharpe' in df.columns:
        print(df.groupby('FeatureSet')['Strategy Sharpe'].mean().sort_values(ascending=False).head(5))

    # 4. Return Leaderboard (prefer Regression)
    if not reg_df.empty:
        return reg_df.groupby(['Model'])['Strategy Sharpe'].mean().sort_values(ascending=False).reset_index()
    elif not cls_df.empty:
        return cls_df.groupby(['Model'])['F1'].mean().sort_values(ascending=False).reset_index()
    return pd.DataFrame()
def generate_report_plots(df: pd.DataFrame, output_dir: Optional[str] = "reports/figures"):
    """Generates standard report plots from results DataFrame.
    
    Args:
        output_dir: Directory to save plots. If None, plots are shown inline.
    """
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path = None
    
    # Regression
    reg_data = df[df['TargetType'] == 'continuous']
    if not reg_data.empty:
        # 1. Main Leaderboard (Sharpe Strategy)
        comparisons.plot_leaderboard(
            reg_data, 
            'Strategy Sharpe', 
            'Leaderboard: Strategy Sharpe Ratio', 
            out_path / 'regression_sharpe_leaderboard.png' if out_path else None
        )
        
        # 2. Risk-Return Scatter (The most informative single chart)
        comparisons.plot_risk_return_scatter(
            reg_data, 
            out_path / 'risk_return_scatter.png' if out_path else None
        )
        
        # 3. Heatmap (Model vs Features)
        comparisons.plot_model_performance_heatmap(
            reg_data,
            'Strategy Sharpe',
            out_path / 'regression_sharpe_heatmap.png' if out_path else None
        )
        
        logger.info("Generated: Sharpe Leaderboard, Risk-Return Scatter, Performance Heatmap")
        
    # Classification
    cls_data = df[df['TargetType'] == 'binary']
    if not cls_data.empty:
        comparisons.plot_leaderboard(
            cls_data, 
            'F1', 
            'Classification F1 Score', 
            out_path / 'classification_f1_leaderboard.png' if out_path else None
        )
        comparisons.plot_model_performance_heatmap(
            cls_data,
            'F1',
            out_path / 'classification_f1_heatmap.png' if out_path else None
        )
    
    logger.info(f"Plots saved to {out_path}")

def load_and_analyze(csv_path: str):
    """Loads results from CSV and runs analysis."""
    if not Path(csv_path).exists():
        logger.error(f"File not found: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
        analyze_results_df(df)
        generate_report_plots(df)
    except Exception as e:
        logger.error(f"Error analyzing results: {e}")
