from traitlets import Bool
import xgboost as xgb
from src.config import TEST_SIZE
from src.evaluation.metrics import evaluate_regression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

def linear_xg_boost(df_s: Dict, base_features: List[str], check_peer: bool = True) -> None:
    target = "Log_Return"
    results = []

    for ticker, df in df_s.items():   
        if check_peer:
            peer_features = [col for col in df.columns if "-" in col]
        features = base_features + peer_features
        features = [f for f in features if f in df.columns]
        
        split_idx = int((1 - TEST_SIZE) * len(df))
        train_data = df.iloc[:split_idx, :]
        test_data = df.iloc[split_idx:, :]


        model = xgb.XGBRegressor()
        model.fit(train_data[features], train_data[target])
        predictions = model.predict(test_data[features])
        
        # metrics of the model
        pred_series = pd.Series(predictions, index=test_data.index)
        metrics = evaluate_regression(test_data[target], pd.Series(predictions, index=test_data.index))

        results.append({
            'Company': ticker,
            'R²': metrics['R2'],
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'Directional Accuracy': metrics['Directional Accuracy'],
            'Strategy Sharpe': metrics['Strategy Sharpe']
        })
        
    results_df = pd.DataFrame(results).set_index("Company")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(results_df.T, annot=True, fmt='.4f', cmap='RdYlGn', 
                cbar=True, linewidths=0.5, ax=ax, 
                xticklabels=results_df.index, yticklabels=results_df.columns, vmin = -1.0, vmax = 1.5)
    ax.set_title(f"XGboost regression model using the features {base_features}", 
                fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


def calssifier_xg_boost(df_s: Dict, base_features: List[str]):
    pass