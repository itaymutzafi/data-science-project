"""Experiment module.

Handles running model comparisons and experiments.
"""

from typing import Dict, Any, Tuple
import pandas as pd
from src.models import training
from src.evaluation import metrics
try:
    from IPython.display import display
except ImportError:
    display = print

def run_model_comparison(
    models: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """
    Runs Walk-Forward Validation for multiple models and compares results.
    
    Args:
        models: Dictionary of {model_name: model_instance}.
        X: Feature matrix.
        y: Target series.
        n_splits: Number of walk-forward splits.
        
    Returns:
        results_df: DataFrame of performance metrics for all models.
        predictions: Dictionary of {model_name: predicted_series} for plotting.
    """
    results = {}
    predictions = {}

    print("Running Walk-Forward Validation...")
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        try:
            model_metrics, model_preds = training.train_and_evaluate(model, X, y, n_splits=n_splits)
            if not model_metrics:
                print(f"Skipping {name} due to error.")
                continue

            results[name] = model_metrics
            # Store predictions (taking the 'Predicted' column)
            if not model_preds.empty:
                predictions[name] = model_preds['Predicted']

            # Print summary
            metrics.print_eval(model_metrics, name)

        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            import traceback
            traceback.print_exc()

    # Compile Metric Results
    results_df = pd.DataFrame(results).T
    
    print("\nSummary Results:")
    display(results_df)

    return results_df, predictions
