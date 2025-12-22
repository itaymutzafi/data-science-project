"""Trainer module.

Contains model training utilities including Walk-Forward Validation.
"""

import copy
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from src.evaluation.metrics import evaluate_regression

try:
    from sklearn.base import clone
except Exception:
    clone = None

def time_series_split(data: pd.DataFrame, n_splits: int = 5):
    """
    Performs time series split.
    Wrapper for sklearn TimeSeriesSplit.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return tscv.split(data)

def _init_model_instance(model_spec: Any) -> Any:
    """Return a fresh model instance from a class, factory, or existing estimator."""
    if callable(model_spec) and not hasattr(model_spec, "fit"):
        return model_spec()

    if clone is not None and hasattr(model_spec, "fit"):
        try:
            return clone(model_spec)
        except Exception:
            pass

    try:
        return copy.deepcopy(model_spec)
    except Exception:
        return model_spec

def train_and_evaluate(
    model: Any, 
    X: pd.DataFrame, 
    y: pd.Series, 
    n_splits: int = 5
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Runs Walk-Forward Validation (Expanding Window) on the model.
    
    1. Splits time series into k folds (preserving order).
    2. For each fold:
       - Train on indices [0 ... train_end]
       - Predict on indices [train_end+1 ... test_end]
    3. Aggregates predictions to form a continuous "out-of-sample" series.
    4. Computes metrics on the full out-of-sample series.
    
    Args:
        model: Model instance (must have fit and predict methods).
        X: Feature DataFrame.
        y: Target Series.
        n_splits: Number of Walk-Forward splits.
        
    Returns:
        metrics: Dictionary of aggregated performance metrics.
        predictions: DataFrame containing Actual vs Predicted for the test period.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    all_y_true = []
    all_y_pred = []
    
    print(f"Starting Walk-Forward Validation ({n_splits} splits)...")
    
    fold = 1
    for train_index, test_index in tscv.split(X):
        # 1. Split Data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # 2. Train (Fit)
        # Clone model or re-init? 
        # Ideally we should re-initialize to avoid data leakage from previous folds if the model keeps state.
        # But simple sklearn models usually reset on fit().
        # Our LSTM wrapper resets weights? No, __init__ sets them. fit() just runs train loop.
        # We should logically create a new instance or reset weights.
        # For simplicity here, assuming model.fit() retrains from scratch or essentially adapts.
        # IF we want strict retraining, we'd need a model factory.
        # Let's assume fit() is sufficient or we accept "online learning" if weights persist (which is also valid for TS).
        # Actually for strict Evaluation, complete retrain is cleaner.
        
        # Hack for PyTorch wrappers that don't reset:
        if hasattr(model, 'reset_weights'):
            model.reset_weights() # Theoretically
        
        model.fit(X_train, y_train)
        
        # 3. Predict
        preds = model.predict(X_test)
        
        # Handle NA from LSTM sequences or lags
        # Align indices
        valid_idx = preds.dropna().index.intersection(y_test.index)
        
        if len(valid_idx) == 0:
            print(f"Fold {fold}: No valid predictions (likely sequence length issue). Skipping.")
            continue
            
        fold_preds = preds.loc[valid_idx]
        fold_true = y_test.loc[valid_idx]
        
        all_y_true.append(fold_true)
        all_y_pred.append(fold_preds)
        
        # Optional: Print fold metrics
        # fold_metrics = evaluate_regression(fold_true, fold_preds)
        # print(f"Fold {fold}: MSE={fold_metrics['MSE']:.6f}")
        
        fold += 1
        
    # 4. Aggregate
    if not all_y_true:
        print("No predictions generated.")
        return {}, pd.DataFrame()
        
    full_y_true = pd.concat(all_y_true)
    full_y_pred = pd.concat(all_y_pred)
    
    # Calculate global metrics on concatenated OOS predictions
    metrics = evaluate_regression(full_y_true, full_y_pred)
    
    results_df = pd.DataFrame({
        'Actual': full_y_true,
        'Predicted': full_y_pred
    })
    
    return metrics, results_df


def run_experiment(
    models: Any,
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any] | None = None,
    logger: Any = None
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """Run walk-forward evaluation for one or many models with a common interface.

    Args:
        models: Estimator, model class/factory, or dict of {name: estimator/factory}.
        X: Feature matrix.
        y: Target series.
        config: Optional configuration dictionary. Supports 'n_splits' and 'experiment_name'.
        logger: Optional callable for logging progress (defaults to print).

    Returns:
        results_df: Metrics per model (index = model name).
        predictions: Mapping of model name to prediction series aligned to test indices.
    """
    cfg = config or {}
    n_splits = cfg.get("n_splits", 5)
    log = logger or (lambda msg: print(msg))

    model_dict = models if isinstance(models, dict) else {"model": models}
    metrics_store: Dict[str, Dict[str, float]] = {}
    predictions_store: Dict[str, pd.Series] = {}

    log(f"Running experiment with {len(model_dict)} model(s); n_splits={n_splits}.")
    for name, model_spec in model_dict.items():
        log(f"Evaluating {name}...")
        model_instance = _init_model_instance(model_spec)
        metrics, preds_df = train_and_evaluate(model_instance, X, y, n_splits=n_splits)
        if not metrics:
            log(f"{name}: no metrics produced.")
            continue
        metrics_store[name] = metrics
        if not preds_df.empty:
            predictions_store[name] = preds_df["Predicted"]

    results_df = pd.DataFrame(metrics_store).T if metrics_store else pd.DataFrame()
    return results_df, predictions_store
