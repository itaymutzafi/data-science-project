from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
import pandas as pd
from src.evaluation.metrics import evaluate_classification
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from src.config import DEF_SPLITS

def run_binary_cls_with_feature_importance(
    data: pd.DataFrame,
    target_col: str,
    model,
    ticker: str,
    n_splits: int = DEF_SPLITS
):
    """
    Walk-forward validation based on experiment _run_walk_forward_validation
    """

    X = data.drop(columns=[target_col])
    y = data[target_col]

    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Scale Features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Convert back to DataFrame to preserve column names (helpful for feature importance later)
        X_train_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X.columns)
        X_val_df = pd.DataFrame(X_val_scaled, index=X_val.index, columns=X.columns)

        # Train
        model_fold = clone(model)
        model_fold.fit(X_train_df, y_train)

        # Predict
        preds = model_fold.predict(X_val_df)
        preds_series = pd.Series(preds, index=X_val.index)
        preds_series = (preds_series > 0.5).astype(int)

        fold_metrics = evaluate_classification(y_val, preds_series)

        all_results.append({
            "Ticker": ticker,
            "Fold": fold,
            **fold_metrics
        })

    return pd.DataFrame(all_results)


models_for_target = [LogisticRegression()] #, XGBClassifier()]