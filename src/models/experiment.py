"""Experiment Orchestrator.

This module manages the execution of the experimental grid, iterating over
combinations of time-series data (tickers), feature sets, and machine learning models.
"""

import itertools
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.features import sets, targets
from src.models import registry
from src.evaluation import metrics

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run.
    
    Attributes:
        tickers (List[str]): List of stock tickers to process.
        feature_sets (List[str]): Names of feature sets to test.
        models (List[str]): Names of models to evaluate (must match registry).
        target_type (str): 'continuous' (regression) or 'binary' (classification).
        target_horizon (int): Prediction horizon in days.
        start_date (str): Start date string (YYYY-MM-DD).
        end_date (str): End date string (YYYY-MM-DD).
        feature_set_params (Optional[Dict[str, List[str]]]): Custom feature set definitions.
        n_splits (int): Number of time-series cross-validation splits.
    """
    tickers: List[str]
    feature_sets: List[str]
    models: List[str]
    target_type: str
    target_horizon: int
    start_date: str
    end_date: str
    feature_set_params: Optional[Dict[str, List[str]]] = None
    n_splits: int = 3


class ExperimentRunner:
    """Executes the experiment grid search and tracks results."""

    def __init__(self, df_dict: Dict[str, pd.DataFrame], config: ExperimentConfig):
        """Initialize the runner.

        Args:
            df_dict (Dict[str, pd.DataFrame]): Dictionary mapping tickers to their dataframes.
            config (ExperimentConfig): Experiment configuration object.
        """
        self.raw_dfs = df_dict
        self.config = config
        self.results: List[Dict[str, Any]] = []

    def _prepare_data(self, ticker: str, feature_set: str) -> Tuple[pd.DataFrame, str, List[str]]:
        """Prepares X and y for a specific ticker and feature set.

        Args:
            ticker (str): Ticker symbol.
            feature_set (str): Name of the feature set.

        Returns:
            Tuple[pd.DataFrame, str, List[str]]:
                - Cleaned DataFrame with features and target.
                - Name of the target column.
                - List of valid feature column names.

        Raises:
            ValueError: If no features are available or data is empty.
        """
        df = self.raw_dfs[ticker].copy()

        # 1. Generate Target
        df, target_col = targets.create_target_variable(
            df,
            horizon=self.config.target_horizon,
            target_type=self.config.target_type
        )

        # 2. Select Features
        if isinstance(feature_set, list):
            feats = feature_set
        else:
            # Check config map first, then preset registry
            if self.config.feature_set_params and feature_set in self.config.feature_set_params:
                feats = self.config.feature_set_params[feature_set]
            else:
                feats = sets.get_feature_set(feature_set)

        # Verify features exist
        available_feats = [f for f in feats if f in df.columns]
        missing_feats = set(feats) - set(available_feats)
        if missing_feats:
            logger.warning(f"Ticker {ticker}: Missing features {missing_feats}")

        if not available_feats:
            raise ValueError(f"No features available for {ticker} in set {feature_set}")

        # 3. Filter DataFrame
        cols = available_feats + [target_col]
        # Ensure we have date index and no NaNs
        data = df[cols].dropna()
        return data, target_col, available_feats

    def _is_model_compatible(self, model_name: str) -> bool:
        """Checks if the model is compatible with the current target type.
        
        Args:
            model_name (str): Name of the model.

        Returns:
            bool: True if compatible, False otherwise.
        """
        is_classification = self.config.target_type in ['binary', 'multiclass']
        
        # Classification models
        cls_models = {"LogisticRegression", "RandomForestClassifier", "XGBClassifier"}
        # Regression models (explicit list to avoid ambiguity)
        reg_models = {"Ridge", "SVR", "RandomForest", "RandomForest_Deep", 
                      "XGB_Conservative", "XGB_Aggressive", "LSTM", "NaiveBaseline", 
                      "RandomBaseline", "MarketBenchmark", "CAPMBaseline"}

        if is_classification:
            if model_name in reg_models and model_name not in cls_models:
                # Strictly skip regressors for classification tasks
                return False
        else:
            if model_name in cls_models:
                # Strictly skip classifiers for regression tasks
                return False
        
        return True

    def run(self) -> None:
        """Execute the configured experiment grid."""
        iterator = itertools.product(
            self.config.tickers,
            self.config.feature_sets,
            self.config.models
        )

        # Estimate total work
        total_steps = len(self.config.tickers) * len(self.config.feature_sets) * len(self.config.models)

        logger.info(f"Starting Experiment: {total_steps} combinations.")
        logger.info(f"Target: {self.config.target_type} ({self.config.target_horizon}D)")

        for ticker, fset_name, model_name in tqdm(iterator, total=total_steps, desc="Running Grid"):
            try:
                self._run_single_experiment(ticker, fset_name, model_name)
            except Exception as e:
                logger.error(f"Failed {ticker}|{fset_name}|{model_name}: {e}")

    def _run_single_experiment(self, ticker: str, fset_name: str, model_name: str) -> None:
        """Runs a single combination of Ticker, FeatureSet, and Model.

        Args:
            ticker (str): Ticker symbol.
            fset_name (str): Feature set identifier.
            model_name (str): Model identifier.
        """
        # 1. Compatibility Check
        if not self._is_model_compatible(model_name):
            return

        # 2. Data Preparation
        data, target_col, feature_cols = self._prepare_data(ticker, fset_name)

        # 3. Model Instantiation
        # Pass input_size to registry; it handles whether the model needs it (e.g., LSTM) or ignores it.
        input_size = len(feature_cols)
        model = registry.get_model(model_name, input_size=input_size)

        # 4. Execution (Walk-Forward Validation)
        self._run_walk_forward_validation(data, target_col, model, model_name, ticker, fset_name)

    def _run_walk_forward_validation(
        self,
        data: pd.DataFrame,
        target_col: str,
        model: Any,
        model_name: str,
        ticker: str,
        fset_name: str
    ) -> None:
        """Performs Time-Series Walk-Forward validation.

        Splits data, scales features, trains model, and logs evaluation metrics.
        """
        X = data.drop(columns=[target_col])
        y = data[target_col]

        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Scale Features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Convert back to DataFrame to preserve column names (helpful for feature importance later)
            X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X.columns)
            X_val_scaled_df = pd.DataFrame(X_val_scaled, index=X_val.index, columns=X.columns)

            # Train
            logger.debug(f"  Fold {fold + 1}/{self.config.n_splits} | Training {model_name}...")
            try:
                model.fit(X_train_scaled_df, y_train)
            except Exception as e:
                logger.warning(f"  Fold {fold + 1} training failed for {model_name}: {e}")
                continue

            # Predict
            try:
                preds = model.predict(X_val_scaled_df)
                preds_series = pd.Series(preds, index=X_val.index)
            except Exception as e:
                logger.warning(f"  Fold {fold + 1} prediction failed for {model_name}: {e}")
                continue

            # Evaluate
            is_cls = self.config.target_type in ['binary', 'multiclass']
            if is_cls:
                fold_metrics = metrics.evaluate_classification(y_val, preds_series)
            else:
                fold_metrics = metrics.evaluate_regression(y_val, preds_series)

            # Store result
            result_row = {
                "Ticker": ticker,
                "FeatureSet": fset_name,
                "Model": model_name,
                "TargetType": self.config.target_type,
                "Diff": self.config.target_horizon,
                "Fold": fold,
                **fold_metrics
            }
            self.results.append(result_row)

    def get_results_df(self) -> pd.DataFrame:
        """Convert results list to DataFrame.

        Returns:
            pd.DataFrame: Aggregated results.
        """
        return pd.DataFrame(self.results)
