"""Model registry and factory for dynamic instantiation.

This module provides a centralized factory pattern for retrieving machine learning model instances.
It handles generic initialization as well as model-specific requirements (e.g., input size for LSTMs).
"""

from typing import Any, Dict, List, Optional
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVR
from . import baselines
from . import advanced

try:
    from xgboost import XGBRegressor, XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


def get_model(name: str, input_size: Optional[int] = None, **kwargs) -> Any:
    """Factory function to retrieve a model instance by name.

    Handles instantiation of various regression and classification models.
    Automatically routes `input_size` to models that require it (e.g., LSTM)
    and filters it out for models that do not.

    Args:
        name (str): Name of the model (e.g., 'Ridge', 'LSTM', 'NaiveBaseline').
        input_size (Optional[int]): Number of input features. Required for some deep learning models.
        **kwargs: Additional hyperparameters to pass to the model constructor.

    Returns:
        Any: Instantiated model (sklearn-compatible interface).

    Raises:
        ValueError: If the model name is unknown.
        ImportError: If a required dependency (e.g., XGBoost) is missing.
    """
    # --- Baselines ---
    if name == "NaiveBaseline":
        return baselines.NaiveBaseline(**kwargs)
    elif name == "RandomBaseline":
        return baselines.RandomBaseline(**kwargs)
    elif name == "MarketBenchmark":
        return baselines.MarketBenchmark(**kwargs)
    elif name == "ClassificationBaselineRandom":
        return baselines.ClassificationBaseline(**kwargs, strategy="random")
    elif name == "ClassificationBaselineZero":
        return baselines.ClassificationBaseline(**kwargs, strategy="always_0")
    elif name == "ClassificationBaselineOne":
        return baselines.ClassificationBaseline(**kwargs, strategy="always_1")
    elif name == "ClassificationBaselineMajor":
        return baselines.ClassificationBaseline(**kwargs, strategy="majority")

    # --- Standard ML Models (Scikit-Learn) ---
    elif name == "Ridge":
        return Ridge(**kwargs)
    elif name == "SVR":
        return SVR(**kwargs)
    elif name == "RandomForest":
        return RandomForestRegressor(**kwargs)
    elif name == "RandomForestClassifier":
        # Default n_estimators=100 if not specified
        kwargs.setdefault('n_estimators', 100)
        return RandomForestClassifier(**kwargs)
    elif name == "LogisticRegression":
        return LogisticRegression(**kwargs)

    # --- XGBoost ---
    elif name == "XGBRegressor":
        if not _XGB_AVAILABLE:
            raise ImportError("XGBoost not installed.")
        return XGBRegressor(**kwargs)
    elif name == "XGBClassifier":
        if not _XGB_AVAILABLE:
            raise ImportError("XGBoost not installed.")
        return XGBClassifier(**kwargs)
    elif name == "XGB_Conservative":
        if not _XGB_AVAILABLE:
            raise ImportError("XGBoost not installed.")
        # Preset: Lower learning rate, shallow depth
        defaults = {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.01, 'objective': 'reg:squarederror'}
        defaults.update(kwargs)
        return XGBRegressor(**defaults)
    elif name == "XGB_Aggressive":
        if not _XGB_AVAILABLE:
            raise ImportError("XGBoost not installed.")
        # Preset: Higher learning rate, deeper trees
        defaults = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'objective': 'reg:squarederror'}
        defaults.update(kwargs)
        return XGBRegressor(**defaults)

    # --- Advanced / Deep Learning ---
    elif name == "LSTM":
        if input_size is None:
            # Check kwargs for override
            if 'input_size' in kwargs:
                input_size = kwargs.pop('input_size')
        
        if input_size is None:
             raise ValueError(f"Model '{name}' requires 'input_size' to be passed to get_model.")

        return advanced.LSTMRegressor(input_size=input_size, **kwargs)

    elif name == "RandomForest_Deep":
        # Preset: Deeper trees, more estimators
        defaults = {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 2}
        defaults.update(kwargs)
        return RandomForestRegressor(**defaults)

    else:
        raise ValueError(f"Unknown model name: {name}")


def list_available_models() -> List[str]:
    """Returns a list of all registered model names.

    Returns:
        List[str]: List of valid model identifiers.
    """
    return [
        "NaiveBaseline",
        "RandomBaseline",
        "MarketBenchmark",
        "CAPMBaseline",
        "ClassificationBaselineRandom",
        "ClassificationBaselineZero",
        "ClassificationBaselineOne",
        "ClassificationBaselineMajor",
        "Ridge",
        "SVR",
        "RandomForest",
        "RandomForestClassifier",
        "LogisticRegression",
        "XGBRegressor",
        "XGBClassifier",
        "LSTM",
        "RandomForest_Deep",
        "XGB_Conservative",
        "XGB_Aggressive"
    ]
