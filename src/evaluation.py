from typing import Dict, Any
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score


def evaluate_basic(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Compute simple evaluation metrics.
    More advanced evaluation will be added later based on the project needs.

    Returns
    -------
    dict with accuracy and F1 score.
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted")
    }
