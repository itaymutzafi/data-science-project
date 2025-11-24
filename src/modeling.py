from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator


def create_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Basic train-test splitting.
    Final modeling strategy will be defined after topic approval.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(model: BaseEstimator, X_train: pd.DataFrame, y_train: pd.Series) -> BaseEstimator:
    """
    Fit a model to the training data.

    Returns
    -------
    BaseEstimator
        The trained model.
    """
    model.fit(X_train, y_train)
    return model
