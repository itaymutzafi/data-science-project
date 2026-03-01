"""Abstract base class for all models in the project."""

from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
import joblib
from pathlib import Path


class BaseModel(ABC):
    """
    Abstract base class for all time-series regression models.
    Enforces a consistent Scikit-Learn like interface (fit, predict)
    with added persistence utilities.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """
        Train the model.

        Args:
            X: Feature matrix indexed by timestamp.
            y: Target series indexed by timestamp.
        
        Returns:
            self: The fitted model instance.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate predictions.

        Args:
            X: Feature matrix.

        Returns:
            pd.Series: Predictions aligned with X.index.
        """
        pass

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the model to disk. 
        Default implementation uses joblib serialization.
        Overridden by models requiring specific saving logic (e.g., PyTorch).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseModel":
        """
        Load a model from disk.
        Default implementation uses joblib deserialization.
        """
        return joblib.load(path)
