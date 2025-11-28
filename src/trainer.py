"""Trainer module (skeleton).

Contains model training utilities. Implementations are intentionally left
as placeholders so modelers can design experiments from first-principles.
"""

from typing import Any


def time_series_cv_train(model, X, y, n_splits: int = 3):
    """Train `model` using a time-series cross-validation strategy.

    Returns list of trained models or training artifacts.
    """
    raise NotImplementedError("time_series_cv_train to be implemented")


def train_holdout(model, X_train, y_train):
    """Train `model` on a single holdout training set.

    To be implemented.
    """
    raise NotImplementedError("train_holdout to be implemented")
