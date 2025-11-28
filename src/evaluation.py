"""Evaluation module (skeleton).

Defines the business and technical metrics used in the project. Implementations
are intentionally omitted and should be implemented when the modeling
decisions are finalized.
"""

from typing import Any


def calculate_sharpe_ratio(returns, risk_free_rate: float = 0.0):
    """Compute (annualized) Sharpe ratio for a daily returns series.

    To be implemented.
    """
    raise NotImplementedError("calculate_sharpe_ratio to be implemented")


def evaluate_regression(y_true, y_pred):
    """Compute regression & business metrics (MSE, R2, DA, Sharpe).

    To be implemented.
    """
    raise NotImplementedError("evaluate_regression to be implemented")


def print_eval(metrics, model_name: str = "Model"):
    """Pretty-print evaluation metrics for reports.

    To be implemented.
    """
    raise NotImplementedError("print_eval to be implemented")
