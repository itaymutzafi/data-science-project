"""Transformers module (skeleton).

Stateless mathematical transformations and indicators live here.
Implementations are intentionally left as placeholders so the team can
re-implement from first-principles during development.
"""

from typing import Any


def calc_log_return(prices) -> Any:
    """Return the log-returns series for `prices`.

    To be implemented by the team.
    """
    raise NotImplementedError("calc_log_return to be implemented")


def moving_average(series, window: int):
    """Return rolling mean over `window` periods."""
    raise NotImplementedError("moving_average to be implemented")


def compute_rsi(series, window: int = 14):
    """Compute Relative Strength Index (RSI).

    Implementation intentionally omitted.
    """
    raise NotImplementedError("compute_rsi to be implemented")


def zscore(series):
    """Return z-score normalization of `series`."""
    raise NotImplementedError("zscore to be implemented")
