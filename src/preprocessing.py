"""Preprocessing module (skeleton).

This module contains DataFrame-level glue logic such as merges, lags,
and target creation. Implementations are intentionally omitted so the
team can design the final pipeline collaboratively.
"""

from typing import Any


def merge_sentiment(fin_df, sentiment_df):
    """Left-join sentiment onto financial data and fill missing values.

    To be implemented by the team.
    """
    raise NotImplementedError("merge_sentiment to be implemented")


def create_target_next_day(df, return_col: str = 'Log_Return'):
    """Create T+1 return target column.

    To be implemented.
    """
    raise NotImplementedError("create_target_next_day to be implemented")


def basic_cleaning(df):
    """Perform minimal DataFrame cleaning.

    To be implemented.
    """
    raise NotImplementedError("basic_cleaning to be implemented")


def split_X_y(df, label_column: str):
    """Split DataFrame into X (features) and y (target).

    To be implemented.
    """
    raise NotImplementedError("split_X_y to be implemented")
