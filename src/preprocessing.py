import pandas as pd
from typing import List, Tuple


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform minimal cleaning operations.
    Specific steps will be defined once the project dataset is selected.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.

    Returns
    -------
    pd.DataFrame
        A clean copy of the input.
    """
    df_clean = df.copy()
    df_clean = df_clean.dropna(how="all")
    return df_clean


def split_X_y(df: pd.DataFrame, label_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a DataFrame into features and labels.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    label_column : str
        Name of the target variable.

    Returns
    -------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Labels.
    """
    X = df.drop(columns=[label_column])
    y = df[label_column]
    return X, y
