"""Feature summary and quick inspection utilities."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict


def get_metadata(dfs: Dict[str, pd.DataFrame], *, verbose: bool = False) -> None:
    """Optionally print dataset shapes and first dataframe columns."""
    if not verbose:
        return

    print("===== shape =====")
    for t, df in dfs.items():
        print(t, df.shape)

    print("\n==== columns ====")
    for _, df in dfs.items():
        print(df.columns)
        break


def features_corellation(aapl_df: pd.DataFrame) -> None:
    """Plot a Pearson correlation heatmap for numeric features."""
    numeric_df = aapl_df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr(method="pearson")

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Matrix (Pearson) - AAPL")
    plt.show()
