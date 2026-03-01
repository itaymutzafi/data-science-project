import pandas as pd
from typing import Dict
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def get_metadata(dfs: Dict[str, pd.DataFrame]):
    print("===== shape =====")
    for t, df in dfs.items():
        print(t, df.shape)
    
    print("\n==== columns ====")
    for _, df in dfs.items():
        print(df.columns)
        break


def features_corellation(aapl_df: pd.DataFrame) -> None:
    numeric_df = aapl_df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr(method="pearson")

    plt.figure(figsize=(12,10))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Matrix (Pearson) - AAPL")
    plt.show()
