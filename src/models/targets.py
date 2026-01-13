import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
from src.models import run_binary_cls_with_feature_importance
from src.config import COMPANY_COLORS, DEF_SPLITS
from src.models.binary_classification import models_for_target as models
from sklearn.linear_model import LogisticRegression


def target_daily(df):
    return (df["Log_Return"].shift(-1) > 0).astype(int)

def target_threshold(df, thr=0.001):
    return (df["Log_Return"].shift(-1) > thr).astype(int)

def target_multiday(df, h=3):
    future_sum = df["Log_Return"].shift(-1).rolling(h).sum()
    return (future_sum.shift(-(h-1)) > 0).astype(int)

TARGETS = {
    "daily": lambda df: target_daily(df),
    "threshold": lambda df: target_threshold(df, 0.001),
    "multiday": lambda df: target_multiday(df, 3),
}


def check_targets(dfs: Dict[str, pd.DataFrame], n_splits:int = DEF_SPLITS):
    for model in models:
        targets_df = run_model_for_target(dfs, model, n_splits)
        evaluation_metrics_target_plt(targets_df)
        print(avg_accuracy_per_target(targets_df))

def run_model_for_target(dfs: Dict[str, pd.DataFrame], model, n_splits: int = DEF_SPLITS) -> pd.DataFrame:
    all_summary_rows = []
    
    for target_name, target_fn in TARGETS.items():
        print(f"Checking for target: {target_name}")
        for ticker, df_orig in dfs.items():
            df = df_orig.copy()
            df["TargetBinary"] = target_fn(df)
            df = df.dropna()
            n_after = len(df)

            results_df = run_binary_cls_with_feature_importance(
                data=df,
                target_col="TargetBinary",
                model=model,
                ticker=ticker,
                n_splits=n_splits
            )

            mean_all = results_df.drop(columns=["Fold"]).mean(numeric_only=True)

            all_summary_rows.append({
                **mean_all.to_dict(),
                "Ticker": ticker,
                "Target": target_name,
                "Rows": n_after
            })
    
    return pd.DataFrame(all_summary_rows)

def evaluation_metrics_target_plt(df):
    metric = "Accuracy"
    plt.figure(figsize=(9, 4))

    for ticker in df["Ticker"].unique():
        sub = df[df["Ticker"] == ticker]

        plt.scatter(
            sub["Target"],
            sub[metric],
            color=COMPANY_COLORS.get(ticker),
            label=ticker,
            s=60,
            alpha=0.8
        )

    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Target Definition")
    plt.ylabel(metric)
    plt.title(f"{metric} by Target")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def avg_accuracy_per_target(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df
        .groupby("Target", as_index=False)["Accuracy"]
        .mean()
        .round(4)
        .rename(columns={"Accuracy": "AvgAccuracy"})
    )
