import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
from src.models import run_binary_cls_with_feature_importance
from src.config import COMPANY_COLORS
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


def check_targets(dfs: Dict[str, pd.DataFrame]):
    # for model in models:
        # targets_df = run_m/odel_for_target(dfs, )
        # evaluation_metrics_target_plt(targets_df)
    model = LogisticRegression()
    targets_df = run_model_for_target(dfs, model)
    evaluation_metrics_target_plt(targets_df)

def run_model_for_target(dfs: Dict[str, pd.DataFrame], model) -> pd.DataFrame:
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
                ticker=ticker
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
    metrics = ["Accuracy", "MSE"]

    _, axes = plt.subplots(
        nrows=len(metrics),
        figsize=(9, 4 * len(metrics))
    )

    for ax, metric in zip(axes, metrics):
        for ticker in df["Ticker"].unique():
            sub = df[df["Ticker"] == ticker]

            ax.plot(
                sub["Target"],
                sub[metric],
                marker="o",
                color=COMPANY_COLORS.get(ticker),
                label=ticker
            )

        ax.set_title(f"{metric} by Target")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)

    axes[0].legend()
    axes[-1].set_xlabel("Target Definition")
    plt.tight_layout()
    plt.show()
