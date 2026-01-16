import pandas as pd
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from src.models import run_binary_cls_with_feature_importance
from src.config import DEF_SPLITS, COMPANY_COLORS

EXCLUDE_TARGET_COLS = ["TargetRegression", "TargetBinary"]

def forward_feature_selection(
    df: pd.DataFrame,
    target_col: str,
    model,
    ticker: str,
    max_features: int | None = None,
    n_splits: int = DEF_SPLITS
):
    results = []
    selected_features = []

    candidate_features = [
        c for c in df.columns
        if c not in [target_col] and c not in EXCLUDE_TARGET_COLS
    ]

    max_features = max_features or len(candidate_features)

    for step in range(1, max_features + 1):
        print(f"[{ticker}] Forward step {step}")
        best_feat = None
        best_score = -np.inf

        for feat in candidate_features:
            trial_features = selected_features + [feat]

            df_sub = df[trial_features + [target_col]]

            res = run_binary_cls_with_feature_importance(
                data=df_sub,
                target_col=target_col,
                model=model,
                ticker=ticker,
                n_splits=n_splits
            )

            score = res["Accuracy"].mean()

            if score > best_score:
                best_score = score
                best_feat = feat

        # update state
        selected_features.append(best_feat)
        candidate_features.remove(best_feat)

        results.append({
            "Ticker": ticker,
            "Step": step,
            "NumFeatures": len(selected_features),
            "AddedFeature": best_feat,
            "Accuracy": best_score,
        })

        print(f"[{ticker}] Added: {best_feat} | Accuracy: {best_score:.4f}")

    return pd.DataFrame(results)

def run_feature_selection(dfs: Dict[str, pd.DataFrame], n_splits: int = DEF_SPLITS) -> pd.DataFrame:
    all_forward_histories = []

    for ticker, df in dfs.items():
        print(f"\n==============================")
        print(f"Forward selection for {ticker}")
        print(f"==============================")

        df["TargetBinary"] = (df["Log_Return"].shift(-1) > 0).astype(int)
        df = df.dropna()

        history_df = forward_feature_selection(
            df=df,
            target_col="TargetBinary",
            model=LogisticRegression(),
            ticker=ticker,
            n_splits=n_splits
            # max_features=25
        )

        history_df["Ticker"] = ticker
        all_forward_histories.append(history_df)

    return pd.concat(all_forward_histories, ignore_index=True)

def feature_selection_plot(df: pd.DataFrame):
    plt.figure(figsize=(9, 6))

    for ticker in df["Ticker"].unique():
        sub = df[df["Ticker"] == ticker]
        color = COMPANY_COLORS.get(ticker)

        plt.plot(
            sub["NumFeatures"],
            sub["Accuracy"],
            marker="o",
            label=ticker,
            color=color
        )

        # mark max
        best_idx = sub["Accuracy"].idxmax()
        best_row = sub.loc[best_idx]

        plt.scatter(
            best_row["NumFeatures"],
            best_row["Accuracy"],
            s=100,
            color=color
        )

        plt.text(
            best_row["NumFeatures"],
            best_row["Accuracy"],
            str(int(best_row["NumFeatures"])),
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold"
        )

    plt.xlabel("Number of Features")
    plt.ylabel("Accuracy")
    plt.title("Forward Selection – Best Feature Count per Ticker")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def get_best_k_features(df: pd.DataFrame, K: int=10) -> Dict[str, List[str]]:
    best_features = {}

    for ticker in df["Ticker"].unique():
        sub = (
            df[df["Ticker"] == ticker]
            .sort_values("Step")
        )

        feats = sub["AddedFeature"].head(K).tolist()
        best_features[ticker] = feats

        print(f"\nTicker: {ticker}")
        print(f"Top {K} features:")
        for i, f in enumerate(feats, 1):
            print(f"  {i}. {f}")

    return best_features
