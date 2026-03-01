import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from itertools import combinations
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import colorsys

from src.models import run_binary_cls_with_feature_importance, run_binary_cls_embedded_importance
from src.config import DEF_SPLITS, COMPANY_COLORS, SPLITS
from src.features import (
    build_feature_to_block_map,
    experiment_create_target_variable,
    preprocess_day_feature,
    preprocess_month_feature,
)
from src.utils.feature_names import canonicalize_feature_columns


EXCLUDE_TARGET_COLS = ["TargetRegression", "TargetBinary"]


def _ensure_binary_target(df: pd.DataFrame, target_col: str = "TargetBinary") -> pd.DataFrame:
    """Ensure a binary target column exists for feature-selection routines."""
    prepared = df.copy()
    if target_col not in prepared.columns:
        if "Log_Return" not in prepared.columns:
            raise KeyError(f"Missing '{target_col}' and 'Log_Return' is unavailable to derive it.")
        prepared[target_col] = (prepared["Log_Return"].shift(-1) > 0).astype(int)
    return prepared


def _as_ticker_frame_dict(dfs: Dict[str, pd.DataFrame] | pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Normalize feature-selection input into {ticker: dataframe} format."""
    if isinstance(dfs, dict):
        return {ticker: canonicalize_feature_columns(frame.copy()) for ticker, frame in dfs.items()}
    if isinstance(dfs, pd.DataFrame):
        if "Ticker" not in dfs.columns:
            raise ValueError("DataFrame input must include a 'Ticker' column for feature selection.")
        return {ticker: canonicalize_feature_columns(frame.copy()) for ticker, frame in dfs.groupby("Ticker")}
    raise TypeError("Feature selection input must be a dict[ticker, dataframe] or a dataframe with 'Ticker'.")


def _numeric_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    """Return numeric candidate features, excluding target helper columns."""
    excluded = {target_col, *EXCLUDE_TARGET_COLS}
    return [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]


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

    candidate_features = _numeric_feature_columns(df, target_col)

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

def run_feature_selection(dfs: Dict[str, pd.DataFrame] | pd.DataFrame, n_splits: int = DEF_SPLITS) -> pd.DataFrame:
    dfs = _as_ticker_frame_dict(dfs)
    all_forward_histories = []

    for ticker, df in dfs.items():
        print(f"\n==============================")
        print(f"Forward selection for {ticker}")
        print(f"==============================")

        df = _ensure_binary_target(df, "TargetBinary").dropna()

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
    plt.close()

def get_best_k_features(df: pd.DataFrame, K: int=10, add_prints: bool = True) -> Dict[str, List[str]]:
    best_features = {}

    for ticker in df["Ticker"].unique():
        sub = (
            df[df["Ticker"] == ticker]
            .sort_values("Step")
        )

        feats = sub["AddedFeature"].head(K).tolist()
        best_features[ticker] = feats

        if add_prints:
            print(f"\nTicker: {ticker}")
            print(f"Top {K} features:")
            for i, f in enumerate(feats, 1):
                print(f"  {i}. {f}")

    return best_features


def get_embedding_importance_features(
    dfs: dict,
    k: int = 10,
    target_col: str = "TargetBinary",
    model=None,
    n_splits: int = DEF_SPLITS,
    add_prints: bool = True
):
    if model is None:
        model = LogisticRegression()

    embedding_features = {}
    embedding_accuracy = {}

    for ticker, ticker_df in dfs.items():
        ticker_df = _ensure_binary_target(ticker_df, target_col).dropna()
        feature_cols = _numeric_feature_columns(ticker_df, target_col)
        if not feature_cols:
            embedding_features[ticker] = []
            embedding_accuracy[ticker] = np.nan
            continue
        model_df = ticker_df[feature_cols + [target_col]]

        # --- 1. Get feature importance ---
        imp = run_binary_cls_embedded_importance(
            data=model_df,
            target_col=target_col,
            model=model,
            ticker=ticker,
            n_splits=n_splits,
        )

        if imp.empty:
            embedding_features[ticker] = []
            embedding_accuracy[ticker] = np.nan
            continue

        top_features = (
            imp.groupby("Feature")["Importance"]
            .mean()
            .sort_values(ascending=False)
            .head(k)
            .index
            .tolist()
        )

        embedding_features[ticker] = top_features

        # --- 2. Evaluate accuracy using same pipeline ---
        res = run_binary_cls_with_feature_importance(
            data=model_df[top_features + [target_col]],
            target_col=target_col,
            model=model,
            ticker=ticker,
            n_splits=n_splits
        )

        embedding_accuracy[ticker] = res["Accuracy"].mean()

        if add_prints:
            print(
                f"{ticker}: "
                f"Accuracy={embedding_accuracy[ticker]:.4f} | "
                f"Top-{k} features={top_features}"
            )

    return embedding_features, embedding_accuracy


def get_experimet_lr_best_features(
    results_cls: pd.DataFrame,
    ticker_diverse_sets: Dict[str, Dict[int, List[str]]],
) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
    """
    For each ticker:
    - consider only LogisticRegression
    - aggregate Accuracy over folds
    - select the FeatureSet with highest mean Accuracy

    Returns:
        1) dict[ticker] -> list of feature names
        2) dict[ticker] -> best accuracy
    """
    # 1. Filter to LogisticRegression only
    df = results_cls[results_cls["Model"] == "LogisticRegression"].copy()

    if df.empty:
        raise ValueError("No LogisticRegression results found")

    # 2. Aggregate Accuracy over folds
    agg = (
        df
        .groupby(["Ticker", "FeatureSet"])["Accuracy"]
        .mean()
        .reset_index()
    )

    # 3. Pick best FeatureSet per ticker
    best_per_ticker = (
        agg
        .sort_values("Accuracy", ascending=False)
        .groupby("Ticker", as_index=False)
        .first()
    )

    # 4. Build outputs
    best_features = {}
    best_accuracy = {}

    for _, row in best_per_ticker.iterrows():
        ticker = row["Ticker"]

        # FeatureSet is stored as string in results CSV; convert to int to match
        # the keys inside ticker_diverse_sets (1..n). If conversion fails or the
        # id is missing, raise a clear error instead of returning None later.
        try:
            fset_id = int(row["FeatureSet"])
        except (TypeError, ValueError):
            raise ValueError(f"Invalid FeatureSet id for ticker {ticker}: {row['FeatureSet']}")

        acc = row["Accuracy"]

        features = ticker_diverse_sets[ticker].get(fset_id)
        if features is None:
            raise KeyError(f"FeatureSet {fset_id} not found for ticker {ticker} in ticker_diverse_sets")

        best_features[ticker] = features
        best_accuracy[ticker] = acc
        print(f"{ticker}: {features}")

    return best_features, best_accuracy


def count_blocks(features_by_ticker: Dict[str, List[str]], feature_to_block: Dict[str, str]) -> Dict[str, int]:
    """
    Count how many times each block appears across all tickers.
    """
    counts = defaultdict(int)

    for features in features_by_ticker.values():
        for f in features:
            block = feature_to_block.get(f)
            if block:
                counts[block] += 1

    return counts


def build_block_count_df(
    experiment_features: Dict[str, List[str]],
    embedded_features: Dict[str, List[str]],
    wrapper_features: Dict[str, List[str]],
) -> pd.DataFrame:
    feature_to_block, blocks = build_feature_to_block_map()

    exp_counts = count_blocks(experiment_features, feature_to_block)
    emb_counts = count_blocks(embedded_features, feature_to_block)
    wrap_counts = count_blocks(wrapper_features, feature_to_block)

    all_blocks = sorted(blocks.keys())

    df = pd.DataFrame({
        "Block": all_blocks,
        "Experiment": [exp_counts.get(b, 0) for b in all_blocks],
        "Embedding": [emb_counts.get(b, 0) for b in all_blocks],
        "Wrapper": [wrap_counts.get(b, 0) for b in all_blocks],
    })

    return df


def plot_block_usage_stacked(df: pd.DataFrame):
    plt.figure(figsize=(12, 7))
    bottom = np.zeros(len(df))
    colors = {
        "Experiment": "#4C72B0",    
        "Embedding": "#DD8452",        
        "Wrapper": "#55A868", 
    }

    for col in ["Experiment", "Embedding", "Wrapper"]:
        plt.bar(
            df["Block"],
            df[col],
            bottom=bottom,
            label=col,
            color=colors[col],
            edgecolor="none",   
        )
        bottom += df[col].values

    plt.xlabel("Feature Block", fontsize=14, weight="bold")
    plt.ylabel("Feature Count (across all tickers)", fontsize=14, weight="bold")
    plt.title("Feature Block Usage by Selection Strategy", fontsize=16, weight="bold")

    plt.xticks(rotation=35, ha="right", fontsize=12)
    max_y = int(bottom.max())
    plt.yticks(range(0, max_y + 1, 1), fontsize=12)

    plt.legend(title="Source", fontsize=11, title_fontsize=12)
    plt.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    
    plt.show()
    plt.close()


def powerset(features: List[str]) -> List[List[str]]:
    """
    Return all non-empty subsets of features.
    """
    subsets = []
    for r in range(1, len(features) + 1):
        subsets.extend(combinations(features, r))
    return [list(s) for s in subsets]


def evaluate_feature_subsets(
    data: pd.DataFrame,
    target_col: str,
    subsets: List[List[str]],
    model,
    ticker: str,
    n_splits: int,
    original_features: List[str],
    original_accuracy: float,
) -> pd.DataFrame:
    rows = []
    data = data.dropna()
    original_set = set(original_features)

    for subset_idx, features in enumerate(subsets):
        valid_features = [f for f in features if f in data.columns and f != target_col]
        feature_set = set(valid_features)

        # --- CASE 1: this is the original best set ---
        if feature_set == original_set:
            acc = original_accuracy

        # --- CASE 2: evaluate normally ---
        else:
            if not valid_features:
                acc = np.nan
            else:
                df_sub = data[valid_features + [target_col]]

                res = run_binary_cls_with_feature_importance(
                    data=df_sub,
                    target_col=target_col,
                    model=model,
                    ticker=ticker,
                    n_splits=n_splits
                )

                acc = res["Accuracy"].mean() if not res.empty else np.nan

        rows.append({
            "Ticker": ticker,
            "SubsetIndex": subset_idx,
            "NumFeatures": len(valid_features),
            "Accuracy": acc,
            "Features": valid_features,
        })

    return pd.DataFrame(rows)


def get_subset_results(
    top_features: Dict[str, List[str]],
    dfs: Dict[str, pd.DataFrame],
    original_accuracy: Dict[str, float],
):
    all_results = []

    for ticker, feature_set in top_features.items():
        prepared_df = _ensure_binary_target(dfs[ticker], "TargetBinary").dropna()
        available_features = [f for f in feature_set if f in prepared_df.columns]
        missing_features = [f for f in feature_set if f not in prepared_df.columns]
        if missing_features:
            print(f"{ticker}: dropped unavailable features {missing_features}")

        if not available_features:
            print(f"{ticker}: no available features left for subset evaluation")
            continue

        subsets = powerset(available_features)
        print(f"{ticker}: {len(subsets)} subsets")

        df_subsets = evaluate_feature_subsets(
            data=prepared_df,
            target_col="TargetBinary",
            subsets=subsets,
            model=LogisticRegression(),
            ticker=ticker,
            n_splits=SPLITS,
            original_features=available_features,
            original_accuracy=original_accuracy.get(ticker, np.nan),
        )

        all_results.append(df_subsets)

    if not all_results:
        return pd.DataFrame(columns=["Ticker", "SubsetIndex", "NumFeatures", "Accuracy", "Features"])
    return pd.concat(all_results, ignore_index=True)



def get_all_top_features(
    top_experiment_features: Dict[str, List[str]], 
    forward_results_df: pd.DataFrame,
    dfs: Dict[str, pd.DataFrame],
    ticker_diverse_sets: Dict[str, List[str]],
    k: int = 3
) -> Dict[str, Set[str]]:
    all_top_features = {}
    top_embedding_features = get_best_k_features(forward_results_df, k, False)
    top_wrapper_features, _ = get_embedding_importance_features(dfs, k, add_prints=False)

    for ticker, exp_features in top_experiment_features.items():
        all_top_features[ticker] = set(exp_features).union(top_embedding_features[ticker], top_wrapper_features[ticker])
        print(f"{ticker}: {all_top_features[ticker]}")
    
    return all_top_features


def get_best_accuracy_feature_selection(
    top_features_strategies: Dict[str, Set[str]],
    dfs: Dict[str, pd.DataFrame],
    top_experiment_accuracy: Dict[str, float],
    top_10_embedding_accuracy: Dict[str, float],
    forward_results_df: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    wrapper_accuracy = (
        forward_results_df.groupby("Ticker")["Accuracy"]
        .max()
        .to_dict()
    )
    baseline_by_source = {
        "top_experiment": top_experiment_accuracy,
        "top_10_embedding": top_10_embedding_accuracy,
        "top_10_wrapper": wrapper_accuracy,
        "all": top_experiment_accuracy,
    }

    for top_name, top_features in top_features_strategies.items():
        print(f"{top_name}:")
        source_baseline = baseline_by_source.get(top_name, top_experiment_accuracy)
        subset_results = get_subset_results(top_features, dfs, source_baseline)
        if subset_results.empty:
            print(f"{top_name}: no valid subset results\n")
            continue
        best_df = (
            subset_results
            .sort_values("Accuracy", ascending=False)
            .groupby("Ticker", as_index=False)
            .first()
            .rename(columns={
                "Accuracy": "BestAccuracy"
            })
        )
        best_subsets_df =  best_df[["Ticker", "BestAccuracy", "SubsetIndex", "Features"]]
        print(f"Best found: \n{best_subsets_df.to_string(index=False)}\n") 

        for _, row in best_subsets_df.iterrows():
            rows.append({
                "Source": top_name,
                "Ticker": row["Ticker"],
                "Accuracy": row["BestAccuracy"],
            })

    # --- ORIGINAL experiment baseline ---
    for ticker, acc in top_experiment_accuracy.items():
        rows.append({
            "Source": "original_experiment",
            "Ticker": ticker,
            "Accuracy": acc,
        })
    
    # --- ORIGINAL wrapper baseline ---
    for ticker, acc in wrapper_accuracy.items():
        rows.append({
            "Source": "original_wrapper",
            "Ticker": ticker,
            "Accuracy": acc,
        })
    
    # --- ORIGINAL embedding baseline ---
    for ticker, acc in top_10_embedding_accuracy.items():
        rows.append({
            "Source": "original_embedding",
            "Ticker": ticker,
            "Accuracy": acc,
        })

    return pd.DataFrame(rows)


def plot_accuracy_by_strategy(df: pd.DataFrame):
    plt.figure(figsize=(12, 7))

    sources = [
        "original_experiment",
        "top_experiment",
        "original_embedding",
        "top_10_embedding",
        "original_wrapper",
        "top_10_wrapper",
        "all",
    ]
    x_pos = {s: i for i, s in enumerate(sources)}
    jitter = 0.08

    # Keep only sources we know (avoids KeyError / weird ordering)
    dfp = df[df["Source"].isin(sources)].copy()

    for ticker in dfp["Ticker"].unique():
        sub = dfp[dfp["Ticker"] == ticker].copy()

        # --- create stable jittered x per ROW ---
        sub["_x"] = sub["Source"].map(x_pos) + np.random.uniform(-jitter, jitter, size=len(sub))

        color = COMPANY_COLORS.get(ticker, "gray")

        # --- plot all points using sub["_x"] ---
        plt.scatter(
            sub["_x"],
            sub["Accuracy"],
            color=color,
            label=ticker,
            s=120,
            alpha=0.8,
        )

        # --- circle the BEST point using the SAME jittered x ---
        best_i = sub["Accuracy"].idxmax()
        best_x = sub.loc[best_i, "_x"]
        best_y = sub.loc[best_i, "Accuracy"]

        plt.scatter(
            [best_x],
            [best_y],
            s=220,
            facecolors="none",
            edgecolors="black",
            linewidths=2.0,
            zorder=10,
        )

    plt.xticks(
        ticks=list(x_pos.values()),
        labels=list(x_pos.keys()),
        fontsize=12,
        rotation=20,
    )

    plt.xlabel("Strategy", fontsize=14, weight="bold")
    plt.ylabel("Best Accuracy", fontsize=14, weight="bold")
    plt.title("Best Accuracy by Feature-Selection Strategy", fontsize=16, weight="bold")

    # Deduplicate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(
        by_label.values(),
        by_label.keys(),
        title="Ticker",
        fontsize=11,
        title_fontsize=12,
        loc="upper left",
    )

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    
    plt.show()
    plt.close()


def forward_feature_selection_per_fold(
    df: pd.DataFrame,
    target_col: str,
    model,
    ticker: str,
    max_features: int | None = None,
    n_splits: int = SPLITS
) -> pd.DataFrame:

    rows = []
    selected_features = []

    candidate_features = _numeric_feature_columns(df, target_col)

    max_features = max_features or len(candidate_features)

    for step in range(1, max_features + 1):
        print(f"[{ticker}] Forward step {step}")

        best_feat = None
        best_mean_score = -np.inf
        best_fold_scores = None

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

            mean_score = res["Accuracy"].mean()

            if mean_score > best_mean_score:
                best_mean_score = mean_score
                best_feat = feat
                best_fold_scores = res[["Fold", "Accuracy"]]

        # update state
        selected_features.append(best_feat)
        candidate_features.remove(best_feat)

        # store per-fold results
        for _, row in best_fold_scores.iterrows():
            rows.append({
                "Ticker": ticker,
                "Step": step,
                "NumFeatures": len(selected_features),
                "Fold": int(row["Fold"]),
                "Accuracy": row["Accuracy"],
                "AddedFeature": best_feat,
            })

        print(f"[{ticker}] Added: {best_feat} | Mean Accuracy: {best_mean_score:.4f}")

    return pd.DataFrame(rows)

def run_forward_selection_per_fold(
    dfs: Dict[str, pd.DataFrame],
    n_splits: int = SPLITS
) -> pd.DataFrame:

    all_results = []

    for ticker, df in dfs.items():
        print(f"\n==============================")
        print(f"Forward selection (per fold) for {ticker}")
        print(f"==============================")

        df = _ensure_binary_target(df, "TargetBinary").dropna()

        hist_df = forward_feature_selection_per_fold(
            df=df,
            target_col="TargetBinary",
            model=LogisticRegression(max_iter=2000, random_state=42),
            ticker=ticker,
            n_splits=n_splits
        )

        all_results.append(hist_df)

    return pd.concat(all_results, ignore_index=True)

def get_fold_time_ranges(
    df: pd.DataFrame,
    n_splits: int
) -> pd.DataFrame:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
        val_dates = df.index[val_idx]

        rows.append({
            "Fold": fold,
            "ValStart": val_dates.min(),
            "ValEnd": val_dates.max(),
        })

    return pd.DataFrame(rows)

def sequential_colors_strong(base_color: str, n: int):
    """
    Generate n clearly distinct sequential colors
    from a single base color using HLS space.
    """
    r, g, b = mcolors.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    colors = []
    for i in range(n):
        # lightness: from light to dark
        li = 0.75 - 0.45 * (i / max(n - 1, 1))
        # saturation: slightly increasing
        si = min(1.0, s + 0.2 * (i / max(n - 1, 1)))

        ri, gi, bi = colorsys.hls_to_rgb(h, li, si)
        colors.append((ri, gi, bi))

    return colors

def plot_forward_selection_per_fold(
    fs_fold_df: pd.DataFrame,
    dfs: dict,
    n_splits: int = SPLITS,
):
    for ticker in fs_fold_df["Ticker"].unique():
        sub = fs_fold_df[fs_fold_df["Ticker"] == ticker]

        # rebuild df exactly as in training
        df = _ensure_binary_target(dfs[ticker], "TargetBinary").dropna()

        fold_ranges = get_fold_time_ranges(df, n_splits)

        # --- company color palette ---
        base_color = COMPANY_COLORS.get(ticker, "#333333")
        fold_colors = sequential_colors_strong(base_color, n_splits)

        plt.figure(figsize=(8, 5))

        for fold in sorted(sub["Fold"].unique()):
            fold_df = sub[sub["Fold"] == fold]
            fr = fold_ranges[fold_ranges["Fold"] == fold].iloc[0]

            label = (
                f"Fold {fold + 1}: "
                f"{fr.ValStart.date()} → {fr.ValEnd.date()}"
            )

            plt.plot(
                fold_df["NumFeatures"],
                fold_df["Accuracy"],
                marker="o",
                linewidth=2,
                color=fold_colors[fold],
                label=label
            )

        plt.xlabel("Number of Features")
        plt.ylabel("Accuracy")
        plt.title(f"{ticker} – Forward Selection per Fold")
        plt.legend(fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        plt.show()
        plt.close()


_EPISODE_SHORT_MAP = {
    "COVID shock / panic repricing": "COVID",
    "Reopening + liquidity-driven bull phase": "Reopen",
    "Inflation shock + aggressive rate hikes": "Inflation/Rates",
    "AI-led mega-cap rally with selective breadth": "AI-led",
    "Late-cycle uncertainty / mixed macro signals": "Mixed",
}


def _macro_episode(date_like) -> str:
    ts = pd.Timestamp(date_like)
    if ts <= pd.Timestamp("2020-04-30"):
        return "COVID shock / panic repricing"
    if ts <= pd.Timestamp("2021-12-31"):
        return "Reopening + liquidity-driven bull phase"
    if ts <= pd.Timestamp("2022-12-31"):
        return "Inflation shock + aggressive rate hikes"
    if ts <= pd.Timestamp("2024-12-31"):
        return "AI-led mega-cap rally with selective breadth"
    return "Late-cycle uncertainty / mixed macro signals"


def _regime_label(bull_share: float) -> str:
    if pd.isna(bull_share):
        return "Unavailable"
    if bull_share >= 0.60:
        return "Bull-dominant"
    if bull_share <= 0.40:
        return "Bear-dominant"
    return "Mixed/Transition"


def _scalar_stat(slice_df: pd.DataFrame, col: str, stat: str = "mean") -> float:
    if col not in slice_df.columns:
        return np.nan

    obj = slice_df[col]
    if isinstance(obj, pd.DataFrame):
        ser = pd.to_numeric(obj.stack(dropna=False), errors="coerce")
    else:
        ser = pd.to_numeric(obj, errors="coerce")

    if stat == "mean":
        value = ser.mean()
    elif stat == "std":
        value = ser.std()
    else:
        raise ValueError(f"Unsupported stat: {stat}")

    return float(value) if pd.notna(value) else np.nan


def _prepare_binary_frame_for_fold_context(
    df: pd.DataFrame,
    feature_candidates: List[str],
) -> tuple[pd.DataFrame, List[str], str]:
    prepared = canonicalize_feature_columns(df.copy())
    prepared = preprocess_day_feature(preprocess_month_feature(prepared))
    if not isinstance(prepared.index, pd.DatetimeIndex):
        prepared.index = pd.to_datetime(prepared.index)

    prepared, target_col = experiment_create_target_variable(
        prepared,
        target_type="binary",
        horizon=1,
    )
    # Protect against duplicated names after feature joins.
    prepared = prepared.loc[:, ~prepared.columns.duplicated(keep="first")]

    feature_cols = [c for c in feature_candidates if c in prepared.columns]
    if not feature_cols:
        fallback = [
            "Close",
            "Log_Return",
            "MA20",
            "MA50",
            "MA200",
            "RSI",
            "VIX_Index",
            "Nasdaq_100",
        ]
        feature_cols = [c for c in fallback if c in prepared.columns]

    required = list(feature_cols) + [target_col]
    if "Regime_Bull" in prepared.columns:
        required.append("Regime_Bull")
    if "Log_Return" in prepared.columns:
        required.append("Log_Return")
    required = list(dict.fromkeys(required))

    prepared = prepared[required].dropna().sort_index()
    return prepared, feature_cols, target_col


def run_fold_regime_context_analysis(
    dfs: Dict[str, pd.DataFrame],
    *,
    candidate_features_map: Optional[Dict[str, List[str]]] = None,
    n_splits: int = DEF_SPLITS,
    model: Optional[LogisticRegression] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build per-fold diagnostic tables that link validation accuracy to regime context.

    Returns:
        fold_context_df:
            Detailed per-ticker/per-fold table.
        fold_context_summary_df:
            Per-ticker summary with mean and dispersion statistics.
    """
    candidate_features_map = candidate_features_map or {}
    model = model or LogisticRegression(max_iter=2000, random_state=42)

    rows: List[Dict] = []

    for ticker, df_ticker in dfs.items():
        candidate_features = candidate_features_map.get(ticker, [])
        prepared_df, use_features, target_col = _prepare_binary_frame_for_fold_context(
            df_ticker,
            candidate_features,
        )

        if len(prepared_df) <= n_splits or not use_features:
            continue

        X = prepared_df[use_features]
        y = prepared_df[target_col].astype(int)
        tscv = TimeSeriesSplit(n_splits=n_splits)

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            model_fold = clone(model)
            model_fold.fit(X_train_scaled, y_train)
            preds = model_fold.predict(X_val_scaled)
            accuracy = float((preds == y_val.values).mean())

            train_slice = prepared_df.iloc[train_idx]
            val_slice = prepared_df.iloc[val_idx]

            train_bull_share = _scalar_stat(train_slice, "Regime_Bull", stat="mean")
            val_bull_share = _scalar_stat(val_slice, "Regime_Bull", stat="mean")
            regime_shift_abs = (
                abs(train_bull_share - val_bull_share)
                if pd.notna(train_bull_share) and pd.notna(val_bull_share)
                else np.nan
            )
            val_logret_std = _scalar_stat(val_slice, "Log_Return", stat="std")

            macro_episode = _macro_episode(val_slice.index.min())
            rows.append(
                {
                    "Ticker": ticker,
                    "Fold": fold_idx,
                    "Accuracy": round(accuracy, 4),
                    "TrainStart": train_slice.index.min().date(),
                    "TrainEnd": train_slice.index.max().date(),
                    "ValStart": val_slice.index.min().date(),
                    "ValEnd": val_slice.index.max().date(),
                    "MacroEpisode": macro_episode,
                    "EpisodeShort": _EPISODE_SHORT_MAP.get(macro_episode, "Other"),
                    "TrainBullShare": round(train_bull_share, 3) if pd.notna(train_bull_share) else np.nan,
                    "ValBullShare": round(val_bull_share, 3) if pd.notna(val_bull_share) else np.nan,
                    "ValRegime": _regime_label(val_bull_share),
                    "RegimeShiftAbs": round(regime_shift_abs, 3) if pd.notna(regime_shift_abs) else np.nan,
                    "ValLogRetStd": round(val_logret_std, 5) if pd.notna(val_logret_std) else np.nan,
                    "NumFeatures": len(use_features),
                }
            )

    detailed_cols = [
        "Ticker",
        "Fold",
        "Accuracy",
        "TrainStart",
        "TrainEnd",
        "ValStart",
        "ValEnd",
        "MacroEpisode",
        "EpisodeShort",
        "TrainBullShare",
        "ValBullShare",
        "ValRegime",
        "RegimeShiftAbs",
        "ValLogRetStd",
        "NumFeatures",
    ]
    summary_cols = [
        "Ticker",
        "MeanAccuracy",
        "StdAccuracy",
        "MeanRegimeShift",
        "MeanValVolatility",
    ]

    if not rows:
        return (
            pd.DataFrame(columns=detailed_cols),
            pd.DataFrame(columns=summary_cols),
        )

    fold_context_df = pd.DataFrame(rows).sort_values(["Ticker", "Fold"]).reset_index(drop=True)
    fold_context_summary_df = (
        fold_context_df.groupby("Ticker", as_index=False)
        .agg(
            MeanAccuracy=("Accuracy", "mean"),
            StdAccuracy=("Accuracy", "std"),
            MeanRegimeShift=("RegimeShiftAbs", "mean"),
            MeanValVolatility=("ValLogRetStd", "mean"),
        )
        .sort_values("Ticker")
        .reset_index(drop=True)
    )

    return fold_context_df, fold_context_summary_df


def build_fold_regime_context_compact_table(fold_context_df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact reader-facing subset of the fold-context table."""
    compact_cols = [
        "Ticker",
        "Fold",
        "Accuracy",
        "ValStart",
        "ValEnd",
        "EpisodeShort",
        "ValRegime",
        "RegimeShiftAbs",
    ]
    if fold_context_df.empty:
        return pd.DataFrame(columns=compact_cols)
    return fold_context_df[compact_cols].copy()


def display_fold_regime_context_tables(
    fold_context_df: pd.DataFrame,
    fold_context_summary_df: pd.DataFrame,
) -> None:
    """Display compact fold table and per-ticker summary in notebook-friendly format."""
    compact_df = build_fold_regime_context_compact_table(fold_context_df)

    print("Fold-level diagnostics (5-fold walk-forward):")
    try:
        from IPython.display import display  # type: ignore

        display(compact_df)
        print("Per-ticker aggregate diagnostics:")
        display(fold_context_summary_df.round(4))
    except ImportError:
        print(compact_df.to_string(index=False))
        print("Per-ticker aggregate diagnostics:")
        print(fold_context_summary_df.round(4).to_string(index=False))


def plot_fold_regime_context(fold_context_df: pd.DataFrame) -> None:
    """
    Plot fold diagnostics:
    1) Heatmap of validation accuracy by ticker x fold.
    2) Unified line chart for all tickers with project-standard colors.
    """
    if fold_context_df.empty:
        print("No fold-context rows were produced; skipping plots.")
        return

    acc_matrix = (
        fold_context_df.pivot(index="Ticker", columns="Fold", values="Accuracy").sort_index()
    )

    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(acc_matrix.values, cmap="YlGnBu", aspect="auto")

    for i in range(acc_matrix.shape[0]):
        for j in range(acc_matrix.shape[1]):
            value = acc_matrix.values[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=9)

    ax.set_xticks(range(acc_matrix.shape[1]))
    ax.set_xticklabels([f"Fold {fold}" for fold in acc_matrix.columns])
    ax.set_yticks(range(acc_matrix.shape[0]))
    ax.set_yticklabels(acc_matrix.index)
    ax.set_title("Validation Accuracy by Ticker and Fold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 4.8))

    y_min = max(0.0, fold_context_df["Accuracy"].min() - 0.03)
    y_max = min(1.0, fold_context_df["Accuracy"].max() + 0.03)
    xticks = sorted(fold_context_df["Fold"].unique())

    for ticker in sorted(fold_context_df["Ticker"].unique()):
        sub = fold_context_df[fold_context_df["Ticker"] == ticker].sort_values("Fold")
        color = COMPANY_COLORS.get(ticker, "#333333")
        ax.plot(
            sub["Fold"],
            sub["Accuracy"],
            marker="o",
            linewidth=2,
            color=color,
            label=ticker,
        )

    ax.set_title("Validation Accuracy Across Folds by Ticker")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Validation Accuracy")
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(xticks)
    ax.grid(alpha=0.3)
    ax.legend(title="Ticker")
    plt.tight_layout()
    plt.show()
    plt.close()
