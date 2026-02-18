import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from src.config import MONTHNAMES


def test_seasonality(df: pd.DataFrame, return_col: str,  time_precision: str, alpha: float = 0.05) -> dict:
    """
    Generic seasonality significance testing using datetime-based categories
    """
    results = {}
    data = df.copy()

    choose_base = sorted(data[time_precision].dropna().unique())
    baseline = choose_base[0]

    # Prepare grouped returns
    grouped_returns = [
        group[return_col].dropna().values
        for _, group in data.groupby(time_precision)
    ]

    # global tests
    f_stat, p_anova = stats.f_oneway(*grouped_returns)
    kw_stat, p_kw = stats.kruskal(*grouped_returns)

    # OLS regression
    model = smf.ols(
        f"{return_col} ~ C({time_precision})",
        data=data
    ).fit()

    coef_table = (
        pd.DataFrame({
            "Coefficient": model.params,
            "p_value": model.pvalues
        })
        .drop("Intercept")
    )

    coef_table["Significant"] = coef_table["p_value"] < alpha
    coef_table["Direction_vs_baseline"] = coef_table["Coefficient"].apply(
        lambda x: "Higher" if x > 0 else "Lower"
    )

    # design the result
    results["baseline_day"] = baseline
    results["global_tests"] = {
        "anova_F": f_stat,
        "anova_p": p_anova,
        "kruskal_H": kw_stat,
        "kruskal_p": p_kw,
        "seasonality_detected": (p_anova < alpha) and (p_kw < alpha)
    }

    results["day_effects"] = coef_table
    return results


def plot_all_tickers_seasonality(all_results, time_label, ticker_colors, alpha=0.05, feature_name="Return"):
    """Grouped bar chart of OLS coefficients for all tickers, with significance hatching."""
    tickers = list(all_results.keys())
    n_tickers = len(tickers)

    first_result = next(iter(all_results.values()))
    effects = first_result["day_effects"]
    raw_labels = [idx.split(".")[-1].rstrip("]") for idx in effects.index]
    if time_label == "Month":
        period_labels = [MONTHNAMES[int(lbl) - 1] for lbl in raw_labels]
        baseline = MONTHNAMES[int(first_result["baseline_day"]) - 1]
    else:
        period_labels = raw_labels
        baseline = first_result["baseline_day"]
    n_periods = len(period_labels)

    width = 0.8 / n_tickers
    x = np.arange(n_periods)

    fig, ax = plt.subplots(figsize=(14, 5))

    for i, ticker in enumerate(tickers):
        res = all_results[ticker]
        eff = res["day_effects"]
        coefs = eff["Coefficient"].values
        sigs = eff["Significant"].values

        color = ticker_colors.get(ticker, f"C{i}")
        offset = x + (i - (n_tickers - 1) / 2) * width
        bars = ax.bar(
            offset, coefs, width,
            color=color, edgecolor="black", linewidth=0.5, alpha=0.8,
            label=ticker,
        )
        for bar, sig in zip(bars, sigs):
            if not sig:
                bar.set_hatch("//")
                bar.set_alpha(0.35)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(period_labels, rotation=45, ha="right")
    ax.set_xlabel(time_label)
    ax.set_ylabel("Coefficient vs Baseline")
    ax.set_title(f"{feature_name} Seasonality by {time_label} (baseline = {baseline})")

    handles, labels = ax.get_legend_handles_labels()
    hatched_patch = Patch(facecolor="white", edgecolor="gray", hatch="//", label="Not significant")
    handles.append(hatched_patch)
    ax.legend(handles=handles, loc="best", fontsize=8)

    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

def seasonality_summary_table(all_results_day, all_results_month, alpha=0.05):
    """Single summary DataFrame with global test results for all tickers and both time dimensions."""
    rows = []
    for time_label, all_results in [("Day", all_results_day), ("Month", all_results_month)]:
        for ticker, res in all_results.items():
            g = res["global_tests"]
            n_sig = int(res["day_effects"]["Significant"].sum())
            n_periods = len(res["day_effects"])
            rows.append({
                "Ticker": ticker,
                "Period": time_label,
                "ANOVA F": round(g["anova_F"], 4),
                "ANOVA p": round(g["anova_p"], 4),
                "Kruskal H": round(g["kruskal_H"], 4),
                "Kruskal p": round(g["kruskal_p"], 4),
                "Significant Periods": f"{n_sig}/{n_periods}",
                "Seasonality Detected": "Yes" if g["seasonality_detected"] else "No",
            })

    summary = pd.DataFrame(rows)
    display(
        summary.style
        .map(
            lambda v: "color: green; font-weight: bold" if v == "Yes"
            else ("color: red" if v == "No" else ""),
            subset=["Seasonality Detected"],
        )
        .format({
            "ANOVA F": "{:.4f}",
            "ANOVA p": "{:.4g}",
            "Kruskal H": "{:.4f}",
            "Kruskal p": "{:.4g}",
        })
    )