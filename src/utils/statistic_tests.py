import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

def test_seasonality(
    df: pd.DataFrame,
    return_col: str, 
    Day_or_Month: str,
    alpha: float = 0.05,
) -> dict:
    """
    Generic seasonality significance testing using datetime-based categories.
    """

    results = {}
    data = df.copy()

    choose_base = sorted(data[Day_or_Month].dropna().unique())
    baseline = choose_base[0]


    # Prepare grouped returns
    grouped_returns = [
        group[return_col].dropna().values
        for _, group in data.groupby(Day_or_Month)
    ]

    # global tests
    f_stat, p_anova = stats.f_oneway(*grouped_returns)
    kw_stat, p_kw = stats.kruskal(*grouped_returns)

    # OLS regression
    model = smf.ols(
        f"{return_col} ~ C({Day_or_Month})",
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

from IPython.display import display

def display_seasonality_results(results, alpha=0.05):
    """
    Pretty display of seasonality test results.
    """

    # ===============================
    # 1. Headline summary
    # ===============================
    print("📊 SEASONALITY TEST RESULTS")
    print("=" * 40)

    baseline = results["baseline_day"]
    detected = results["global_tests"]["seasonality_detected"]

    print(f"Baseline period: {baseline}")
    print(
        "Seasonality detected: "
        + ("YES ✅" if detected else "NO ❌")
    )
    print()

    # ===============================
    # 2. Global tests table
    # ===============================
    global_tests_df = pd.DataFrame({
        "Statistic": ["ANOVA F", "ANOVA p-value", "Kruskal H", "Kruskal p-value"],
        "Value": [
            results["global_tests"]["anova_F"],
            results["global_tests"]["anova_p"],
            results["global_tests"]["kruskal_H"],
            results["global_tests"]["kruskal_p"],
        ]
    })

    print("🔎 Global significance tests")
    display(global_tests_df.style.format({"Value": "{:.4g}"}))

    # ===============================
    # 3. Day / Month effects table
    # ===============================
    effects = results["day_effects"].copy()

    effects = effects.reset_index().rename(
        columns={"index": "Period"}
    )

    effects["Coefficient"] = effects["Coefficient"].round(5)
    effects["p_value"] = effects["p_value"].round(4)

    effects["Significant"] = effects["Significant"].map(
        {True: "Yes", False: "No"}
    )

    print(f"📅 Effects relative to baseline ({baseline})")
    display(effects)
