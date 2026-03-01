"""Regression and classification evaluation metrics."""

import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score

CLS_METRICS = {"Accuracy": False, "Precision": False, "Recall": False}
REG_METRICS = {"RMSE": True, "R2": False, "Directional Accuracy": False}


def _safe_information_coefficient(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute Pearson IC safely, avoiding runtime warnings on degenerate folds."""
    aligned = pd.concat([y_true, y_pred], axis=1).dropna()
    if aligned.empty:
        return 0.0

    a = aligned.iloc[:, 0]
    b = aligned.iloc[:, 1]
    if a.nunique() < 2 or b.nunique() < 2:
        return 0.0

    with np.errstate(invalid="ignore", divide="ignore"):
        ic = a.corr(b)
    if pd.isna(ic) or np.isinf(ic):
        return 0.0
    return float(ic)


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sharpe ratio for daily returns."""
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / 252
    mean_excess_return = excess_returns.mean()
    std_excess_return = excess_returns.std()

    if std_excess_return == 0:
        return 0.0

    return np.sqrt(252) * (mean_excess_return / std_excess_return)



def calculate_max_drawdown(returns: pd.Series) -> float:
    """Compute maximum drawdown for a returns series."""
    if returns.empty:
        return 0.0
    cumulative = (1 + returns.fillna(0)).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sortino ratio for daily returns."""
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return np.inf

    downside_std = downside_returns.std()

    if downside_std == 0:
        return np.inf

    return np.sqrt(252) * (excess_returns.mean() / downside_std)


def calculate_calmar_ratio(returns: pd.Series) -> float:
    """Compute Calmar ratio from annualized return and max drawdown."""
    if returns.empty:
        return 0.0

    max_dd = abs(calculate_max_drawdown(returns))
    if max_dd == 0:
        return 0.0

    ann_return = returns.mean() * 252

    return ann_return / max_dd


def calculate_profit_factor(returns: pd.Series) -> float:
    """Compute profit factor from positive and negative return sums."""
    if returns.empty:
        return 0.0

    profits = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())

    if losses == 0:
        return np.inf if profits > 0 else 0.0

    return profits / losses


def evaluate_regression(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str = None,
    n_features: int | None = None,
) -> Dict[str, float]:
    """Compute regression and strategy-oriented evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    if n_features is None:
        adjusted_r2 = np.nan
    else:
        n_obs = len(y_true)
        denom = n_obs - int(n_features) - 1
        if denom <= 0:
            adjusted_r2 = np.nan
        else:
            adjusted_r2 = 1 - ((1 - r2) * (n_obs - 1) / denom)

    correct_direction = np.sign(y_true) == np.sign(y_pred)
    da = np.mean(correct_direction)

    if model_name is not None and model_name in {"NaiveBaseline", "MarketBenchmark", "RandomBaseline"}:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        strategy_sharpe = 0.0
        strategy_sortino = 0.0
        strategy_calmar = 0.0
        profit_factor = 0.0
        max_dd = 0.0
        ic = 0.0
    else:
        y_true_bin = (y_true > 0).astype(int)
        y_pred_bin = (y_pred > 0).astype(int)
        precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

        strategy_returns = np.sign(y_pred) * y_true
        strategy_sharpe = calculate_sharpe_ratio(strategy_returns)
        strategy_sortino = calculate_sortino_ratio(strategy_returns)
        strategy_calmar = calculate_calmar_ratio(strategy_returns)
        profit_factor = calculate_profit_factor(strategy_returns)
        max_dd = calculate_max_drawdown(strategy_returns)

        ic = _safe_information_coefficient(y_true, y_pred)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Adjusted R2": adjusted_r2,
        "Directional Accuracy": da,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Strategy Sharpe": strategy_sharpe,
        "Strategy Sortino": strategy_sortino,
        "Strategy Calmar": strategy_calmar,
        "Profit Factor": profit_factor,
        "Max Drawdown": max_dd,
        "IC": ic
    }


def evaluate_classification(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Compute classification metrics."""
    if y_true.empty or y_pred.empty:
        return {}

    acc = np.mean(y_true == y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1
    }
