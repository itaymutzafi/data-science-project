from typing import Dict
from cycler import cycler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (COMPANY_COLORS, TICKER_TO_COMPANY_MAP, DEF_SPLITS)
from src.utils import statistic_tests as st

_STYLE_APPLIED = False
ACADEMIC_PALETTE = ["#1B263B", "#0A9396", "#EE9B00", "#CA6702", "#9B2226"]


def set_style() -> None:
    """Apply global plot styling once per session."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return

    sns.set_theme(style="whitegrid", context="paper", palette=ACADEMIC_PALETTE)
    plt.rcParams.update(
        {
            "figure.figsize": (12, 6),
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "lines.linewidth": 2.0,
            "grid.alpha": 0.25,
            "figure.dpi": 150,
            "axes.facecolor": "#fbfbfd",
            "axes.edgecolor": "#cccccc",
            "axes.prop_cycle": cycler(color=ACADEMIC_PALETTE),
        }
    )
    _STYLE_APPLIED = True


def apply_academic_style(ax: plt.Axes, title: str | None = None) -> None:
    """Format axes with consistent academic styling."""
    if title:
        ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.set_facecolor("#fbfbfd")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def ensure_dataframe(data: pd.DataFrame | Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Helper: Normalize input to a single DataFrame with a Ticker column."""
    if isinstance(data, dict):
        processed_frames = []
        for ticker, df in data.items():
            temp = df.copy()
            # Ensure datetime index and remove timezone for consistent plotting/merging
            if not isinstance(temp.index, pd.DatetimeIndex):
                temp.index = pd.to_datetime(temp.index)
            if temp.index.tz is not None:
                temp.index = temp.index.tz_localize(None)

            temp["Ticker"] = ticker
            processed_frames.append(temp)
        return pd.concat(processed_frames, axis=0) if processed_frames else pd.DataFrame()
    return data
