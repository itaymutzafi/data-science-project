"""Plotting utilities for consistent academic visualization.

This module defines the global style settings for matplotlib/seaborn
to ensure all figures in the report have a uniform, professional appearance.
"""

import matplotlib.pyplot as plt
import seaborn as sns

def set_style():
    """Sets the global plotting style for the project."""
    # Use a clean, academic style
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # Custom overrides for better readability in reports
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 1.5,
        "grid.alpha": 0.3,
        "figure.dpi": 150  # High resolution for export
    })

def save_fig(fig, filename: str, folder: str = "results"):
    """Saves a figure to the results folder with consistent settings."""
    # Implementation placeholder for future use
    # fig.savefig(f"{folder}/{filename}", bbox_inches='tight')
    pass
