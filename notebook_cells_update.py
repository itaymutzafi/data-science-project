"""
Notebook Draft - Feature Engineering Update
Paste the following cells into your Jupyter Notebook to use the new standardized feature pipeline.
"""

# Cell 1: Setup & Feature Generation
# ------------------------------------------------------------------------------
from src.features.moving_average import add_ma_features, ma_plot
from src.features.volatility import add_volatility_features, volatility_comparison_plot
from src.features.external_market import add_macro_features, fetch_auxiliary_data
from src.config import FEATURE_WINDOWS

# 1. Fetch External Data (VIX, Treasury, etc.)
aux_data = fetch_auxiliary_data()

# 2. Add Standardized Features
# Adds MA20, MA50, MA200 (defined in config)
add_ma_features(dfs)

# Adds Vol20 and caps outliers > 3 sigma
add_volatility_features(dfs)

# Adds VIX, VIX_MA, VIX_Gap
add_macro_features(dfs, aux_data)

print(f"Feature Engineering Complete.")
print(f"Standard Windows: {FEATURE_WINDOWS}")
# ------------------------------------------------------------------------------


# Cell 2: Visualization (Verification)
# ------------------------------------------------------------------------------
print("\n--- Moving Averages (Trend) ---")
ma_plot(dfs)

print("\n--- Market Volatility (Risk) ---")
volatility_comparison_plot(dfs)
# ------------------------------------------------------------------------------
