# Cell ID: setup
# --- Setup & Configuration ---
import sys
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Internal Modules (src/)
from src.data.loader import fetch_sample_data
from src.evaluation.analysis import check_stationarity, run_baseline_analysis
from src.models.baselines import NaiveBaseline, RandomBaseline, CAPMBaseline
from src.evaluation.metrics import evaluate_regression, print_eval
from src.evaluation.plots import set_style, plot_price_vs_returns, plot_walk_forward_validation, plot_autocorrelation
from src.features.preprocessing import LogReturnTransformer

# Apply Academic Plotting Style
# %matplotlib inline
set_style()

print(f"Environment: Python {platform.python_version()}")
print("Pipeline modules loaded successfully.")

# Cell ID: adf_test
# 1. Data Ingestion
df_research = fetch_sample_data("AAPL", period="2y")

# 2. Feature Engineering (Log Returns)
log_transformer = LogReturnTransformer()
df_research = log_transformer.transform(df_research)
df_research.dropna(inplace=True)

# 3. Visualization
plot_price_vs_returns(df_research, 'Log_Returns')

# 4. Hypothesis Testing (ADF)
check_stationarity(df_research['Close'], "Raw Close Price")
check_stationarity(df_research['Log_Returns'], "Log Returns")

# Cell ID: sec_2_3_code
# Calculate Log Returns if not already present (assuming df_research has 'Close')
if 'Log_Return' not in df_research.columns:
    df_research['Log_Return'] = np.log(df_research['Close'] / df_research['Close'].shift(1))

plot_autocorrelation(df_research['Log_Return'].dropna())

# Cell ID: validation_viz
# Visualization of Walk-Forward Validation (Expanding Window)
# from src.evaluation.plots import plot_walk_forward_validation

plot_walk_forward_validation()

# Cell ID: baseline_code
# Prepare Train/Test Split
train_size = int(len(df_aapl) * 0.8)
train_data = df_aapl.iloc[:train_size]
test_data = df_aapl.iloc[train_size:]

# Run Baseline Analysis
run_baseline_analysis(
    y_train=train_data['Log_Return'].dropna(),
    y_test=test_data['Log_Return'].dropna(),
    X_test=test_data
)

