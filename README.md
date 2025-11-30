# Stock Market Prediction Pipeline (AAPL)

**Workshop in Data Science — Team 003**  
_(Itay, Moran, Shaked)_

This repository contains an end-to-end machine learning pipeline for predicting short-term stock movements (Log Returns) of Apple Inc. (AAPL). The project is designed with a "Notebook as a Report" philosophy, where the main narrative lives in Jupyter notebooks while the heavy lifting is handled by a structured Python package (`src/`).

---

## 📂 Repository Structure

The project follows a domain-driven hybrid architecture:

```text
├── data/               # Raw and processed datasets
├── notebooks/          # Exploratory and development notebooks
├── src/                # Source code (The "Engine")
│   ├── data/           # Data ingestion and loading (w/ Caching)
│   ├── features/       # Feature engineering (Transformers)
│   ├── models/         # Model definitions (Baselines, CAPM, ML, DL)
│   ├── evaluation/     # Metrics, plotting (ACF), and analysis tools
│   └── utils/          # Configuration and helpers
├── tests/              # Unit tests for the pipeline
├── Final_Project_Report.ipynb  # Main project report (The "Story")
└── requirements.txt    # Project dependencies
```

---

## 🚀 Getting Started

### Prerequisites
# Stock Market Prediction Pipeline

## Project Structure

```
.
├── Final_Project_Report.ipynb  # Main analysis notebook (The "Report")
├── README.md                   # Project documentation
├── environment.yml             # Conda environment specification
├── requirements.txt            # Pip requirements
├── data/
│   └── raw/                    # Cached raw data (parquet)
├── literature/                 # Project literature and materials
├── src/                        # Source code package
│   ├── __init__.py
│   ├── config.py               # Configuration settings
│   ├── data/
│   │   └── __init__.py         # Data loader exports
│   ├── features/
│   │   ├── __init__.py
│   │   ├── indicators.py       # Technical indicators (RSI, MA)
│   │   └── preprocessing.py    # Transformers (Log Returns)
│   ├── models/
│   │   ├── __init__.py         # Model exports
│   │   ├── baselines.py        # Baseline models (Naive, CAPM)
│   │   └── training.py         # Training logic
│   ├── evaluation/
│   │   ├── __init__.py         # Evaluation exports
│   │   ├── analysis.py         # Statistical tests & analysis
│   │   ├── metrics.py          # Performance metrics
│   │   └── plots.py            # Visualization utilities
│   └── utils/                  # Utility functions
└── tests/                      # Unit tests
    ├── conftest.py
    ├── test_evaluation.py
    ├── test_features.py
    └── test_models.py
```

## Module Roles

### `src.data`
Handles data ingestion and storage.
- **`loader.py`**: Fetches data from `yfinance` and implements **local caching** in `data/raw/` to prevent redundant API calls.

### `src.features`
Responsible for feature engineering and preprocessing.
- **`preprocessing.py`**: Contains `LogReturnTransformer` to convert non-stationary prices to stationary log-returns.
- **`indicators.py`**: Computes technical indicators like RSI and Moving Averages.

### `src.models`
Contains predictive models and baselines.
- **`baselines.py`**: Implements benchmark models including:
    - `NaiveBaseline`: Zero-return assumption.
    - `RandomBaseline`: Monte Carlo simulation.
    - `MarketBenchmark`: Buy & Hold strategy.
    - **`CAPMBaseline`**: Capital Asset Pricing Model ($R_i = R_f + \beta(R_m - R_f)$).

### `src.evaluation`
Utilities for validating and visualizing results.
- **`analysis.py`**: Runs stationarity tests (ADF) and baseline comparisons.
- **`metrics.py`**: Calculates MSE, RMSE, MAE, R2, Sharpe Ratio, and Directional Accuracy.
- **`plots.py`**: Generates academic-style figures, including **Autocorrelation (ACF)** and Walk-Forward Validation plots.

## Setup & Usage

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Report:**
    Open `Final_Project_Report.ipynb` in Jupyter Lab/Notebook and execute all cells.
    The notebook is self-contained and will fetch data automatically.

3.  **Run Tests:**
    ```bash
    pytest tests/
    ```

2.  **Open the Report:**
    Start Jupyter Lab/Notebook and open `Final_Project_Report.ipynb`.  
    **Important:** Ensure you select the kernel **`Python (DS Project .venv)`**.

---

## 🛠️ Methodology

### 1. Problem Formulation

We predict **Logarithmic Returns** ($Y_t$) instead of raw prices ($P_t$) to ensure stationarity, a critical assumption for many ML models.

### 2. Architecture

We use a **Hybrid Architecture**:

- **Object-Oriented (Classes):** For stateful components like Transformers (e.g., `LogReturnTransformer`) and Models. This allows integration with Scikit-Learn pipelines.
- **Functional:** For stateless utilities like data loading and metric calculation.

### 3. Evaluation

We optimize for **MSE** (Statistical fit) but evaluate success using **Sharpe Ratio** (Financial risk-adjusted return) and **Directional Accuracy** (Trading utility).

---

## 🧪 Testing

The project includes a comprehensive test suite in `tests/` covering:

- **Features:** Correctness of transformers (shapes, NaN handling).
- **Models:** Interface compliance and baseline logic.
- **Evaluation:** Mathematical correctness of metrics.
