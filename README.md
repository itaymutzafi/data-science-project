# Stock Market Prediction Pipeline (AAPL)

**Workshop in Data Science — Team 003**  
_(Itay, Moran, Shaked)_

This repository contains an end-to-end machine learning pipeline for predicting short-term stock movements (Log Returns) of Apple Inc. (AAPL). The project is designed with a "Notebook as a Report" philosophy, where the main narrative lives in `Final_Project_Report.ipynb` while the heavy lifting is handled by a structured Python package (`src/`).

---

## 📂 Repository Structure

The project follows a domain-driven hybrid architecture:

```text
.
├── Final_Project_Report.ipynb  # Main project report (The "Story")
├── README.md                   # Project documentation
├── environment.yml             # Conda environment specification
├── requirements.txt            # Pip requirements
├── data/                       # Raw and processed datasets
├── literature/                 # Project literature and materials
├── src/                        # Source code (The "Engine")
│   ├── config.py               # Configuration settings
│   ├── data/                   # Data ingestion and loading
│   │   └── loader.py
│   ├── features/               # Feature engineering
│   │   ├── indicators.py       # Technical indicators (RSI, MA)
│   │   └── preprocessing.py    # Transformers (Log Returns)
│   ├── models/                 # Model definitions
│   │   ├── baselines.py        # Baseline models (Naive, CAPM)
│   │   └── training.py         # Training logic
│   ├── evaluation/             # Analysis tools
│   │   ├── analysis.py         # Statistical tests
│   │   ├── metrics.py          # Performance metrics
│   │   └── plots.py            # Visualization utilities
│   └── utils/                  # Utility functions
└── tests/                      # Unit tests
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Lab or Notebook

### Installation

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Report:**
    Open `Final_Project_Report.ipynb` in Jupyter Lab/Notebook and execute all cells.
    The notebook is self-contained and will fetch data automatically using the `src` package.

3.  **Run Tests:**
    To verify the integrity of the pipeline, run the test suite:
    ```bash
    pytest tests/
    ```

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
