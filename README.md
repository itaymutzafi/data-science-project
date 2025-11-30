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

- Python 3.12+
- Git

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <repo_url>
    cd data-science-project
    ```

2.  **Set up the environment (Recommended):**

    ```bash
    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3.  **Register the Jupyter Kernel:**
    ```bash
    python -m ipykernel install --user --name=ds-project-venv --display-name "Python (DS Project .venv)"
    ```

### Running the Project

1.  **Run Tests:** Verify that the environment and logic are correct.

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
