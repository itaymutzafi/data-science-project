<!--
Academic Submission README
Workshop in Data Science — Final Project
Team 003
-->

# Stock Return Prediction Under Multi-Source Signals

**Workshop in Data Science (Final Project), Tel Aviv University**  
**Team 003**

This repository is the full submission package for the course final project, including:
1. The final report notebook.
2. A modular Python codebase for data ingestion, feature engineering, modeling, and evaluation.
3. Environment and execution scripts for reproducible reruns.

---

## 1. Submission Scope

The primary academic deliverable is:

- `Final_Project_Report.ipynb`

Supporting source code is organized under `src/` and is designed to be reusable, readable, and reproducible for all report sections.

---

## 2. Executive Summary

The project studies short-horizon return prediction for large-cap technology equities (`AAPL`, `AMZN`, `GOOG`, `MSFT`) by combining:

- Price-based technical indicators.
- Market context signals (macro proxies and peer information).
- News sentiment features derived from FinBERT.

Two prediction tracks are evaluated:

1. **Continuous track**: forward return magnitude.
2. **Discrete track**: forward direction (up/down), treated as the main decision-oriented branch.

All experiments are evaluated under chronological, leakage-safe protocols (walk-forward validation and target alignment controls).

---

## 3. Research Design and Guardrails

### 3.1 Time-Series Integrity

The project enforces time-order correctness by:

- Constructing forward targets with explicit horizon alignment.
- Avoiding random train/validation shuffles for model evaluation.
- Applying per-fold scaling inside walk-forward loops.

### 3.2 Feature Integrity

Feature generation follows explicit engineering blocks (technical, macro, sentiment, interactions) with audit helpers for:

- Missing defined features.
- Unexpected extra columns.
- Consistent naming via canonicalization utilities.

### 3.3 Caching and Reproducibility

External data pulls (prices, auxiliary series, Prophet outputs, SEC filings, sentiment outputs) use a cache-first policy with legacy-path fallback.  
This reduces run variance and runtime while preserving deterministic behavior whenever cached artifacts are available.

---

## 4. Repository Layout

```text
.
├── Final_Project_Report.ipynb
├── README.md
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── scripts/
│   ├── setup_env.sh
│   ├── install_kernel.sh
│   └── reset_env.sh
├── data/
│   ├── cache/
│   ├── processed/
│   └── raw/
├── literature/
└── src/
    ├── config.py
    ├── data/
    ├── evaluation/
    ├── features/
    ├── models/
    └── utils/
```

---

## 5. Environment Setup

### 5.1 Recommended One-Command Setup

```bash
bash scripts/setup_env.sh
```

This script:

1. Creates `.venv` if missing.
2. Installs all dependencies from `requirements.txt`.
3. Installs the project in editable mode.
4. Registers Jupyter kernel `Python (Data Science Project)`.

### 5.2 Manual `venv + pip` Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --user --name data-science-project --display-name "Python (Data Science Project)"
```

### 5.3 Conda Setup

```bash
conda env create -f environment.yml
conda activate ds-project
python -m ipykernel install --user --name data-science-project --display-name "Python (Data Science Project)"
```

---

## 6. Running the Full Submission Notebook

1. Activate the environment.
2. Start Jupyter:

```bash
jupyter lab
```

3. Open `Final_Project_Report.ipynb`.
4. Select kernel `Python (Data Science Project)`.
5. Run cells top-to-bottom.

---

## 7. Reproducibility Controls Used in the Notebook

The notebook exposes explicit runtime controls for compute budget:

### 7.1 Sentiment Inference Depth

`SENTIMENT_DEPTH` presets:

- `low` / `quick` -> 1 headline per day-company.
- `medium` / `balanced` -> 3 headlines.
- `high` -> 5 headlines.
- `deep` / `full` / `all` -> no cap.

### 7.2 Feature-Set Search Budget

- `CONTINUOUS_SEARCH_BUDGET`
- `DISCRETE_SEARCH_BUDGET`

Higher values evaluate a larger fraction of sampled feature sets and increase runtime.

---

## 8. Data and Artifact Policy

- Raw and processed data artifacts are under `data/`.
- Cache directories are used for reproducible reruns and reduced external API dependence.
- Large local artifacts remain ignored by Git according to `.gitignore`.

Expected generated artifacts include:

- Cached price and macro parquet files.
- Sentiment feature cache.
- Optional Prophet and SEC filings cache.
- Experiment result tables generated during notebook execution.

---

## 9. Quality Gates Before Submission

Run these checks from repository root:

```bash
ruff check src
python -m compileall src
pytest -q
```

Notes:

- `ruff check src` should pass cleanly.
- `compileall` verifies import/parse integrity.
- If `pytest` reports no tests collected, this indicates no formal test suite is currently defined (known limitation).

---

## 10. Known Limitations

1. The project currently relies primarily on notebook-level validation and diagnostics rather than a full automated unit-test suite.
2. Runtime for sentiment and model-zoo sections depends on chosen compute budgets and local hardware.
3. External APIs (market/news sources) may introduce data availability differences across reruns when cache is refreshed.

---

## 11. Team and Course Context

- Course: Workshop in Data Science (Tel Aviv University)
- Submission Type: Final academic project
- Team: 003

---

## 12. License

MIT License (see `pyproject.toml` metadata).
