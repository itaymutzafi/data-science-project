# Stock Return Prediction Under Multi-Source Signals

**Workshop in Data Science (Final Project), Tel Aviv University**  
**Team 003**

## Project Cover Page
This repository contains the full implementation and report for a final-course project on short-horizon stock return prediction for four large-cap technology companies:
`AAPL`, `AMZN`, `GOOG`, and `MSFT`.

The work is organized as a reproducible research artifact:
- `Final_Project_Report.ipynb` is the primary academic deliverable.
- `src/` contains modular code for data ingestion, feature engineering, model screening, and evaluation.
- `data/` stores raw and processed artifacts needed for reruns.

The main methodological emphasis is reliability under realistic time-series constraints:
strict chronological validation, leakage-safe feature alignment, and transparent benchmark comparisons.

## Abstract
We model two complementary prediction targets:
- **Continuous target**: next-step return magnitude.
- **Discrete target**: next-step return direction (up/down).

Signals combine market microstructure proxies, technical indicators, macro context, and news-based sentiment features derived with FinBERT.  
Model quality is evaluated out-of-sample under walk-forward protocols, with the discrete branch treated as the primary decision-oriented track.

## Repository Structure
```text
.
├── Final_Project_Report.ipynb
├── README.md
├── requirements.txt
├── environment.yml
├── scripts/
│   ├── setup_env.sh
│   ├── install_kernel.sh
│   └── reset_env.sh
├── data/
│   ├── raw/
│   └── processed/
├── literature/
├── reports/
└── src/
    ├── config.py
    ├── data/
    ├── features/
    ├── models/
    ├── evaluation/
    ├── pipelines/
    └── utils/
```

## Environment Setup
### Prerequisites
- Python `3.11` recommended (`>=3.9` supported by package metadata).
- macOS/Linux shell (scripts are Bash-based).
- Internet access for market-data pulls (`yfinance`) and, when recomputing sentiment, model/news dependencies.

### Recommended Setup (One Command)
```bash
bash scripts/setup_env.sh
```

This command:
1. Creates `.venv` if needed.
2. Installs dependencies from `requirements.txt`.
3. Installs the package in editable mode.
4. Registers Jupyter kernel `Python (Data Science Project)`.

### Alternative Setup
#### `pip + venv`
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --user --name data-science-project --display-name "Python (Data Science Project)"
```

#### Conda
```bash
conda env create -f environment.yml
conda activate ds-project
python -m ipykernel install --user --name data-science-project --display-name "Python (Data Science Project)"
```

## Data Requirements
Expected data locations:
- News corpus (shared file): `data/raw/`
- Runtime caches (primary): `data/cache/`
- Legacy fallback caches (auto-migrated on read): `data/raw/` and `data/processed/`

Key files commonly used by the notebook:
- `data/raw/news_last_5y.parquet`
- `data/cache/prices/AAPL.parquet`, `data/cache/prices/AMZN.parquet`, `data/cache/prices/GOOG.parquet`, `data/cache/prices/MSFT.parquet`
- `data/cache/market/auxiliary_market_data.parquet`
- `data/cache/prophet/<TICKER>_prophet_predictions.parquet`
- `data/cache/sec_filings/<TICKER>_sec_filings.parquet`
- `data/cache/sentiment/daily_sentiment_features.csv`

If primary caches are missing, the pipeline can regenerate them. If legacy caches exist,
the code reads them and materializes the primary cache layout automatically.

## How To Run
1. Activate the project environment.
2. Open Jupyter Lab/Notebook:
   ```bash
   jupyter lab
   ```
3. Open `Final_Project_Report.ipynb`.
4. Select kernel `Python (Data Science Project)`.
5. Run cells sequentially from top to bottom.

## Reproducibility Controls (Section 6.1)
The report exposes explicit control knobs for compute budget and experiment breadth.

### Sentiment depth
In the sentiment cell, `SENTIMENT_DEPTH` controls per-day/per-company sampling before FinBERT inference:
- `low` or `quick` -> `1` headline
- `medium` or `balanced` -> `3` headlines
- `high` -> `5` headlines
- `deep`, `full`, or `all` -> use all headlines (`0` cap)

### Search breadth
In the robustness experiment section:
- `CONTINUOUS_SEARCH_BUDGET`
- `DISCRETE_SEARCH_BUDGET`

Higher values increase subset search coverage and runtime.  
Current notebook defaults are calibrated for practical reruns on student hardware.

## Methodological Guardrails
The codebase enforces several principles to protect validity:
- Time-aware train/validation sequencing (no random shuffling across time).
- Lagged integration of sentiment features to reduce leakage risk.
- Feature-set governance via `src/features/sets.py`, including collinearity-aware block sampling.
- Consistent naming canonicalization via `src/utils/feature_names.py`.
- Two-tier schema validation in `src/data/validate_schema.py`:
  - Default lightweight startup gate (`strict=False`).
  - Full schema/type checks when explicitly enabled (`strict=True`), e.g. after fresh API refresh.

## Engineering Notes
- The project favors notebook-report reproducibility over ad hoc scripts.
- Caches are unified under `data/cache/` for deterministic reruns and clearer ownership.
- Backward compatibility is preserved: legacy caches under `data/raw`/`data/processed` are still readable.
- The shared news source remains stable at `data/raw/news_last_5y.parquet`.
- Schemas are stored under `src/data/schemas/` (legacy path fallback is kept for compatibility).
- Environment helpers are provided under `scripts/`:
  - `setup_env.sh`: first-time setup.
  - `install_kernel.sh`: kernel registration only.
  - `reset_env.sh`: full local environment reset.

## Suggested Evaluation Protocol (For Reviewers)
1. Run `bash scripts/setup_env.sh`.
2. Launch `Final_Project_Report.ipynb`.
3. Execute all cells in order.
4. Inspect Section `6.1` for robustness experiment outputs:
   - Continuous: `RMSE`, `R2`, `Adjusted R2`, directional accuracy.
   - Discrete: `Accuracy`, `Precision`, `Recall`, benchmark comparison.

## Team
- Itay
- Moran
- Shaked

## Academic Context
Submitted as partial fulfillment of the requirements of the Workshop in Data Science course at Tel Aviv University.
