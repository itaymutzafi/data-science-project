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

## Environment Setup
### Prerequisites
- Python `3.11` recommended (`>=3.9` supported by package metadata).
- macOS/Linux shell (scripts are Bash-based).
- Internet access for market-data pulls (`yfinance`) and when recomputing sentiment (model/news dependencies).

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
