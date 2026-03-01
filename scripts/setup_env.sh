#!/bin/bash
# One-shot environment setup for reproducible local execution.

set -euo pipefail

# Move to project root
cd "$(dirname "$0")/.."

VENV=".venv"
PYTHON_BIN="${PYTHON:-python3}"
KERNEL_NAME="data-science-project"
KERNEL_DISPLAY="Python (Data Science Project)"

if [ ! -d "$VENV" ]; then
    echo "[setup] Creating virtual environment at $VENV"
    "$PYTHON_BIN" -m venv "$VENV"
else
    echo "[setup] Using existing virtual environment at $VENV"
fi

source "$VENV/bin/activate"

echo "[setup] Upgrading pip..."
pip install --upgrade pip

echo "[setup] Installing project requirements..."
pip install -r requirements.txt

echo "[setup] Installing project in editable mode..."
pip install -e .

echo "[setup] Installing ipykernel and registering kernel '$KERNEL_NAME'..."
pip install ipykernel
python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY"

echo "[setup] Completed successfully. Select kernel '$KERNEL_DISPLAY' in Jupyter/VS Code."
