#!/bin/bash
# One-shot environment setup: create venv (if missing), install deps, editable package, and Jupyter kernel.

set -e

# Move to project root
cd "$(dirname "$0")/.."

VENV=".venv"
PYTHON_BIN="${PYTHON:-python3}"
KERNEL_NAME="data-science-project"
KERNEL_DISPLAY="Python (Data Science Project)"

if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment at $VENV"
    "$PYTHON_BIN" -m venv "$VENV"
else
    echo "Using existing virtual environment at $VENV"
fi

source "$VENV/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing project requirements..."
pip install -r requirements.txt

echo "Installing project in editable mode..."
pip install -e .

echo "Installing ipykernel and registering kernel '$KERNEL_NAME'..."
pip install ipykernel
python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY"

echo "Setup complete. Select kernel '$KERNEL_DISPLAY' in Jupyter/VS Code."
