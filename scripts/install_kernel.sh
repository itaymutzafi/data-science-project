#!/bin/bash
# Register the project Jupyter kernel from an existing virtual environment.

set -euo pipefail

# Ensure we are in the project root
cd "$(dirname "$0")/.."

# Check if .venv exists
if [ -d ".venv" ]; then
    echo "[kernel] Activating virtual environment..."
    source .venv/bin/activate
else
    echo "[kernel] Error: .venv not found."
    exit 1
fi

echo "[kernel] Installing ipykernel..."
pip install ipykernel

echo "[kernel] Registering kernel 'data-science-project'..."
python -m ipykernel install --user --name=data-science-project --display-name "Python (Data Science Project)"

echo "[kernel] Kernel installation completed successfully."
