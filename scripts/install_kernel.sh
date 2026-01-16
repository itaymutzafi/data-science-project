#!/bin/bash
# Re-install Jupyter Kernel for Data Science Project

# Ensure we are in the project root
cd "$(dirname "$0")/.."

# Check if .venv exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Error: .venv not found!"
    exit 1
fi

echo "Installing ipykernel..."
pip install ipykernel

echo "Registering kernel 'data-science-project'..."
python -m ipykernel install --user --name=data-science-project --display-name "Python (Data Science Project)"

echo "Kernel installation complete."
