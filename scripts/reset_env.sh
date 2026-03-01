#!/bin/bash
# Recreate the local development environment from scratch.

set -euo pipefail

# Ensure we are in the project root
cd "$(dirname "$0")/.."

echo "[reset] Stopping Python processes that currently use .venv (if any)..."
PIDS="$(lsof +D .venv 2>/dev/null | awk '/python/ {print $2}' | sort -u || true)"
if [ -n "$PIDS" ]; then
    kill -9 $PIDS 2>/dev/null || true
fi

echo "[reset] Removing existing .venv directory..."
rm -rf .venv

echo "[reset] Creating a fresh .venv..."
python3 -m venv .venv

echo "[reset] Activating .venv..."
source .venv/bin/activate

echo "[reset] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pip install ipykernel

echo "[reset] Registering kernel 'data-science-project'..."
python -m ipykernel install --user --name=data-science-project --display-name "Python (Data Science Project)"

echo "[reset] Completed successfully."
