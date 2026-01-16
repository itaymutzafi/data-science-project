#!/bin/bash
# Completely reset the development environment

# Ensure we are in the project root
cd "$(dirname "$0")/.."

echo "⚠️  Killing any processes using .venv..."
# Try to kill processes using the .venv directory (Mac/Linux)
# lsof list open files, grep filters for .venv, awk gets PID, xargs kills
lsof +D .venv | grep python | awk '{print $2}' | xargs kill -9 2>/dev/null

echo "🗑️  Deleting old .venv..."
rm -rf .venv

echo "🐍 Creating new .venv..."
# Using python3 (which appears to be 3.13 on your system)
python3 -m venv .venv

echo "uq  Activating .venv..."
source .venv/bin/activate

echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install ipykernel

echo "⚙️  Registering Kernel..."
python -m ipykernel install --user --name=data-science-project --display-name "Python (Data Science Project)"

echo "✅ Done! Please restart VS Code."
