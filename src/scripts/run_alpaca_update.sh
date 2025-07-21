#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Go to the project root (two levels up from src/scripts/)
cd "$SCRIPT_DIR/../.."
source venv/bin/activate
python src/alpaca_update_db.py
python src/summary_update.py