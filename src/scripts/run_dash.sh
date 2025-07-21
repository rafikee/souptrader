#!/bin/bash

# Kill any existing Dash process
pkill -f "python.*dash_app.py" || true

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Go to the project root (two levels up from src/scripts/)
cd "$SCRIPT_DIR/../.."

# Start Dash app in the background
source venv/bin/activate
nohup python src/dash_app.py > logs/dash.log 2>&1 &

echo "Dash app started. Logs are in logs/dash.log" 