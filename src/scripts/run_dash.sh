#!/bin/bash

# Kill any existing Dash process
pkill -f "python.*dash_app.py" || true

# Start Dash app in the background
cd /home/rafikee/dev/souptrader
source venv/bin/activate
nohup python src/dash_app.py > logs/dash.log 2>&1 &

echo "Dash app started. Logs are in logs/dash.log" 