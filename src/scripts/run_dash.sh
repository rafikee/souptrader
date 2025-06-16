#!/bin/bash

# Kill any existing Dash process
pkill -f "python.*dash_app.py" || true

# Start Dash app in the background
cd /home/ubuntu/souptrader
source bin/activate
nohup python3 src/dash_app.py > logs/dash.log 2>&1 &

echo "Dash app started. Logs are in logs/dash.log" 