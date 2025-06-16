#!/bin/bash

# Absolute paths
PROJECT_DIR="$HOME/souptrader"
APP_DIR="$PROJECT_DIR/src"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/fastapi.log"

# Activate virtual environment
source "$PROJECT_DIR/bin/activate"

# Kill any existing uvicorn process for this app
pkill -f "uvicorn main:app"

cd "$APP_DIR"

# Start FastAPI server with logging to ~/souptrader/logs
nohup uvicorn main:app --host 0.0.0.0 --port=8080 --app-dir "$APP_DIR" > "$LOG_FILE" 2>&1 &

echo "✅ FastAPI server started on port 8080. Logs: $LOG_FILE"
