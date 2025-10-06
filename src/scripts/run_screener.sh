#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Go to the project root (two levels up from src/scripts/)
cd "$SCRIPT_DIR/../.."

# Ensure logs directory exists
mkdir -p logs
LOG_FILE="logs/screener.log"

{
  echo "===== Screener run started at $(date) ====="
  
  # Activate virtual environment
  source venv/bin/activate
  
  # Run the screener
  python src/screener/main.py
  
  echo "Screener completed at $(date)"
  echo
} > "$LOG_FILE" 2>&1
