#!/bin/bash
# CANSLIM v2 screener — runs main_v2.py, logs to screener_v2.log
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

mkdir -p logs
LOG_FILE="logs/screener_v2.log"

{
  echo "===== CANSLIM v2 run started at $(date) ====="
  
  source venv/bin/activate
  
  python src/screener/main_v2.py
  
  echo "CANSLIM v2 completed at $(date)"
  echo
} >> "$LOG_FILE" 2>&1
