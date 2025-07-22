#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Go to the project root (two levels up from src/scripts/)
cd "$SCRIPT_DIR/../.."

# Ensure logs directory exists
mkdir -p logs
LOG_FILE="logs/db_backup.log"

{
  echo "===== Backup run started at $(date) ====="

  # Settings
  TIMESTAMP=$(date +%F-%H%M)
  DB_PATH="data/souptrader.db"
  BACKUP_NAME="souptrader_$TIMESTAMP.db"
  TMP_BACKUP="/tmp/$BACKUP_NAME"
  NAS_USER="rafikee"
  NAS_HOST="192.168.0.16"          # Or your NAS IP
  NAS_DIR="/volume1/homes/rafikee/trading_db_backups"  # Change to your NAS target folder

  echo "Backing up database..."

  # Create a consistent SQLite backup
  sqlite3 "$DB_PATH" ".backup '$TMP_BACKUP'"

  echo "Copying backup to NAS..."

  # Copy it to the NAS
  rsync -avz -e ssh "$TMP_BACKUP" "$NAS_USER@$NAS_HOST:$NAS_DIR/"

  echo "Cleaning up temporary backup file..."

  # Clean up temp file
  rm "$TMP_BACKUP"

  echo "Backup completed at $(date)"
  echo
} > "$LOG_FILE" 2>&1
