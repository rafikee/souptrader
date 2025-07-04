#!/bin/bash
cd /home/rafikee/dev/souptrader
source venv/bin/activate
python src/alpaca_update_db.py
python src/summary_update.py