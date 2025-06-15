#!/bin/bash
cd /home/ubuntu/souptrader
source bin/activate
python3 src/core/alpaca_update_db.py
python3 src/core/summary_update.py