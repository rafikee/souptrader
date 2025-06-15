#!/bin/bash
cd /home/ubuntu/souptrader
source bin/activate
python3 src/alpaca_update_db.py
python3 src/summary_update.py