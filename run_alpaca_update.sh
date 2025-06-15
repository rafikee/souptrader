#!/bin/bash
cd /home/ubuntu/souptrader
source bin/activate
python3 alpaca_update_db.py
python3 summary_update.py