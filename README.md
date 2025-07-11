# SoupTrader

A trading performance tracking system that integrates with Alpaca API to track and analyze trading performance. In my version I manually uploaded historical data not from Alpaca into the db. Now this program updates my database from Alpaca daily.

## Setup

Firstly you'll have to create a python virtual environment. I createed venv folder in the root. Then make sure to install all the requirements.

1. Create a `.env` file with your Alpaca API credentials
```bash
cp .env.example .env
# Edit .env with your actual API credentials
```

2. Initialize the database
```bash
python3 setup_db.py
```

## Usage

The system consists of two main scripts:

1. `alpaca_update_db.py` - Fetches latest trades and updates from Alpaca API
2. `summary_update.py` - Calculates and updates performance metrics

You can run both scripts using:
```bash
./run_alpaca_update.sh
```

Then there is the dashboard, there is a script that runs the dashboard in the background

## Database Structure

The system uses SQLite with the following tables:
- `trades` - Raw trade data from Alpaca
- `funds` - Fund transfers and deposits
- `monthly_summary` - Monthly performance metrics
- `quarterly_summary` - Quarterly performance metrics
- `yearly_summary` - Yearly performance metrics
- `yearly_stock_pnl` - Yearly P&L by stock

## Logging

Logs are stored in:
- `alpaca_update.log` - Alpaca API update logs
- `summary_update.log` - Performance calculation logs
- `setup.log` - Database setup logs 

## Things to consider 

- There is a cronjob that runs this script at 1 am daily run_alpaca_update.sh
- You need to make sure all the paths are updated in the scripts and in the py files
- Any historic trades not from alpaca need to be loaded manually