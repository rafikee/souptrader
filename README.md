# SoupTrader

A trading performance tracking system that integrates with Alpaca API to track and analyze trading performance.

## Setup

1. Clone the repository
```bash
git clone <repository-url>
cd souptrader
```

2. Create and activate a virtual environment

If you are starting from scratch:
```bash
python3 -m venv .
source bin/activate
```

If you already have a virtual environment set up in your project directory:
```bash
cd souptrader
source bin/activate
```

3. Install required packages
```bash
pip install -r requirements.txt
```

Required packages (listed in requirements.txt):
- pandas: For data manipulation and analysis
- python-dotenv: For loading environment variables
- requests: For making API calls to Alpaca

4. Create a `.env` file with your Alpaca API credentials
```bash
cp .env.example .env
# Edit .env with your actual API credentials
```

5. Initialize the database
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