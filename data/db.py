import sqlite3
import pandas as pd

conn = sqlite3.connect('souptrader.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE "swings_funding"   (
  "id" TEXT NOT NULL PRIMARY KEY,
  "flow" TEXT,
  "date" DATE,
  "amount" DECIMAL(12,2),
  "platform" TEXT
)
''')

# Commit and close
conn.commit()
conn.close()

'''
# Create the table if it doesn't exist
CREATE TABLE IF NOT EXISTS monthly_summary (
    month TEXT PRIMARY KEY,
    realized_profit REAL
)

# Create quarterly_summary table
CREATE TABLE IF NOT EXISTS quarterly_summary (
    quarter TEXT PRIMARY KEY,
    realized_profit REAL,
    cumulative_deposits REAL,
    annualized_return REAL
)

# Create yearly_summary table
CREATE TABLE IF NOT EXISTS yearly_summary (
    year INTEGER PRIMARY KEY,
    realized_profit REAL,
    options_profit REAL,
    total_fees REAL,
    total_dividends REAL,
    total_deposits REAL,
    annualized_return REAL,
    total_trades_closed INTEGER,
    win_rate REAL,
    avg_trade_duration REAL,
    avg_profit_per_trade REAL,
    avg_position_size REAL,
    simple_avg_return REAL,
    position_weighted_return REAL,
    avg_winning_position_size REAL,
    avg_winning_return_pct REAL,
    avg_losing_position_size REAL,
    avg_losing_return_pct REAL
)

CREATE TABLE IF NOT EXISTS yearly_stock_pnl (
    ticker TEXT,
    year INTEGER,
    profit REAL,
    trade_count INTEGER,
    PRIMARY KEY (ticker, year)
)
'''


'''
Creating the DB using this

CREATE TABLE "trades"    (
  "id" TEXT NOT NULL PRIMARY KEY,
  "side" TEXT,
  "filled_time" DATETIME,
  "amount" DECIMAL(12,4),
  "qty" DECIMAL(14,4),
  "price" DECIMAL(14,4),
  "platform" TEXT,
  "status" TEXT,
  "category" TEXT,
  "symbol" TEXT,
  "asset_class" TEXT,
  "description" TEXT
)

'''