'''
import sqlite3
import pandas as pd

conn = sqlite3.connect('souptrader.db')
df = pd.read_sql_query(f"SELECT * FROM trades", conn)
conn.close()
print(df)


# upload csv to table
conn = sqlite3.connect('souptrader.db')
df = pd.read_csv('import.csv')
df.to_sql('trades', conn, if_exists='append', index=False)
conn.close()
'''

# create monthly_summary table
import sqlite3

# Connect to your SQLite DB
conn = sqlite3.connect('souptrader.db')
cursor = conn.cursor()

# Create the table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS monthly_summary (
    month TEXT PRIMARY KEY,
    realized_profit REAL
)
''')

# Create quarterly_summary table
cursor.execute('''
CREATE TABLE IF NOT EXISTS quarterly_summary (
    quarter TEXT PRIMARY KEY,
    realized_profit REAL,
    cumulative_deposits REAL,
    annualized_return REAL
)
''')

# Create yearly_summary table
cursor.execute('''
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
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS yearly_stock_pnl (
    ticker TEXT,
    year INTEGER,
    profit REAL,
    trade_count INTEGER,
    PRIMARY KEY (ticker, year)
)
''')

# Commit and close
conn.commit()
conn.close()

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


And this for the funds table
CREATE TABLE "funds"   (
  "id" TEXT NOT NULL PRIMARY KEY,
  "type" TEXT,
  "date" DATE,
  "amount" DECIMAL(12,2),
  "platform" TEXT
)
'''