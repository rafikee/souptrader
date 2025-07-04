import sqlite3
import logging

logging.basicConfig(
    filename='/home/rafikee/dev/souptrader/logs/setup.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def create_tables():
    try:
        logging.info('Starting database setup')
        conn = sqlite3.connect('/home/rafikee/dev/souptrader/data/souptrader.db')
        cursor = conn.cursor()

        # Create trades table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id TEXT PRIMARY KEY,
            symbol TEXT,
            asset_class TEXT,
            qty REAL,
            price REAL,
            amount REAL,
            side TEXT,
            status TEXT,
            filled_time TEXT,
            platform TEXT,
            category TEXT
        )
        ''')
        logging.info('Created trades table')

        # Create funds table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS funds (
            id TEXT PRIMARY KEY,
            type TEXT,
            amount REAL,
            date TEXT,
            platform TEXT
        )
        ''')
        logging.info('Created funds table')

        # Create monthly_summary table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS monthly_summary (
            month TEXT PRIMARY KEY,
            realized_profit REAL
        )
        ''')
        logging.info('Created monthly_summary table')

        # Create quarterly_summary table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS quarterly_summary (
            quarter TEXT PRIMARY KEY,
            realized_profit REAL,
            cumulative_deposits REAL,
            annualized_return REAL
        )
        ''')
        logging.info('Created quarterly_summary table')

        # Create yearly_summary table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS yearly_summary (
            year INTEGER PRIMARY KEY,
            realized_profit REAL,
            total_fees REAL,
            total_dividends REAL,
            options_profit REAL,
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
        logging.info('Created yearly_summary table')

        # Create yearly_stock_pnl table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS yearly_stock_pnl (
            ticker TEXT,
            year INTEGER,
            profit REAL,
            PRIMARY KEY (ticker, year)
        )
        ''')
        logging.info('Created yearly_stock_pnl table')

        conn.commit()
        conn.close()
        logging.info('Database setup completed successfully')
        print('Database setup completed successfully')

    except Exception as e:
        logging.error(f'Error in database setup: {str(e)}')
        if 'conn' in locals():
            conn.close()
        print(f'Error in database setup: {str(e)}')

if __name__ == '__main__':
    create_tables() 