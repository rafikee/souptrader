import sqlite3
import os
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(PROJECT_ROOT, 'data', 'souptrader.db')


def _get_conn():
    return sqlite3.connect(DB_PATH)


def init_tracker_tables():
    conn = _get_conn()
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS screener_tracker (
        ticker TEXT NOT NULL,
        date TEXT NOT NULL,
        close_price REAL,
        score REAL,
        screener_version TEXT NOT NULL,
        PRIMARY KEY (ticker, date, screener_version)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS screener_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        screener_version TEXT NOT NULL,
        entry_date TEXT NOT NULL,
        entry_price REAL,
        exit_date TEXT,
        exit_price REAL,
        return_pct REAL
    )
    ''')

    conn.commit()
    conn.close()


def record_screener_run(tickers_data, screener_version, run_date=None):
    """Record today's screener results and reconcile open/closed signals.

    Parameters
    ----------
    tickers_data : list[dict]
        Each dict must have 'ticker' and 'price'; 'score' is optional.
    screener_version : str
        'v1' or 'v2'.
    run_date : str | None
        Override date (YYYY-MM-DD). Defaults to today.
    """
    if not tickers_data:
        print("[Tracker] No tickers to record — skipping.")
        return

    init_tracker_tables()

    if run_date is None:
        run_date = datetime.now().strftime('%Y-%m-%d')

    conn = _get_conn()
    cursor = conn.cursor()

    today_tickers = set()

    for item in tickers_data:
        ticker = item['ticker']
        price = item['price']
        score = item.get('score')
        today_tickers.add(ticker)

        cursor.execute('''
        INSERT OR REPLACE INTO screener_tracker
            (ticker, date, close_price, score, screener_version)
        VALUES (?, ?, ?, ?, ?)
        ''', (ticker, run_date, price, score, screener_version))

    # Find currently active (open) signals for this screener version
    cursor.execute('''
    SELECT id, ticker FROM screener_signals
    WHERE screener_version = ? AND exit_date IS NULL
    ''', (screener_version,))
    active_signals = cursor.fetchall()

    active_tickers = {t for _, t in active_signals}

    # Close signals for tickers that dropped off the screener
    for signal_id, ticker in active_signals:
        if ticker not in today_tickers:
            cursor.execute('''
            SELECT close_price FROM screener_tracker
            WHERE ticker = ? AND screener_version = ?
            ORDER BY date DESC LIMIT 1
            ''', (ticker, screener_version))
            row = cursor.fetchone()
            exit_price = row[0] if row else None

            cursor.execute('SELECT entry_price FROM screener_signals WHERE id = ?',
                           (signal_id,))
            entry_row = cursor.fetchone()
            entry_price = entry_row[0] if entry_row else None

            return_pct = None
            if entry_price and exit_price and entry_price > 0:
                return_pct = round(((exit_price - entry_price) / entry_price) * 100, 2)

            cursor.execute('''
            UPDATE screener_signals
            SET exit_date = ?, exit_price = ?, return_pct = ?
            WHERE id = ?
            ''', (run_date, exit_price, return_pct, signal_id))

    # Open new signals for tickers appearing for the first time (or re-appearing)
    new_count = 0
    for item in tickers_data:
        ticker = item['ticker']
        if ticker not in active_tickers:
            cursor.execute('''
            INSERT INTO screener_signals
                (ticker, screener_version, entry_date, entry_price)
            VALUES (?, ?, ?, ?)
            ''', (ticker, screener_version, run_date, item['price']))
            new_count += 1

    conn.commit()

    closed_count = sum(1 for _, t in active_signals if t not in today_tickers)
    continuing = len(today_tickers & active_tickers)

    print(f"\n[Tracker] {screener_version} | Date: {run_date}")
    print(f"[Tracker] Today: {len(today_tickers)} | New signals: {new_count} "
          f"| Continuing: {continuing} | Closed: {closed_count}")

    conn.close()
