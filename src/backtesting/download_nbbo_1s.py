"""
Databento NBBO 1s downloader

Downloads consolidated NBBO 1-second data for a given symbol.
Date ranges are configured in src/backtesting/download_config.yaml (uses trading_start_date
to trading_end_date with no warmup offset).

Saves per-day Parquet files under 
"data/backtest_data/{SYMBOL}/nbbo1s/YYYY-MM-DD.parquet".

Usage:
  python -m src.backtesting.download_nbbo_1s --symbol SMR

Notes:
- Uses Databento Historical client (async with bounded concurrency).
- Filters to regular trading hours using exchange_calendars (XNYS).
- Skips existing files if present (idempotent).
- Reads DATABENTO_API_KEY from environment (.env supported).
- Timestamps are UTC.
- Requires MY_API_KEY in environment (.env supported via python-dotenv).
- Date ranges configured in src/backtesting/download_config.yaml
"""

import io
import os
import sys
import argparse
import yaml
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio
import warnings
import time

import pandas as pd
from dotenv import load_dotenv, dotenv_values
import exchange_calendars as xcals

try:
    from databento import Historical
except ImportError as e:
    print("ERROR: databento package not found. Activate your venv and install it: 'pip install databento'", file=sys.stderr)
    raise

# Get project root (3 levels up from src/backtesting/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'backtest_data')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
# Config file is in the same directory as this script
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'download_config.yaml')
DATASET = "XNAS.ITCH"
SCHEMA = "bbo-1s"

# Calendar for RTH filtering (match backtest_intraday)
XNYS = xcals.get_calendar('XNYS')


def load_config() -> Dict[str, Any]:
    """Load download configuration from YAML file."""
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: Config file not found at {CONFIG_FILE}")
        sys.exit(1)
    
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Databento NBBO 1s to Parquet")
    parser.add_argument(
        "--symbol",
        required=True,
        help="Single stock symbol (e.g., SMR)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and overwrite existing day files",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def log_nbbo_output(symbol, force, output_lines, is_new_session=False):
    """Log NBBO download output to file, appending to existing log."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = os.path.join(LOGS_DIR, 'nbbo_download.log')
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # If new session, clear the file first
    mode = 'w' if is_new_session else 'a'
    
    with open(log_file, mode) as f:
        if is_new_session:
            f.write("=" * 60 + "\n")
            f.write("NEW DOWNLOAD SESSION STARTED\n")
            f.write("=" * 60 + "\n\n")
        
        f.write(f"NBBO Download - {timestamp}\n")
        f.write(f"Symbol: {symbol}\n")
        f.write(f"Force: {force}\n")
        f.write("-" * 40 + "\n")
        
        for line in output_lines:
            f.write(line + "\n")
        
        f.write("-" * 40 + "\n")
        f.write(f"Completed at {timestamp}\n\n")


def daterange_days(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    days: List[pd.Timestamp] = []
    current = pd.Timestamp(year=start.year, month=start.month, day=start.day, tz="UTC")
    last = pd.Timestamp(year=end.year, month=end.month, day=end.day, tz="UTC")
    while current <= last:
        days.append(current)
        current = current + pd.Timedelta(days=1)
    return days


def trading_days(start: pd.Timestamp, end: pd.Timestamp):
    """Get trading days using exchange calendar sessions."""
    # Use exchange calendar sessions (dates are session labels)
    sessions = XNYS.sessions_in_range(start.tz_localize(None), end.tz_localize(None))
    # Convert session labels to UTC midnight timestamps for filename formatting
    return [pd.Timestamp(s).tz_localize('UTC') for s in sessions]


def get_client() -> Historical:
    load_dotenv()
    # Require app-level API key and validate against .env
    provided_key = os.getenv("MY_API_KEY")
    if not provided_key:
        print("ERROR: MY_API_KEY not provided. Set it in the environment.", file=sys.stderr)
        sys.exit(1)
    env_vals = dotenv_values()
    expected_key = env_vals.get("MY_API_KEY")
    if not expected_key:
        print("ERROR: MY_API_KEY not set in .env on the server.", file=sys.stderr)
        sys.exit(1)
    if provided_key != expected_key:
        print("ERROR: Invalid MY_API_KEY.", file=sys.stderr)
        sys.exit(1)
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        print("ERROR: DATABENTO_API_KEY not found in environment (.env)")
        sys.exit(1)
    # Historical reads API key from environment; no args supported
    return Historical()


def is_rth_utc(ts_utc: pd.Timestamp) -> bool:
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.tz_localize("UTC")
    # Convert to ET day for session lookup
    et = ts_utc.tz_convert("US/Eastern")
    d = et.date()
    if not XNYS.is_session(d):
        return False
    open_ts = XNYS.session_open(d)
    close_ts = XNYS.session_close(d)
    return open_ts <= ts_utc <= close_ts


async def fetch_day_to_parquet(client: Historical, symbol: str, day: pd.Timestamp, sem: asyncio.Semaphore, progress_callback, retries: int = 3) -> tuple[str, int]:
    sym = symbol.upper()
    out_dir = os.path.join(DATA_ROOT, sym, "nbbo1s")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{day.strftime('%Y-%m-%d')}.parquet")

    # Skip existing files to avoid re-downloading
    if os.path.exists(out_path):
        return f"OK: Exists, skipping: {sym} {day.date()}", 0

    # Set start/end for the calendar day in UTC
    day_start = pd.Timestamp(day.strftime('%Y-%m-%d') + " 00:00:00", tz="UTC")
    day_end = pd.Timestamp(day.strftime('%Y-%m-%d') + " 23:59:59.999999999", tz="UTC")

    attempt = 0
    while True:
        try:
            async with sem:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    coro = client.timeseries.get_range_async(
                        dataset=DATASET,
                        symbols=[sym],
                        schema=SCHEMA,
                        start=day_start.isoformat(),
                        end=day_end.isoformat(),
                        stype_in="raw_symbol",
                    )
                    store = await coro
            break
        except Exception as e:
            attempt += 1
            if attempt > retries:
                return f"ERROR: {sym} {day.date()}", 0
            backoff = min(2 ** attempt, 30)
            await asyncio.sleep(backoff)

    # Convert to DataFrame
    df = store.to_df()

    if df.empty:
        return f"No data: {sym} {day.date()}", 0

    # Standardize columns
    # Ensure UTC tz-aware timestamps, coerce invalid to NaT
    if 'ts_event' in df.columns:
        df['ts_event'] = pd.to_datetime(df['ts_event'], utc=True, errors='coerce')
    else:
        # Some schemas use 'ts_event' consistently; guard anyway
        time_col = 'ts_event'
        if time_col not in df.columns:
            raise RuntimeError("Expected 'ts_event' column not found in NBBO data")

    # Drop rows with invalid timestamps before filtering
    df = df[df['ts_event'].notna()]
    # Filter to RTH using XNYS
    if not df.empty:
        df = df[df['ts_event'].apply(is_rth_utc)]

    # Keep only essential NBBO fields (bbo-1s uses _00 suffix for top-of-book)
    keep_cols = ["ts_event","bid_px_00","ask_px_00","bid_sz_00","ask_sz_00","symbol"]
    df = df[[c for c in keep_cols if c in df.columns]]

    # Sort and write Parquet
    df = df.sort_values("ts_event").reset_index(drop=True)
    df.to_parquet(out_path, index=False)
    
    # Update progress
    progress_callback(day.strftime('%Y-%m-%d'), len(df))
    
    return f"OK: {sym} {day.date()}", len(df)


def main() -> None:
    args = parse_args()
    symbol = args.symbol.strip().upper()

    # Load config and get date range
    config = load_config()
    trading_start = config['trading_start_date']
    trading_end = config['trading_end_date']
    
    # Parse dates
    start = pd.Timestamp(trading_start, tz="UTC")
    
    # Handle "today" keyword for end date
    if trading_end.lower() == "today":
        end = pd.Timestamp.now(tz="UTC").normalize()
    else:
        end = pd.Timestamp(trading_end, tz="UTC")
    
    print("=" * 60)
    print("NBBO Download Configuration")
    print("=" * 60)
    print(f"Trading Start Date: {config['trading_start_date']}")
    print(f"Trading End Date: {config['trading_end_date']}")
    print(f"Date Range: {start.date()} → {end.date()}")
    print(f"Note: NBBO uses trading dates directly (no warmup offset)")
    print("=" * 60)
    print()

    client = get_client()

    days = daterange_days(start, end)
    
    # Use trading days for progress calculation (more accurate percentages)
    trading_days_list = trading_days(start, end)
    
    # Capture output for logging
    output_lines = []
    output_lines.append(f"Starting NBBO download for symbol: {symbol}")
    output_lines.append(f"Force mode: {args.force}")
    output_lines.append(f"Date range: {start.date()} to {end.date()} (capped to today)")
    output_lines.append(f"Total calendar days: {len(days)}")
    output_lines.append(f"Expected trading days: {len(trading_days_list)}")
    output_lines.append("")

    # Pre-scan: if all expected daily files exist for the date range, skip this symbol entirely
    out_dir = os.path.join(DATA_ROOT, symbol, "nbbo1s")
    
    # Check if the nbbo1s subfolder exists
    if not os.path.isdir(out_dir):
        # Subfolder doesn't exist, need to download
        pass
    else:
        expected_files = [os.path.join(out_dir, f"{d.strftime('%Y-%m-%d')}.parquet") for d in days]
        if expected_files and all(os.path.exists(p) for p in expected_files) and not args.force:
            msg = f"All NBBO 1s daily files exist for {symbol}; skipping symbol."
            print(msg)
            output_lines.append(msg)
            log_nbbo_output(symbol, args.force, output_lines)
            return

    # Progress tracking
    completed = 0
    total_rows = 0
    start_time = time.time()
    
    def progress_callback(date_str, rows):
        nonlocal completed, total_rows
        completed += 1
        total_rows += rows
        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 else 0
        # Use trading days for accurate percentage calculation
        total_trading_days = len(trading_days_list)
        eta = (total_trading_days - completed) / rate if rate > 0 else 0
        
        progress_msg = f"Progress: {completed}/{total_trading_days} ({completed/total_trading_days*100:.1f}%) | {total_rows:,} rows | {rate:.1f} days/sec | ETA: {eta/60:.1f}min"
        print(f"\r{progress_msg}", end="", flush=True)
        output_lines.append(progress_msg)
        
        # Log immediately for real-time updates
        log_nbbo_output(symbol, args.force, [progress_msg], is_new_session=False)

    async def runner():
        sem = asyncio.Semaphore(6)  # bounded concurrency
        tasks = [fetch_day_to_parquet(client, symbol, day, sem, progress_callback) for day in days]
        results = await asyncio.gather(*tasks)
        
        # Clean progress summary
        ok_count = sum(1 for msg, rows in results if msg.startswith("OK:"))
        errors = [msg for msg, rows in results if msg.startswith("ERROR:")]
        no_data = [msg for msg, rows in results if msg.startswith("No data:")]
        
        # Calculate success percentage based on trading days
        success_pct = (ok_count / len(trading_days_list) * 100) if len(trading_days_list) > 0 else 0
        
        summary_msg = f"✓ Downloaded: {ok_count}/{len(trading_days_list)} trading days ({success_pct:.1f}%), {total_rows:,} total rows"
        print(f"\n\n{summary_msg}")
        output_lines.append("")
        output_lines.append(summary_msg)
        
        if no_data:
            no_data_msg = f"  • No data available: {len(no_data)} days (weekends/holidays or missing data)"
            print(no_data_msg)
            output_lines.append(no_data_msg)
        
        if errors:
            error_msg = f"  ✗ API Errors: {len(errors)} days (rate limits or API failures)"
            print(error_msg)
            output_lines.append(error_msg)

    # Capture stdout from the async runner
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()
    try:
        asyncio.run(runner())
        captured_text = captured_output.getvalue()
        if captured_text.strip():
            # Add each line of captured output
            for line in captured_text.strip().split('\n'):
                if line.strip():  # Only add non-empty lines
                    output_lines.append(line)
    finally:
        sys.stdout = old_stdout
    
    # Log the output (always append, never start new session from command line)
    log_nbbo_output(symbol, args.force, output_lines, is_new_session=False)


if __name__ == "__main__":
    main()
