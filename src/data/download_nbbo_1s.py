"""
Databento NBBO 1s downloader

Downloads consolidated NBBO 1-second data for a given symbol between
2024-09-01 and 2025-08-31 from dataset "XNAS.ITCH" using schema "bbo-1s".
Saves per-day Parquet files under 
"/mnt/usb/souptrader/data/{SYMBOL}/nbbo1s/YYYY-MM-DD.parquet".

Usage:
  python -m src.data.download_nbbo_1s --symbol SMR

Notes:
- Uses Databento Historical client (async with bounded concurrency).
- Filters to regular trading hours using exchange_calendars (XNYS).
- Overwrites existing files if present.
- Reads DATABENTO_API_KEY from environment (.env supported).
- Timestamps are UTC.
"""

import os
import sys
import argparse
from typing import List, Optional
from datetime import datetime, timedelta
import asyncio
import warnings
import time

import pandas as pd
from dotenv import load_dotenv
import exchange_calendars as xcals

try:
    from databento import Historical
except ImportError as e:
    print("ERROR: databento package not found. Activate your venv and install it: 'pip install databento'", file=sys.stderr)
    raise

DATA_ROOT = "/mnt/usb/souptrader/data"
START_DATE_STR = "2024-09-01"
END_DATE_STR = "2025-08-31"
DATASET = "XNAS.ITCH"
SCHEMA = "bbo-1s"

# Calendar for RTH filtering (match backtest_intraday)
XNYS = xcals.get_calendar('XNYS')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Databento NBBO 1s to Parquet")
    parser.add_argument(
        "--symbol",
        required=True,
        help="Single stock symbol (e.g., SMR)",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def daterange_days(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    days: List[pd.Timestamp] = []
    current = pd.Timestamp(year=start.year, month=start.month, day=start.day, tz="UTC")
    last = pd.Timestamp(year=end.year, month=end.month, day=end.day, tz="UTC")
    while current <= last:
        days.append(current)
        current = current + pd.Timedelta(days=1)
    return days


def get_client() -> Historical:
    load_dotenv()
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

    # Overwrite policy: always overwrite
    if os.path.exists(out_path):
        os.remove(out_path)

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

    start = pd.Timestamp(START_DATE_STR, tz="UTC")
    end = pd.Timestamp(END_DATE_STR, tz="UTC")

    client = get_client()

    days = daterange_days(start, end)

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
        eta = (len(days) - completed) / rate if rate > 0 else 0
        
        print(f"\rProgress: {completed}/{len(days)} ({completed/len(days)*100:.1f}%) | "
              f"{total_rows:,} rows | {rate:.1f} days/sec | ETA: {eta/60:.1f}min", end="", flush=True)

    async def runner():
        sem = asyncio.Semaphore(6)  # bounded concurrency
        tasks = [fetch_day_to_parquet(client, symbol, day, sem, progress_callback) for day in days]
        results = await asyncio.gather(*tasks)
        
        # Clean progress summary
        ok_count = sum(1 for msg, rows in results if msg.startswith("OK:"))
        errors = [msg for msg, rows in results if msg.startswith("ERROR:")]
        no_data = [msg for msg, rows in results if msg.startswith("No data:")]
        
        print(f"\n\nCompleted: {ok_count}/{len(days)} days, {total_rows:,} total rows")
        if errors:
            print(f"Errors: {len(errors)}")
        if no_data:
            print(f"No data: {len(no_data)}")

    asyncio.run(runner())


if __name__ == "__main__":
    main()
