"""
Alpaca historical bars downloader

Downloads split-adjusted 5-minute and daily bars for given symbols between
2024-01-01 and 2025-12-31 and stores them as monthly Parquet files under
"data/backtest_data/{SYMBOL}/5Min/YYYY-MM.parquet" and
"data/backtest_data/{SYMBOL}/Daily/YYYY-MM.parquet".

Usage:
  python -m src.backtesting.download_bars --symbols AAPL,MSFT,SMR

Notes:
- Uses Alpaca StockHistoricalDataClient and adjustment="split".
- Saves all trading/non-trading hours (no filtering); timestamps are UTC.
- Skips downloading a month if the target file already exists (idempotent).
- Requires APCA_API_KEY_ID and APCA_API_SECRET_KEY in environment (.env supported via python-dotenv).
"""

import io
import os
import sys
import argparse
from datetime import datetime, timedelta
from typing import List, Tuple

import pandas as pd
from dotenv import load_dotenv, dotenv_values
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


# Get project root (3 levels up from src/backtesting/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'backtest_data')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
START_DATE_STR = "2024-01-01"
END_DATE_STR = "2025-12-31"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Alpaca historical 5Min and Daily bars to Parquet")
    parser.add_argument(
        "--symbols",
        required=True,
        help="Comma-separated list of stock symbols (e.g., AAPL,MSFT,SMR)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and overwrite existing month files",
    )
    return parser.parse_args()


def month_range(start: pd.Timestamp, end: pd.Timestamp) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Yield (month_start, month_end_inclusive) pairs covering [start, end]."""
    result: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    current = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    end_month_start = pd.Timestamp(year=end.year, month=end.month, day=1, tz="UTC")
    while current <= end_month_start:
        next_month = (current + pd.offsets.MonthBegin(1))
        # Month end is the last instant before next_month
        month_end = min(end, next_month - pd.Timedelta(nanoseconds=1))
        month_start = max(start, current)
        result.append((month_start, month_end))
        current = next_month
    return result


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def log_bars_output(symbol, timeframe, force, output_lines, is_new_session=False):
    """Log bars download output to file, appending to existing log."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = os.path.join(LOGS_DIR, f'{timeframe.lower()}_bars_download.log')
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # If new session, clear the file first
    mode = 'w' if is_new_session else 'a'
    
    with open(log_file, mode) as f:
        if is_new_session:
            f.write("=" * 60 + "\n")
            f.write(f"NEW {timeframe.upper()} BARS DOWNLOAD SESSION STARTED\n")
            f.write("=" * 60 + "\n\n")
        
        f.write(f"{timeframe} Bars Download - {timestamp}\n")
        f.write(f"Symbol: {symbol}\n")
        f.write(f"Force: {force}\n")
        f.write("-" * 40 + "\n")
        
        for line in output_lines:
            f.write(line + "\n")
        
        f.write("-" * 40 + "\n")
        f.write(f"Completed at {timestamp}\n\n")


def download_month_bars(
    client: StockHistoricalDataClient,
    symbol: str,
    timeframe: TimeFrame,
    timeframe_str: str,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    force: bool,
) -> bool:
    """Download bars for a single month and save to Parquet."""
    
    # Output file path
    month_str = month_start.strftime("%Y-%m")
    output_dir = os.path.join(DATA_ROOT, symbol, timeframe_str)
    ensure_dir(output_dir)
    output_file = os.path.join(output_dir, f"{month_str}.parquet")
    
    # Skip if file exists and not forcing
    if os.path.exists(output_file) and not force:
        print(f"Skipping {symbol} {timeframe_str} {month_str} (file exists)")
        return True
    
    # Create request
    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=timeframe,
        start=month_start.to_pydatetime(),
        end=month_end.to_pydatetime(),
        adjustment="split",
    )

    print(f"Downloading {symbol} {timeframe_str} {month_start.date()} → {month_end.date()}")
    bars = client.get_stock_bars(request)
    if not bars or not hasattr(bars, "df") or bars.df.empty:
        print(f"  No data for {symbol} {timeframe_str} {month_str}")
        return False

    # Convert to DataFrame and save
    df = bars.df
    
    # Reset index to convert timestamp from index to column
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    elif 'timestamp' not in df.columns:
        df = df.reset_index()
    
    df.to_parquet(output_file, index=False)
    print(f"  Saved {len(df)} bars to {output_file}")
    return True


def main():
    load_dotenv()
    
    # Get API credentials
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")
    if not api_key:
        print("Error: APCA_API_KEY_ID not found in environment")
        sys.exit(1)
    if not secret_key:
        print("Error: APCA_API_SECRET_KEY not found in environment")
        sys.exit(1)
    
    # Parse arguments
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    
    # Initialize client
    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    
    # Date range
    start = pd.Timestamp(START_DATE_STR, tz="UTC")
    # Inclusive end date: set to end of day UTC
    end = pd.Timestamp(END_DATE_STR + " 23:59:59", tz="UTC")
    
    # Cap end date to today to avoid downloading future months
    today = pd.Timestamp.now(tz="UTC").normalize()
    if end > today:
        end = today
        print(f"Note: End date capped to today ({today.date()})")
    
    print(f"Downloading 5Min and Daily bars from {start.date()} to {end.date()}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Force: {args.force}")
    print()
    
    # Process each symbol
    for i, symbol in enumerate(symbols):
        print(f"Processing {symbol} ({i+1}/{len(symbols)})")
        
        # Get month ranges
        months = month_range(start, end)
        total_months = len(months)
        
        # Download 5Min bars
        print(f"  Downloading 5Min bars...")
        success_5min = 0
        errors_5min = 0
        for month_start, month_end in months:
            try:
                if download_month_bars(client, symbol, TimeFrame(5, TimeFrameUnit.Minute), "5Min", month_start, month_end, args.force):
                    success_5min += 1
            except Exception as e:
                errors_5min += 1
                print(f"    Error downloading {symbol} 5Min {month_start.strftime('%Y-%m')}: {e}")
        
        pct_5min = (success_5min / total_months * 100) if total_months > 0 else 0
        print(f"    ✓ 5Min: {success_5min}/{total_months} months ({pct_5min:.1f}%)" + (f" | {errors_5min} errors" if errors_5min > 0 else ""))
        
        # Download Daily bars
        print(f"  Downloading Daily bars...")
        success_daily = 0
        errors_daily = 0
        for month_start, month_end in months:
            try:
                if download_month_bars(client, symbol, TimeFrame(1, TimeFrameUnit.Day), "Daily", month_start, month_end, args.force):
                    success_daily += 1
            except Exception as e:
                errors_daily += 1
                print(f"    Error downloading {symbol} Daily {month_start.strftime('%Y-%m')}: {e}")
        
        pct_daily = (success_daily / total_months * 100) if total_months > 0 else 0
        print(f"    ✓ Daily: {success_daily}/{total_months} months ({pct_daily:.1f}%)" + (f" | {errors_daily} errors" if errors_daily > 0 else ""))
        print(f"  Completed {symbol}: 5Min({success_5min}/{total_months}) Daily({success_daily}/{total_months})")
        
        # Log the results
        log_5min_msg = f"✓ Downloaded: {success_5min}/{total_months} months ({pct_5min:.1f}%)" + (f" | {errors_5min} errors" if errors_5min > 0 else "")
        log_daily_msg = f"✓ Downloaded: {success_daily}/{total_months} months ({pct_daily:.1f}%)" + (f" | {errors_daily} errors" if errors_daily > 0 else "")
        log_bars_output(symbol, "5Min", args.force, [log_5min_msg], i == 0)
        log_bars_output(symbol, "Daily", args.force, [log_daily_msg], i == 0)
        print()
    
    print("Download completed!")


if __name__ == "__main__":
    main()