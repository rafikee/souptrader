"""
Alpaca historical bars downloader

Downloads split-adjusted 1-minute, 5-minute and daily bars for given symbols.
Date ranges are configured in src/backtesting/download_config.yaml and automatically
calculated based on warmup periods needed for indicators.

Stores data as monthly Parquet files under:
"data/backtest_data/{SYMBOL}/1Min/YYYY-MM.parquet",
"data/backtest_data/{SYMBOL}/5Min/YYYY-MM.parquet" and
"data/backtest_data/{SYMBOL}/Daily/YYYY-MM.parquet".

Usage:
  python -m src.backtesting.download_bars --symbols AAPL,MSFT,SMR

Notes:
- Uses Alpaca StockHistoricalDataClient and adjustment="split".
- Saves all trading/non-trading hours (no filtering); timestamps are UTC.
- Skips downloading a month if the target file already exists (idempotent).
- Requires APCA_API_KEY_ID and APCA_API_SECRET_KEY in environment (.env supported via python-dotenv).
- Date ranges configured in src/backtesting/download_config.yaml
"""

import io
import os
import sys
import argparse
import yaml
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any

import pandas as pd
from dotenv import load_dotenv, dotenv_values
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


# Get project root (3 levels up from src/backtesting/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'backtest_data')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
# Config file is in the same directory as this script
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'download_config.yaml')


def load_config() -> Dict[str, Any]:
    """Load download configuration from YAML file."""
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: Config file not found at {CONFIG_FILE}")
        sys.exit(1)
    
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_date_ranges(config: Dict[str, Any]) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Calculate appropriate start/end dates for each timeframe based on config.
    
    Returns dict with keys: 'daily', '5min', '1min'
    Each value is a tuple of (start_date, end_date) as pd.Timestamp
    """
    trading_start = config['trading_start_date']
    trading_end = config['trading_end_date']
    warmup_periods = config['warmup_periods']
    
    # Parse trading_start_date
    start_ts = pd.Timestamp(trading_start, tz="UTC")
    
    # Parse trading_end_date (handle "today" keyword)
    if trading_end.lower() == "today":
        end_ts = pd.Timestamp.now(tz="UTC").normalize()
    else:
        end_ts = pd.Timestamp(trading_end + " 23:59:59", tz="UTC")
    
    # Calculate offsets for warmup periods
    # Daily: 200 trading days ≈ 285 calendar days (assuming ~70% market days)
    daily_offset_days = int(warmup_periods * 1.43)  # 200 * 1.43 ≈ 286 days
    
    # 5Min: 200 periods ≈ 2.5 trading days ≈ 3 calendar days (to be safe)
    fivemin_offset_days = 3
    
    # 1Min: 200 periods ≈ 0.5 trading days ≈ 1 calendar day (to be safe)
    onemin_offset_days = 1
    
    # Calculate start dates
    daily_start = start_ts - pd.Timedelta(days=daily_offset_days)
    fivemin_start = start_ts - pd.Timedelta(days=fivemin_offset_days)
    onemin_start = start_ts - pd.Timedelta(days=onemin_offset_days)
    
    return {
        'daily': (daily_start, end_ts),
        '5min': (fivemin_start, end_ts),
        '1min': (onemin_start, end_ts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Alpaca historical 1Min, 5Min and Daily bars to Parquet")
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
    
    # Load config and calculate date ranges
    config = load_config()
    date_ranges = get_date_ranges(config)
    
    print("=" * 60)
    print("Download Configuration")
    print("=" * 60)
    print(f"Trading Start Date: {config['trading_start_date']}")
    print(f"Trading End Date: {config['trading_end_date']}")
    print(f"Warmup Periods: {config['warmup_periods']}")
    print()
    print("Calculated Download Ranges:")
    print(f"  Daily bars:  {date_ranges['daily'][0].date()} → {date_ranges['daily'][1].date()}")
    print(f"  5Min bars:   {date_ranges['5min'][0].date()} → {date_ranges['5min'][1].date()}")
    print(f"  1Min bars:   {date_ranges['1min'][0].date()} → {date_ranges['1min'][1].date()}")
    print()
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Force: {args.force}")
    print("=" * 60)
    print()
    
    # Process each symbol
    for i, symbol in enumerate(symbols):
        print(f"Processing {symbol} ({i+1}/{len(symbols)})")
        
        # Download 1Min bars
        print(f"  Downloading 1Min bars...")
        onemin_start, onemin_end = date_ranges['1min']
        months_1min = month_range(onemin_start, onemin_end)
        total_months_1min = len(months_1min)
        success_1min = 0
        errors_1min = 0
        
        # Initialize log for this timeframe
        if i == 0:  # First symbol starts new session
            log_bars_output(symbol, "1Min", args.force, [f"Starting 1Min bars download for {symbol}"], is_new_session=True)
        
        for month_idx, (month_start, month_end) in enumerate(months_1min, 1):
            try:
                if download_month_bars(client, symbol, TimeFrame(1, TimeFrameUnit.Minute), "1Min", month_start, month_end, args.force):
                    success_1min += 1
                    status = "✓"
                else:
                    status = "○"
            except Exception as e:
                errors_1min += 1
                status = "✗"
                print(f"    Error downloading {symbol} 1Min {month_start.strftime('%Y-%m')}: {e}")
            
            # Log progress after each month
            pct = (month_idx / total_months_1min * 100) if total_months_1min > 0 else 0
            progress_msg = f"Progress: {month_idx}/{total_months_1min} months ({pct:.1f}%) | {success_1min} success, {errors_1min} errors | Latest: {status} {month_start.strftime('%Y-%m')}"
            log_bars_output(symbol, "1Min", args.force, [progress_msg], is_new_session=False)
        
        pct_1min = (success_1min / total_months_1min * 100) if total_months_1min > 0 else 0
        print(f"    ✓ 1Min: {success_1min}/{total_months_1min} months ({pct_1min:.1f}%)" + (f" | {errors_1min} errors" if errors_1min > 0 else ""))
        
        # Download 5Min bars
        print(f"  Downloading 5Min bars...")
        fivemin_start, fivemin_end = date_ranges['5min']
        months_5min = month_range(fivemin_start, fivemin_end)
        total_months_5min = len(months_5min)
        success_5min = 0
        errors_5min = 0
        
        # Initialize log for this timeframe
        if i == 0:  # First symbol starts new session
            log_bars_output(symbol, "5Min", args.force, [f"Starting 5Min bars download for {symbol}"], is_new_session=True)
        
        for month_idx, (month_start, month_end) in enumerate(months_5min, 1):
            try:
                if download_month_bars(client, symbol, TimeFrame(5, TimeFrameUnit.Minute), "5Min", month_start, month_end, args.force):
                    success_5min += 1
                    status = "✓"
                else:
                    status = "○"
            except Exception as e:
                errors_5min += 1
                status = "✗"
                print(f"    Error downloading {symbol} 5Min {month_start.strftime('%Y-%m')}: {e}")
            
            # Log progress after each month
            pct = (month_idx / total_months_5min * 100) if total_months_5min > 0 else 0
            progress_msg = f"Progress: {month_idx}/{total_months_5min} months ({pct:.1f}%) | {success_5min} success, {errors_5min} errors | Latest: {status} {month_start.strftime('%Y-%m')}"
            log_bars_output(symbol, "5Min", args.force, [progress_msg], is_new_session=False)
        
        pct_5min = (success_5min / total_months_5min * 100) if total_months_5min > 0 else 0
        print(f"    ✓ 5Min: {success_5min}/{total_months_5min} months ({pct_5min:.1f}%)" + (f" | {errors_5min} errors" if errors_5min > 0 else ""))
        
        # Download Daily bars
        print(f"  Downloading Daily bars...")
        daily_start, daily_end = date_ranges['daily']
        months_daily = month_range(daily_start, daily_end)
        total_months_daily = len(months_daily)
        success_daily = 0
        errors_daily = 0
        
        # Initialize log for this timeframe
        if i == 0:  # First symbol starts new session
            log_bars_output(symbol, "Daily", args.force, [f"Starting Daily bars download for {symbol}"], is_new_session=True)
        
        for month_idx, (month_start, month_end) in enumerate(months_daily, 1):
            try:
                if download_month_bars(client, symbol, TimeFrame(1, TimeFrameUnit.Day), "Daily", month_start, month_end, args.force):
                    success_daily += 1
                    status = "✓"
                else:
                    status = "○"
            except Exception as e:
                errors_daily += 1
                status = "✗"
                print(f"    Error downloading {symbol} Daily {month_start.strftime('%Y-%m')}: {e}")
            
            # Log progress after each month
            pct = (month_idx / total_months_daily * 100) if total_months_daily > 0 else 0
            progress_msg = f"Progress: {month_idx}/{total_months_daily} months ({pct:.1f}%) | {success_daily} success, {errors_daily} errors | Latest: {status} {month_start.strftime('%Y-%m')}"
            log_bars_output(symbol, "Daily", args.force, [progress_msg], is_new_session=False)
        
        pct_daily = (success_daily / total_months_daily * 100) if total_months_daily > 0 else 0
        print(f"    ✓ Daily: {success_daily}/{total_months_daily} months ({pct_daily:.1f}%)" + (f" | {errors_daily} errors" if errors_daily > 0 else ""))
        print(f"  Completed {symbol}: 1Min({success_1min}/{total_months_1min}) 5Min({success_5min}/{total_months_5min}) Daily({success_daily}/{total_months_daily})")
        
        # Log final summary for each timeframe
        log_1min_msg = f"FINAL SUMMARY: {success_1min}/{total_months_1min} months ({pct_1min:.1f}%)" + (f" | {errors_1min} errors" if errors_1min > 0 else "")
        log_5min_msg = f"FINAL SUMMARY: {success_5min}/{total_months_5min} months ({pct_5min:.1f}%)" + (f" | {errors_5min} errors" if errors_5min > 0 else "")
        log_daily_msg = f"FINAL SUMMARY: {success_daily}/{total_months_daily} months ({pct_daily:.1f}%)" + (f" | {errors_daily} errors" if errors_daily > 0 else "")
        log_bars_output(symbol, "1Min", args.force, [log_1min_msg], is_new_session=False)
        log_bars_output(symbol, "5Min", args.force, [log_5min_msg], is_new_session=False)
        log_bars_output(symbol, "Daily", args.force, [log_daily_msg], is_new_session=False)
        print()
    
    print("Download completed!")


if __name__ == "__main__":
    main()