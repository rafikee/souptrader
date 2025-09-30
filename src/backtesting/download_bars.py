"""
Alpaca historical bars downloader

Downloads split-adjusted 1-minute and 5-minute bars for given symbols between
2024-09-01 and 2025-08-31 and stores them as monthly Parquet files under
"data/backtest_data/{SYMBOL}/{TIMEFRAME}/YYYY-MM.parquet".

Usage:
  python -m src.data.download_bars --symbols AAPL,MSFT,SMR

Notes:
- Uses Alpaca StockHistoricalDataClient and adjustment="split".
- Saves all trading/non-trading hours (no filtering); timestamps are UTC.
- Skips downloading a month if the target file already exists (idempotent).
- Requires MY_API_KEY in environment (.env supported via python-dotenv).
"""

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
START_DATE_STR = "2024-09-01"
END_DATE_STR = "2025-08-31"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Alpaca historical bars to Parquet")
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


def ensure_app_api_key() -> None:
    """Require presence of MY_API_KEY in environment and validate against .env value."""
    # Load .env but do not override provided environment
    load_dotenv()
    provided_key = os.getenv("MY_API_KEY")
    if not provided_key:
        print("ERROR: MY_API_KEY not provided. Set it in the environment.", file=sys.stderr)
        sys.exit(1)
    # Read .env values explicitly to get the expected key
    env_vals = dotenv_values()
    expected_key = env_vals.get("MY_API_KEY")
    if not expected_key:
        print("ERROR: MY_API_KEY not set in .env on the server.", file=sys.stderr)
        sys.exit(1)
    if provided_key != expected_key:
        print("ERROR: Invalid MY_API_KEY.", file=sys.stderr)
        sys.exit(1)


def get_client() -> StockHistoricalDataClient:
    load_dotenv()
    api_key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        print("ERROR: Alpaca API keys not found in environment (.env)")
        sys.exit(1)
    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def fetch_and_write_month(
    client: StockHistoricalDataClient,
    symbol: str,
    timeframe: TimeFrame,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
) -> None:
    timeframe_str = f"{timeframe.amount}{'Min' if timeframe.unit == TimeFrameUnit.Minute else timeframe.unit.value}"
    out_dir = os.path.join(DATA_ROOT, symbol.upper(), timeframe_str)
    ensure_dir(out_dir)
    file_name = f"{month_start.strftime('%Y-%m')}.parquet"
    out_path = os.path.join(out_dir, file_name)

    if os.path.exists(out_path):
        print(f"Exists, skipping: {out_path}")
        return

    request = StockBarsRequest(
        symbol_or_symbols=[symbol.upper()],
        timeframe=timeframe,
        start=month_start.to_pydatetime(),
        end=month_end.to_pydatetime(),
        adjustment="split",
    )

    print(f"Downloading {symbol} {timeframe_str} {month_start.date()} → {month_end.date()}")
    bars = client.get_stock_bars(request)
    if not bars or not hasattr(bars, "df") or bars.df.empty:
        # Write empty file to mark as fetched and avoid re-hitting API
        pd.DataFrame().to_parquet(out_path)
        print(f"No data returned; wrote empty file: {out_path}")
        return

    df = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    if "timestamp" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "timestamp"})

    # Ensure UTC tz-aware timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Sort for consistency
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Save Parquet (Snappy)
    df.to_parquet(out_path, index=False)
    print(f"Wrote: {out_path} ({len(df)} rows)")


def main() -> None:
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    # Require app-level API key
    ensure_app_api_key()

    start = pd.Timestamp(START_DATE_STR, tz="UTC")
    # Inclusive end date: set to end of day UTC
    end = pd.Timestamp(END_DATE_STR + " 23:59:59", tz="UTC")

    client = get_client()

    timeframes = [
        TimeFrame(1, TimeFrameUnit.Minute),
        TimeFrame(5, TimeFrameUnit.Minute),
    ]

    months = month_range(start, end)

    for symbol in symbols:
        sym = symbol.upper()
        # Pre-scan: if all expected monthly files exist for all timeframes, skip this symbol entirely
        all_complete = True
        for timeframe in timeframes:
            timeframe_str = f"{timeframe.amount}{'Min' if timeframe.unit == TimeFrameUnit.Minute else timeframe.unit.value}"
            out_dir = os.path.join(DATA_ROOT, sym, timeframe_str)
            expected_files = [os.path.join(out_dir, f"{m_start.strftime('%Y-%m')}.parquet") for m_start, _ in months]
            # If any expected file missing, this timeframe is incomplete
            if not all(os.path.exists(p) for p in expected_files):
                all_complete = False
                break

        if all_complete and not args.force:
            print(f"All monthly bars exist for {sym} across all timeframes; skipping symbol.")
            continue

        for timeframe in timeframes:
            for month_start, month_end in months:
                timeframe_str = f"{timeframe.amount}{'Min' if timeframe.unit == TimeFrameUnit.Minute else timeframe.unit.value}"
                out_dir = os.path.join(DATA_ROOT, sym, timeframe_str)
                file_name = f"{month_start.strftime('%Y-%m')}.parquet"
                out_path = os.path.join(out_dir, file_name)
                if os.path.exists(out_path) and args.force:
                    try:
                        os.remove(out_path)
                    except OSError:
                        pass
                fetch_and_write_month(client, sym, timeframe, month_start, month_end)


if __name__ == "__main__":
    main()


