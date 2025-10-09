"""
Data validation script

Validates that the required data structure exists for backtesting.
Checks for 1Min, 5Min, Daily, and NBBO data for each ticker with detailed reporting.

Usage:
  python -m src.backtesting.validate_data
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Import shared validation utilities
from src.backtesting.data_validation import (
    get_validation_summary, 
    is_symbol_complete,
    DATA_ROOT,
    BARS_START_DATE_STR,
    BARS_END_DATE_STR,
    NBBO_START_DATE_STR,
    NBBO_END_DATE_STR,
    get_effective_end_date
)

def validate_ticker_data(ticker):
    """Validate data structure for a single ticker with detailed reporting"""
    print(f"\n{'='*60}")
    print(f"Validating: {ticker}")
    print(f"{'='*60}")
    
    # Get detailed validation summary
    summary = get_validation_summary(ticker)
    
    # Show effective date ranges
    bars_end = get_effective_end_date('bars')
    nbbo_end = get_effective_end_date('nbbo')
    print(f"Bars date range: {BARS_START_DATE_STR} to {bars_end.date()} (capped to today)")
    print(f"NBBO date range: {NBBO_START_DATE_STR} to {nbbo_end.date()} (capped to today)")
    print()
    
    # 1Min bars
    info_1min = summary['1Min']
    status_1min = "✅" if info_1min['is_complete'] else "❌"
    print(f"{status_1min} 1Min bars: {info_1min['found']} months")
    if not info_1min['is_complete'] and info_1min['missing']:
        missing_display = info_1min['missing'][:5]  # Show first 5
        more_count = len(info_1min['missing']) - 5
        print(f"   Missing: {', '.join(missing_display)}" + (f" ... and {more_count} more" if more_count > 0 else ""))
    
    # 5Min bars
    info_5min = summary['5Min']
    status_5min = "✅" if info_5min['is_complete'] else "❌"
    print(f"{status_5min} 5Min bars: {info_5min['found']} months")
    if not info_5min['is_complete'] and info_5min['missing']:
        missing_display = info_5min['missing'][:5]  # Show first 5
        more_count = len(info_5min['missing']) - 5
        print(f"   Missing: {', '.join(missing_display)}" + (f" ... and {more_count} more" if more_count > 0 else ""))
    
    # Daily bars
    info_daily = summary['Daily']
    status_daily = "✅" if info_daily['is_complete'] else "❌"
    print(f"{status_daily} Daily bars: {info_daily['found']} months")
    if not info_daily['is_complete'] and info_daily['missing']:
        missing_display = info_daily['missing'][:5]
        more_count = len(info_daily['missing']) - 5
        print(f"   Missing: {', '.join(missing_display)}" + (f" ... and {more_count} more" if more_count > 0 else ""))
    
    # NBBO data
    info_nbbo = summary['NBBO']
    status_nbbo = "✅" if info_nbbo['is_complete'] else "❌"
    print(f"{status_nbbo} NBBO 1s: {info_nbbo['found']} days")
    if not info_nbbo['is_complete'] and info_nbbo['missing']:
        missing_display = info_nbbo['missing'][:5]
        more_count = len(info_nbbo['missing']) - 5
        print(f"   Missing: {', '.join(missing_display)}" + (f" ... and {more_count} more" if more_count > 0 else ""))
    
    return summary['is_complete']

def main():
    """Main validation function"""
    print("\n" + "="*60)
    print("DATA VALIDATION REPORT")
    print("="*60)
    print(f"Data directory: {DATA_ROOT}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Bars date range: {BARS_START_DATE_STR} to {get_effective_end_date('bars').date()} (capped to today)")
    print(f"NBBO date range: {NBBO_START_DATE_STR} to {get_effective_end_date('nbbo').date()} (capped to today)")
    print("="*60)
    
    if not DATA_ROOT.exists():
        print(f"\n❌ Data directory not found: {DATA_ROOT}")
        sys.exit(1)
    
    # Get all ticker directories
    ticker_dirs = [d for d in DATA_ROOT.iterdir() if d.is_dir()]
    
    if not ticker_dirs:
        print("\n❌ No ticker directories found")
        sys.exit(1)
    
    print(f"\nFound {len(ticker_dirs)} ticker directories")
    
    # Validate each ticker
    valid_tickers = []
    invalid_tickers = []
    
    for ticker_dir in sorted(ticker_dirs):
        ticker = ticker_dir.name
        if validate_ticker_data(ticker):
            valid_tickers.append(ticker)
        else:
            invalid_tickers.append(ticker)
    
    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total tickers: {len(ticker_dirs)}")
    print(f"✅ Complete: {len(valid_tickers)}")
    print(f"❌ Incomplete: {len(invalid_tickers)}")
    
    if valid_tickers:
        print(f"\n✅ Complete tickers:")
        for ticker in valid_tickers:
            print(f"   • {ticker}")
    
    if invalid_tickers:
        print(f"\n❌ Incomplete tickers:")
        for ticker in invalid_tickers:
            print(f"   • {ticker}")
    
    print(f"\n{'='*60}")
    if invalid_tickers:
        print("⚠️  Some tickers have incomplete data. Run downloads to complete them.")
        sys.exit(1)
    else:
        print("✅ All tickers validated successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
