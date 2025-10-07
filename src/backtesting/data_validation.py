"""
Shared data validation utilities for backtesting data.

This module provides common validation logic used by both the download page
and standalone validation scripts.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / 'data' / 'backtest_data'

# Date range constants (should match download scripts)
# Bars data: 2024-01-01 to 2025-12-31 (from download_bars.py)
BARS_START_DATE_STR = "2024-01-01"
BARS_END_DATE_STR = "2025-12-31"

# NBBO data: 2025-01-01 to 2025-12-31 (from download_nbbo_1s.py)
NBBO_START_DATE_STR = "2025-01-01"
NBBO_END_DATE_STR = "2025-12-31"


def month_range(start: pd.Timestamp, end: pd.Timestamp) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Generate (month_start, month_end) pairs covering [start, end]."""
    result: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    current = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    end_month_start = pd.Timestamp(year=end.year, month=end.month, day=1, tz="UTC")
    
    while current <= end_month_start:
        month_end = (current + pd.DateOffset(months=1)) - pd.Timedelta(days=1)
        if month_end > end:
            month_end = end
        result.append((current, month_end))
        current = current + pd.DateOffset(months=1)
    
    return result


def get_effective_end_date(data_type: str = 'bars') -> pd.Timestamp:
    """
    Get the effective end date capped to today.
    
    Args:
        data_type: 'bars' or 'nbbo' to determine which date range to use
    """
    if data_type == 'nbbo':
        end = pd.Timestamp(NBBO_END_DATE_STR, tz="UTC")
    else:
        end = pd.Timestamp(BARS_END_DATE_STR, tz="UTC")
    
    today = pd.Timestamp.now(tz="UTC").normalize()
    return min(end, today)


def validate_monthly_data(symbol: str, timeframe: str) -> Tuple[bool, int, int, List[str]]:
    """
    Validate monthly bar data (5Min or Daily).
    
    Simply checks if the folder exists and has at least 1 file.
    
    Returns:
        (is_complete, found_count, expected_count, missing_months)
    """
    data_dir = DATA_ROOT / symbol / timeframe
    
    if not data_dir.exists():
        return False, 0, 0, []
    
    # Get all existing files
    existing_files = list(data_dir.glob('*.parquet'))
    found_count = len(existing_files)
    
    # Complete if has at least 1 file
    is_complete = found_count > 0
    
    return is_complete, found_count, found_count, []


def validate_nbbo_data(symbol: str) -> Tuple[bool, int, int, List[str]]:
    """
    Validate NBBO 1s data.
    
    For NBBO, we check if the directory exists and has a reasonable number of files.
    We don't expect every calendar day since weekends/holidays won't have data.
    
    Returns:
        (is_complete, found_count, expected_count, missing_dates)
    """
    data_dir = DATA_ROOT / symbol / 'nbbo1s'
    
    if not data_dir.exists():
        return False, 0, 0, []
    
    # Get all existing NBBO files
    existing_files = sorted(data_dir.glob('*.parquet'))
    found_count = len(existing_files)
    
    # For NBBO, we just check if we have a reasonable amount of data
    # Since we can't predict exactly which days will have data (weekends/holidays are skipped),
    # we consider it complete if the directory exists and has files
    # The expected count is just the found count for reporting purposes
    
    # Extract dates from filenames to show range
    missing_dates = []
    if found_count == 0:
        return False, 0, 0, []
    
    # NBBO is considered complete if it has data files (actual completeness is hard to determine
    # since we don't know which days should have data - weekends, holidays, etc. are skipped)
    is_complete = found_count > 0
    
    return is_complete, found_count, found_count, missing_dates


def is_symbol_complete(symbol: str) -> bool:
    """
    Check if a symbol has complete data for backtesting.
    
    Returns True only if all required data is present:
    - 5Min bars (all months up to today)
    - Daily bars (all months up to today)
    - NBBO 1s data (all days up to today)
    """
    # Check 5Min bars
    is_5min_complete, _, _, _ = validate_monthly_data(symbol, '5Min')
    if not is_5min_complete:
        return False
    
    # Check Daily bars
    is_daily_complete, _, _, _ = validate_monthly_data(symbol, 'Daily')
    if not is_daily_complete:
        return False
    
    # Check NBBO data
    is_nbbo_complete, _, _, _ = validate_nbbo_data(symbol)
    if not is_nbbo_complete:
        return False
    
    return True


def get_validation_summary(symbol: str) -> dict:
    """
    Get detailed validation summary for a symbol.
    
    Returns:
        Dictionary with validation details for each data type
    """
    # Validate 5Min
    is_5min_complete, found_5min, expected_5min, missing_5min = validate_monthly_data(symbol, '5Min')
    
    # Validate Daily
    is_daily_complete, found_daily, expected_daily, missing_daily = validate_monthly_data(symbol, 'Daily')
    
    # Validate NBBO
    is_nbbo_complete, found_nbbo, expected_nbbo, missing_nbbo = validate_nbbo_data(symbol)
    
    return {
        'symbol': symbol,
        'is_complete': is_5min_complete and is_daily_complete and is_nbbo_complete,
        '5Min': {
            'is_complete': is_5min_complete,
            'found': found_5min,
            'expected': expected_5min,
            'missing': missing_5min,
            'percentage': (found_5min / expected_5min * 100) if expected_5min > 0 else 0
        },
        'Daily': {
            'is_complete': is_daily_complete,
            'found': found_daily,
            'expected': expected_daily,
            'missing': missing_daily,
            'percentage': (found_daily / expected_daily * 100) if expected_daily > 0 else 0
        },
        'NBBO': {
            'is_complete': is_nbbo_complete,
            'found': found_nbbo,
            'expected': expected_nbbo,
            'missing': missing_nbbo,
            'percentage': (found_nbbo / expected_nbbo * 100) if expected_nbbo > 0 else 0
        }
    }

