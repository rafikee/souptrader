"""
Intraday Trading Strategy Backtester

This script backtests intraday trading strategies using Alpaca data.
- Entry signals based on 5-minute candles
- Exit signals based on 1-minute candles
- Long-only positions, no shorts
- One trade per day maximum
- Position sizing: ~$100k (never exceed, whole shares only)

Author: SoupTrader
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import pytz
import exchange_calendars as xcals

# Timezone configuration
UTC = pytz.UTC
ET = pytz.timezone('US/Eastern')

# Exchange calendar for NYSE
NYSE = xcals.get_calendar('XNYS')

# Load environment variables
load_dotenv()

# Debug: Check if environment variables are loaded
api_key = os.getenv("APCA_API_KEY_ID")
secret_key = os.getenv("APCA_API_SECRET_KEY")

print(f"API Key loaded: {'Yes' if api_key else 'No'}")
print(f"Secret Key loaded: {'Yes' if secret_key else 'No'}")

if not api_key or not secret_key:
    print("ERROR: Alpaca API keys not found in environment variables")
    print("Make sure your .env file contains ALPACA_API_KEY and ALPACA_SECRET_KEY")
    exit(1)

# =============================================================================
# CONFIGURATION - MODIFY THESE VALUES
# =============================================================================

# Trading parameters
SYMBOL = "SMR"                    # Stock symbol to backtest
POSITION_SIZE = 100000            # Target position size in dollars (never exceed)
PROFIT_TARGET = 0.01              # 1% profit target
STOP_LOSS = -0.0025                # -0.25% stop loss
BACKTEST_DAYS = 365               # Number of days to backtest (1 week for testing)

# Strategy Selection
STRATEGY = "gap_breakout"          # Options: "sma_alignment" or "gap_breakout"

# Debug Configuration
DEBUG_ENABLED = False             # Set to True to enable debug output
DEBUG_DATE = "2025-08-05"         # Date to debug (YYYY-MM-DD format)
DEBUG_TIME = "09:40"              # Time to debug (HH:MM format, 24-hour)

# Timeframes
ENTRY_TIMEFRAME = TimeFrame(5, TimeFrameUnit.Minute)    # 5-minute for entry signals
EXIT_TIMEFRAME = TimeFrame(1, TimeFrameUnit.Minute)     # 1-minute for exit signals

# =============================================================================
# ALPACA SETUP
# =============================================================================

# Initialize Alpaca client
client = StockHistoricalDataClient(
    api_key=api_key,
    secret_key=secret_key
)

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Setup logging with file clearing"""
    # Clear the log file
    with open('backtest_results.log', 'w') as f:
        f.write('')
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('backtest_results.log'),
            logging.StreamHandler()
        ]
    )

# Initialize logging
setup_logging()

# =============================================================================
# DATA FETCHING FUNCTIONS
# =============================================================================

def fetch_5min_data(symbol, start_date, end_date):
    """
    Fetch 5-minute data for entry signals.
    
    Args:
        symbol (str): Stock symbol
        start_date (datetime): Start date
        end_date (datetime): End date
    
    Returns:
        pd.DataFrame: 5-minute OHLCV data
    """
    try:
        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=ENTRY_TIMEFRAME,
            start=start_date,
            end=end_date,
            adjustment='raw'
        )
        
        bars = client.get_stock_bars(request_params)
        
        if not bars or not hasattr(bars, 'df') or bars.df.empty:
            logging.warning(f"No 5-minute data found for {symbol}")
            return pd.DataFrame()
        
        df = bars.df
        
        # If multi-index (symbol, timestamp), reset to get columns
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        
        # Ensure we have the right column names
        if 'timestamp' not in df.columns and 'time' in df.columns:
            df = df.rename(columns={'time': 'timestamp'})
        
        # Convert timestamp and filter for regular trading hours immediately
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Apply regular trading hours filter using exchange calendar
        df['is_trading_hours'] = df['timestamp'].apply(is_regular_trading_hours)
        df = df[df['is_trading_hours']].copy()
        df = df.drop('is_trading_hours', axis=1)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logging.info(f"Fetched {len(df)} 5-minute candles for {symbol}")
        return df
        
    except Exception as e:
        logging.error(f"Error fetching 5-minute data for {symbol}: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return pd.DataFrame()

def fetch_1min_data(symbol, start_time, end_time):
    """
    Fetch 1-minute data for exit monitoring.
    Only fetch data for the specific trade timeframe.
    
    Args:
        symbol (str): Stock symbol
        start_time (datetime): Trade entry time
        end_time (datetime): Trade exit time or market close
    
    Returns:
        pd.DataFrame: 1-minute OHLCV data
    """
    try:
        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=EXIT_TIMEFRAME,
            start=start_time,
            end=end_time,
            adjustment='raw'
        )
        
        bars = client.get_stock_bars(request_params)
        
        if not bars or not hasattr(bars, 'df') or bars.df.empty:
            logging.warning(f"No 1-minute data found for {symbol}")
            return pd.DataFrame()
        
        df = bars.df
        
        # If multi-index (symbol, timestamp), reset to get columns
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        
        # Ensure we have the right column names
        if 'timestamp' not in df.columns and 'time' in df.columns:
            df = df.rename(columns={'time': 'timestamp'})
        
        # Convert timestamp and filter for regular trading hours immediately
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Apply regular trading hours filter using exchange calendar
        df['is_trading_hours'] = df['timestamp'].apply(is_regular_trading_hours)
        df = df[df['is_trading_hours']].copy()
        df = df.drop('is_trading_hours', axis=1)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logging.info(f"Fetched {len(df)} 1-minute candles for {symbol}")
        return df
        
    except Exception as e:
        logging.error(f"Error fetching 1-minute data for {symbol}: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return pd.DataFrame()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def to_et_time(utc_time):
    """
    Convert UTC time to Eastern Time for display.
    
    Args:
        utc_time (datetime): UTC datetime object
    
    Returns:
        str: Formatted ET time string
    """
    if utc_time.tzinfo is None:
        utc_time = UTC.localize(utc_time)
    et_time = utc_time.astimezone(ET)
    return et_time.strftime('%Y-%m-%d %H:%M:%S %Z')

def is_regular_trading_hours(utc_time):
    """
    Check if a UTC timestamp falls during regular NYSE trading hours.
    Uses exchange calendar to determine exact session times for each day.
    
    Args:
        utc_time (datetime): UTC datetime object
    
    Returns:
        bool: True if during regular trading hours
    """
    if utc_time.tzinfo is None:
        utc_time = UTC.localize(utc_time)
    
    # Convert to Eastern Time to get the trading date
    et_time = utc_time.astimezone(ET)
    
    # Check if it's a trading day
    if not NYSE.is_session(et_time.date()):
        return False
    
    # Get the exact session start and end times for this trading day
    session_start = NYSE.session_open(et_time.date())
    session_end = NYSE.session_close(et_time.date())
    
    # Check if the time falls within the session
    return session_start <= utc_time <= session_end

# =============================================================================
# STRATEGY FUNCTIONS
# =============================================================================

def calculate_smma(data, period, column='close'):
    """
    Calculate Smoothed Moving Average (SMMA).
    
    Args:
        data (pd.DataFrame): OHLCV data
        period (int): Period for SMMA calculation
        column (str): Column to calculate SMMA on
    
    Returns:
        pd.Series: SMMA values
    """
    smma = pd.Series(index=data.index, dtype=float)
    smma.iloc[0] = data[column].iloc[0]
    
    for i in range(1, len(data)):
        if i < period:
            smma.iloc[i] = (smma.iloc[i-1] * i + data[column].iloc[i]) / (i + 1)
        else:
            smma.iloc[i] = (smma.iloc[i-1] * (period - 1) + data[column].iloc[i]) / period
    
    return smma

def calculate_body_size(data):
    """
    Calculate the body size of each candle.
    
    Args:
        data (pd.DataFrame): OHLCV data
    
    Returns:
        pd.Series: Body size (absolute value of close - open)
    """
    return abs(data['close'] - data['open'])

def calculate_vwap(data):
    """
    Calculate Volume Weighted Average Price (VWAP) for each candle.
    
    Args:
        data (pd.DataFrame): OHLCV data
    
    Returns:
        pd.Series: VWAP values
    """
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
    return vwap

def calculate_rsi(data, period=14):
    """
    Calculate Relative Strength Index (RSI) for each candle.
    
    Args:
        data (pd.DataFrame): OHLCV data
        period (int): RSI period (default 14)
    
    Returns:
        pd.Series: RSI values
    """
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def is_green_candle(data, index):
    """
    Check if a candle is green (close > open).
    
    Args:
        data (pd.DataFrame): OHLCV data
        index (int): Index of the candle
    
    Returns:
        bool: True if candle is green
    """
    return data.iloc[index]['close'] > data.iloc[index]['open']

def debug_specific_datetime(data_5min, current_index, smma_21, smma_50, smma_200, body_sizes, daily_smma_21, daily_smma_50, daily_smma_200):
    """
    Debug function to check why a specific date/time didn't trigger a trade.
    """
    # Skip debug if disabled
    if not DEBUG_ENABLED:
        return False
        
    current_row = data_5min.iloc[current_index]
    current_timestamp = current_row['timestamp']
    
    # Convert to Eastern Time for comparison
    et_timestamp = current_timestamp.tz_convert('US/Eastern')
    
    # Check if this matches our target date/time
    if (et_timestamp.strftime('%Y-%m-%d') == DEBUG_DATE and 
        et_timestamp.strftime('%H:%M') == DEBUG_TIME):
        
        logging.info(f"=== DEBUGGING {DEBUG_DATE} {DEBUG_TIME} ===")
        logging.info(f"Current candle: {current_timestamp} - Open: ${current_row['open']:.2f}, Close: ${current_row['close']:.2f}")
        
        prev_index = current_index - 1
        prev_row = data_5min.iloc[prev_index]
        logging.info(f"Previous candle: {prev_row['timestamp']} - Open: ${prev_row['open']:.2f}, Close: ${prev_row['close']:.2f}")
        
        # Check each criterion
        logging.info("Checking criteria:")
        
        # Criterion 1a: Previous candle above SMMA
        smma_value = smma_21.iloc[prev_index]
        logging.info(f"  Criterion 1a - Previous close (${prev_row['close']:.2f}) > SMMA (${smma_value:.2f}): {prev_row['close'] > smma_value}")
        
        # Criterion 1b: Previous candle green
        prev_green = prev_row['close'] > prev_row['open']
        logging.info(f"  Criterion 1b - Previous candle green: {prev_green}")
        
        # Criterion 1c: SMA alignment (5-minute)
        smma_21_val = smma_21.iloc[prev_index]
        smma_50_val = smma_50.iloc[prev_index]
        smma_200_val = smma_200.iloc[prev_index]
        sma_alignment = smma_21_val > smma_50_val > smma_200_val
        logging.info(f"  Criterion 1c - 5-min SMA alignment (21: ${smma_21_val:.2f} > 50: ${smma_50_val:.2f} > 200: ${smma_200_val:.2f}): {sma_alignment}")
        
        # Criterion 1d: Daily SMA alignment
        daily_smma_21_val = daily_smma_21.iloc[prev_index]
        daily_smma_50_val = daily_smma_50.iloc[prev_index]
        daily_smma_200_val = daily_smma_200.iloc[prev_index]
        daily_sma_alignment = daily_smma_21_val > daily_smma_50_val > daily_smma_200_val
        logging.info(f"  Criterion 1d - Daily SMA alignment (21: ${daily_smma_21_val:.2f} > 50: ${daily_smma_50_val:.2f} > 200: ${daily_smma_200_val:.2f}): {daily_sma_alignment}")
        
        # Criterion 2: Body size
        avg_body_size_3 = body_sizes.iloc[prev_index-3:prev_index].mean()
        prev_body_size = body_sizes.iloc[prev_index]
        body_size_ok = prev_body_size >= (2 * avg_body_size_3)
        logging.info(f"  Criterion 2 - Body size (${prev_body_size:.2f}) >= 2X avg (${2*avg_body_size_3:.2f}): {body_size_ok}")
        
        # Criterion 3: Current candle green
        current_green = current_row['close'] > current_row['open']
        logging.info(f"  Criterion 3 - Current candle green: {current_green}")
        
        # Criterion 4: Next candle gap
        if current_index + 1 < len(data_5min):
            next_row = data_5min.iloc[current_index + 1]
            next_open = next_row['open']
            current_close = current_row['close']
            gap_pct = (next_open - current_close) / current_close
            gap_ok = gap_pct >= -0.001  # No more than 0.1% gap down
            logging.info(f"  Criterion 4 - Gap % ({gap_pct*100:.2f}%) >= -0.1%: {gap_ok}")
            logging.info(f"    Gap size: ${next_open - current_close:.2f} ({gap_pct*100:.2f}%)")
        else:
            logging.info(f"  Criterion 4 - No next candle available")
        
        logging.info("=== END DEBUG ===")
        return True  # We found and debugged the target
    
    return False  # Not our target date/time

def debug_gap_breakout(data_5min, current_index, vwap, rsi, data_1min):
    """
    Debug function to check why a specific date/time didn't trigger a gap breakout trade.
    """
    # Skip debug if disabled
    if not DEBUG_ENABLED:
        return False
    
    current_row = data_5min.iloc[current_index]
    current_timestamp = current_row['timestamp']
    et_timestamp = current_timestamp.tz_convert('US/Eastern')
    
    # Check if this matches our debug target
    if (et_timestamp.strftime('%Y-%m-%d') == DEBUG_DATE and 
        et_timestamp.strftime('%H:%M') == DEBUG_TIME):
        
        logging.info("=== GAP BREAKOUT DEBUG ===")
        logging.info(f"Target: {DEBUG_DATE} {DEBUG_TIME}")
        logging.info(f"Current: {et_timestamp.strftime('%Y-%m-%d %H:%M')}")
        logging.info(f"Price: ${current_row['close']:.2f}")
        
        # Check each criterion
        current_time = current_row['timestamp']
        et_time = current_time.tz_convert('US/Eastern')
        
        # Criterion 1: Gap up check
        prev_day_close = None
        for i in range(current_index - 1, -1, -1):
            prev_candle = data_5min.iloc[i]
            prev_et_time = prev_candle['timestamp'].tz_convert('US/Eastern')
            if prev_et_time.date() < et_time.date():
                prev_day_close = prev_candle['close']
                break
        
        if prev_day_close:
            gap_pct = (current_row['open'] - prev_day_close) / prev_day_close
            logging.info(f"  Criterion 1 - Gap up (prev close: ${prev_day_close:.2f}, gap: {gap_pct:.2%}): {gap_pct >= 0.01}")
        else:
            logging.info(f"  Criterion 1 - Gap up: No previous day close found")
        
        # Criterion 2: Time check (within first hour)
        time_ok = et_time.hour == 9
        logging.info(f"  Criterion 2 - Time check (9:30-10:30 AM ET): {time_ok}")
        
        # Criterion 3: Opening range breakout
        day_start_index = None
        for j in range(current_index, -1, -1):
            candle_time = data_5min.iloc[j]['timestamp'].tz_convert('US/Eastern')
            if candle_time.date() < et_time.date():
                day_start_index = j + 1
                break
        
        if day_start_index and day_start_index + 1 < current_index:
            first_candle = data_5min.iloc[day_start_index]
            second_candle = data_5min.iloc[day_start_index + 1]
            
            # Check if both opening candles are green
            both_green = (first_candle['close'] > first_candle['open'] and 
                         second_candle['close'] > second_candle['open'])
            logging.info(f"  Criterion 3a - Both opening candles green: {both_green}")
            
            if both_green:
                opening_range_high = max(first_candle['high'], second_candle['high'])
                breakout_ok = current_row['close'] > opening_range_high
                logging.info(f"  Criterion 3b - Opening range breakout (high: ${opening_range_high:.2f}): {breakout_ok}")
            else:
                logging.info(f"  Criterion 3b - Opening range breakout: Skipped (not both green)")
        else:
            logging.info(f"  Criterion 3 - Opening range breakout: Not enough data")
        
        # Criterion 4: Price above VWAP
        vwap_ok = current_row['close'] > vwap.iloc[current_index]
        logging.info(f"  Criterion 4 - Above VWAP (VWAP: ${vwap.iloc[current_index]:.2f}): {vwap_ok}")
        
        # Criterion 5: RSI > 55
        rsi_ok = rsi.iloc[current_index] > 55
        logging.info(f"  Criterion 5 - RSI > 55 (RSI: {rsi.iloc[current_index]:.1f}): {rsi_ok}")
        
        # Criterion 6: Removed - no extension limit
        logging.info(f"  Criterion 6 - Extension limit: REMOVED")
        
        logging.info("=== END GAP BREAKOUT DEBUG ===")
        return True  # We found and debugged the target
    
    return False  # Not our target date/time

def entry_signal_gap_breakout(data_5min, current_index, vwap, rsi, data_1min):
    """
    Determine if we should enter a long position based on gap breakout strategy.
    
    Entry criteria:
    - Stock must gap up at least +1% from previous close
    - Opening range breakout (first 2 x 5-minute candles)
    - Entry when price breaks above opening range high with volume confirmation (1.5x avg)
    - Price must be above VWAP
    - RSI > 55
    - Skip if stock is too extended (>3% above VWAP at entry)
    
    Args:
        data_5min (pd.DataFrame): 5-minute OHLCV data
        current_index (int): Current candle index
        vwap (pd.Series): Pre-calculated VWAP
        rsi (pd.Series): Pre-calculated RSI
        data_1min (pd.DataFrame): 1-minute OHLCV data for volume confirmation
    
    Returns:
        bool: True if we should enter a position
    """
    # Need at least 2 candles for opening range
    if current_index < 2:
        return False
    
    current_row = data_5min.iloc[current_index]
    current_time = current_row['timestamp']
    
    # Check if this is within the first hour of trading (9:30-10:30 AM ET)
    et_time = current_time.tz_convert('US/Eastern')
    if et_time.hour != 9:  # Only first hour (9:30-10:30 AM ET)
        return False
    
    # Get previous day's close (assuming data is sorted by timestamp)
    prev_day_close = None
    for i in range(current_index - 1, -1, -1):
        prev_candle = data_5min.iloc[i]
        prev_et_time = prev_candle['timestamp'].tz_convert('US/Eastern')
        if prev_et_time.date() < et_time.date():
            prev_day_close = prev_candle['close']
            break
    
    if prev_day_close is None:
        return False
    
    # Criterion 1: Gap up at least +1%
    gap_pct = (current_row['open'] - prev_day_close) / prev_day_close
    if gap_pct < 0.01:  # Less than 1% gap up
        return False
    
    # Criterion 2: Opening range breakout
    # Opening range = first 2 x 5-minute candles of current day (9:30-9:40 AM ET)
    # Find the first two candles of the current day
    day_start_index = None
    for j in range(current_index, -1, -1):
        candle_time = data_5min.iloc[j]['timestamp'].tz_convert('US/Eastern')
        if candle_time.date() < et_time.date():
            day_start_index = j + 1
            break
    
    if day_start_index is None or day_start_index + 1 >= current_index:
        return False
    
    # Check that both opening range candles are green
    first_candle = data_5min.iloc[day_start_index]
    second_candle = data_5min.iloc[day_start_index + 1]
    
    if (first_candle['close'] <= first_candle['open'] or 
        second_candle['close'] <= second_candle['open']):
        return False  # One or both opening candles are red
    
    # Opening range = first 2 candles of the day
    opening_range_high = max(first_candle['high'], second_candle['high'])
    
    # Check if current price breaks above opening range high
    if current_row['close'] <= opening_range_high:
        return False
    
    # Criterion 3: Volume confirmation (1.5x average 1-minute volume)
    # Get 1-minute data for current 5-minute period
    current_5min_start = current_time
    current_5min_end = current_time + pd.Timedelta(minutes=5)
    
    # Filter 1-minute data for this 5-minute period
    mask = (data_1min['timestamp'] >= current_5min_start) & (data_1min['timestamp'] < current_5min_end)
    current_1min_data = data_1min[mask]
    
    if current_1min_data.empty:
        return False
    
    current_volume = current_1min_data['volume'].sum()
    
    # Calculate average 1-minute volume (use last 20 days of data)
    avg_volume = data_1min['volume'].rolling(window=20*78).mean().iloc[-1]  # 20 days * ~78 5-min candles
    if pd.isna(avg_volume):
        return False
    
    if current_volume < 1.5 * avg_volume:
        return False
    
    # Criterion 4: Price above VWAP
    if current_row['close'] <= vwap.iloc[current_index]:
        return False
    
    # Criterion 5: RSI > 55
    if rsi.iloc[current_index] <= 55:
        return False
    
    # Criterion 6: Removed - no extension limit
    
    return True

def entry_signal_5min(data_5min, current_index, smma_21, smma_50, smma_200, body_sizes, daily_smma_21, daily_smma_50, daily_smma_200):
    """
    Determine if we should enter a long position based on 5-minute data.
    
    Entry criteria:
    - Previous candle closed above 21 SMMA AND is green
    - 21 SMMA > 50 SMMA > 200 SMMA (bullish alignment on 5-min)
    - Daily 21 SMMA > Daily 50 SMMA > Daily 200 SMMA (bullish alignment on daily)
    - Body of previous candle is at least 2X the average body size of last 3 candles
    - Current candle is green
    - Next candle opens within acceptable range (no more than 0.1% gap down)
    
    Args:
        data_5min (pd.DataFrame): 5-minute OHLCV data
        current_index (int): Current index in the dataframe
        smma_21 (pd.Series): Pre-calculated 21-period SMMA
        smma_50 (pd.Series): Pre-calculated 50-period SMMA
        smma_200 (pd.Series): Pre-calculated 200-period SMMA
        body_sizes (pd.Series): Pre-calculated body sizes
        daily_smma_21 (pd.Series): Pre-calculated daily 21-period SMMA
        daily_smma_50 (pd.Series): Pre-calculated daily 50-period SMMA
        daily_smma_200 (pd.Series): Pre-calculated daily 200-period SMMA
    
    Returns:
        bool: True if we should enter a position
    """
    # Need at least 200 days worth of candles for daily 200 SMMA calculation
    # 200 days * ~78 candles per day = ~15,600 candles minimum
    min_candles_needed = 200 * 78  # 15,600 candles
    if current_index < min_candles_needed:
        return False
    
    # Get previous candle index
    prev_index = current_index - 1
    
    # Criterion 1: Previous candle closed above 21 SMMA AND is green
    if data_5min.iloc[prev_index]['close'] <= smma_21.iloc[prev_index]:
        return False
    
    # Previous candle must also be green (close > open)
    if data_5min.iloc[prev_index]['close'] <= data_5min.iloc[prev_index]['open']:
        return False
    
    # Criterion 1c: SMA alignment - 21 SMMA > 50 SMMA > 200 SMMA (5-minute)
    if (smma_21.iloc[prev_index] <= smma_50.iloc[prev_index] or 
        smma_50.iloc[prev_index] <= smma_200.iloc[prev_index]):
        return False
    
    # Criterion 1d: Daily SMA alignment - Daily 21 SMMA > Daily 50 SMMA > Daily 200 SMMA
    if (daily_smma_21.iloc[prev_index] <= daily_smma_50.iloc[prev_index] or 
        daily_smma_50.iloc[prev_index] <= daily_smma_200.iloc[prev_index]):
        return False
    
    # Criterion 2: Body of previous candle is at least 2X average body size of last 3 candles
    avg_body_size_3 = body_sizes.iloc[prev_index-3:prev_index].mean()
    prev_body_size = body_sizes.iloc[prev_index]
    
    if prev_body_size < (2 * avg_body_size_3):
        return False
    
    # Criterion 3: Current candle is green
    if data_5min.iloc[current_index]['close'] <= data_5min.iloc[current_index]['open']:
        return False
    
    # Criterion 4: Next candle opens within acceptable range (no more than 0.1% gap down)
    if current_index + 1 >= len(data_5min):
        return False
    
    next_open = data_5min.iloc[current_index + 1]['open']
    current_close = data_5min.iloc[current_index]['close']
    
    # Calculate gap percentage
    gap_pct = (next_open - current_close) / current_close
    
    # Allow small gaps up or down, but reject if gap down is more than 0.1%
    if gap_pct < -0.001:  # -0.1% gap down
        return False
    
    # All criteria met
    return True

def calculate_position_size(price):
    """
    Calculate the number of shares to buy, never exceeding position size.
    
    Args:
        price (float): Current stock price
    
    Returns:
        int: Number of shares to buy
    """
    shares = int(POSITION_SIZE / price)
    actual_value = shares * price
    
    logging.info(f"Price: ${price:.2f}, Shares: {shares}, Value: ${actual_value:,.2f}")
    return shares

def monitor_exit_1min(data_1min_all, entry_price, entry_time):
    """
    Monitor 1-minute data for exit signals using trailing stop loss.
    
    Exit conditions:
    - Trailing stop loss (0.25% below highest price reached)
    - Market close
    - End of data
    
    Trailing stop logic:
    - Start with initial stop loss (-0.25% from entry)
    - When price reaches +0.25% profit, activate trailing stop
    - Trailing stop = highest_price * 0.9975 (0.25% below peak)
    - Check LOW first (conservative), then check HIGH for new peaks
    
    Args:
        data_1min_all (pd.DataFrame): All 1-minute OHLCV data
        entry_price (float): Price at which we entered the position
        entry_time (datetime): Time we entered the position
    
    Returns:
        tuple: (exit_price, exit_time, exit_reason)
    """
    if data_1min_all.empty:
        return entry_price, entry_time, "No 1-minute data"
    
    # Filter 1-minute data to start AFTER entry time (next candle)
    data_1min = data_1min_all[data_1min_all['timestamp'] > entry_time].copy()
    
    if data_1min.empty:
        return entry_price, entry_time, "No 1-minute data after entry"
    
    # Initialize trailing stop variables
    initial_stop_loss = entry_price * (1 + STOP_LOSS)  # -0.25% from entry
    trailing_stop = initial_stop_loss
    current_peak = entry_price
    trailing_activated = False
    activation_threshold = entry_price * 1.0025  # +0.25% profit to activate trailing
    
    logging.info(f"    Monitoring exit from {entry_time} at ${entry_price:.2f}")
    logging.info(f"    Initial stop loss: {STOP_LOSS:.1%} (${initial_stop_loss:.2f})")
    logging.info(f"    Trailing activation: +0.25% (${activation_threshold:.2f})")
    
    for index, row in data_1min.iterrows():
        current_time = row['timestamp']
        high = row['high']
        low = row['low']
        close = row['close']
        
        logging.info(f"    {current_time} - H:${high:.2f} L:${low:.2f} C:${close:.2f}")
        
        # Check trailing stop first (conservative approach)
        if low <= trailing_stop:
            logging.info(f"    TRAILING STOP HIT! Low ${low:.2f} <= Stop ${trailing_stop:.2f}")
            return trailing_stop, current_time, "Trailing Stop"
        
        # Check for new peak and update trailing stop
        if high > current_peak:
            current_peak = high
            
            # Activate trailing stop when we reach +0.25% profit
            if not trailing_activated and high >= activation_threshold:
                trailing_activated = True
                logging.info(f"    TRAILING STOP ACTIVATED at ${high:.2f} (+{((high/entry_price-1)*100):.2f}%)")
            
            # Update trailing stop if activated
            if trailing_activated:
                new_trailing_stop = high * 0.9975  # 0.25% below peak
                if new_trailing_stop > trailing_stop:  # Only move up, never down
                    trailing_stop = new_trailing_stop
                    logging.info(f"    Trailing stop updated to ${trailing_stop:.2f} (peak: ${high:.2f})")
        
        # Check if it's market close using exchange calendar
        if not is_regular_trading_hours(current_time):
            logging.info(f"    MARKET CLOSE - Exiting at ${close:.2f}")
            return close, current_time, "Market Close"
    
    # If we get here, exit at the last available price
    last_row = data_1min.iloc[-1]
    logging.info(f"    END OF DATA - Exiting at ${last_row['close']:.2f}")
    return last_row['close'], last_row['timestamp'], "End of Data"

# =============================================================================
# MAIN BACKTESTING FUNCTION
# =============================================================================

def run_backtest():
    """
    Main backtesting function.
    """
    logging.info(f"Starting backtest for {SYMBOL} using {STRATEGY} strategy")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=BACKTEST_DAYS)
    
    # For daily SMAs, we need extra historical data before the backtest period
    # Add 250 days before start_date to ensure we have enough data for daily 200 SMMA
    data_start_date = start_date - timedelta(days=250)
    
    logging.info(f"Date range: {start_date.date()} to {end_date.date()}")
    
    # Fetch 5-minute data (extended range for daily SMA calculation)
    logging.info("Fetching 5-minute data...")
    data_5min = fetch_5min_data(SYMBOL, data_start_date, end_date)
    if data_5min.empty:
        logging.error("No 5-minute data available for backtesting")
        return
    
    # Fetch all 1-minute data upfront (only for backtest period)
    logging.info("Fetching 1-minute data...")
    data_1min = fetch_1min_data(SYMBOL, start_date, end_date)
    if data_1min.empty:
        logging.error("No 1-minute data available for backtesting")
        return
    
    # 1-minute data is already filtered for trading hours in fetch_1min_data
    data_1min['timestamp'] = pd.to_datetime(data_1min['timestamp'])
    data_1min = data_1min.sort_values('timestamp').reset_index(drop=True)
    
    # Pre-calculate indicators once (major performance optimization)
    # Calculate indicators based on strategy
    logging.info("Calculating indicators...")
    
    if STRATEGY == "sma_alignment":
        smma_21 = calculate_smma(data_5min, 21)
        smma_50 = calculate_smma(data_5min, 50)
        smma_200 = calculate_smma(data_5min, 200)
        body_sizes = calculate_body_size(data_5min)
        
        # Calculate daily SMAs for trend confirmation
        logging.info("Calculating daily SMAs...")
        daily_smma_21 = calculate_smma(data_5min, 21 * 78)  # 21 days * ~78 5-min candles per day
        daily_smma_50 = calculate_smma(data_5min, 50 * 78)  # 50 days * ~78 5-min candles per day  
        daily_smma_200 = calculate_smma(data_5min, 200 * 78)  # 200 days * ~78 5-min candles per day
        
    elif STRATEGY == "gap_breakout":
        vwap = calculate_vwap(data_5min)
        rsi = calculate_rsi(data_5min, 14)
        
    logging.info("Indicators calculated, starting backtest...")
    
    # Track trades and results
    trades = []
    daily_trades = {}  # Track trades per day to ensure only one per day
    
    # Find the actual start index for the backtest period
    backtest_start_time = pd.Timestamp(start_date, tz='UTC')
    backtest_start_index = None
    for i in range(len(data_5min)):
        if data_5min.iloc[i]['timestamp'] >= backtest_start_time:
            backtest_start_index = i
            break
    
    if backtest_start_index is None:
        logging.error("Could not find start of backtest period in data")
        return
    
    # Main backtesting loop (only process backtest period)
    if STRATEGY == "sma_alignment":
        min_candles_needed = 200 * 78  # 15,600 candles for daily 200 SMMA
    elif STRATEGY == "gap_breakout":
        min_candles_needed = 20 * 78   # 1,560 candles for RSI calculation (20 days)
    
    loop_start = max(min_candles_needed, backtest_start_index)
    
    logging.info(f"Data length: {len(data_5min)}")
    logging.info(f"Min candles needed: {min_candles_needed}")
    logging.info(f"Backtest start index: {backtest_start_index}")
    logging.info(f"Loop start: {loop_start}")
    logging.info(f"Loop end: {len(data_5min) - 1}")
    logging.info(f"Total candles to process: {len(data_5min) - 1 - loop_start}")
    
    for i in range(loop_start, len(data_5min) - 1):
        current_row = data_5min.iloc[i]
        current_date = current_row['timestamp'].date()
            
        # Skip if we already traded today
        if current_date in daily_trades:
            continue
        
        # Check for entry signal based on strategy
        entry_signal = False
        if STRATEGY == "sma_alignment":
            # Debug specific date/time if requested
            debug_specific_datetime(data_5min, i, smma_21, smma_50, smma_200, body_sizes, daily_smma_21, daily_smma_50, daily_smma_200)
            entry_signal = entry_signal_5min(data_5min, i, smma_21, smma_50, smma_200, body_sizes, daily_smma_21, daily_smma_50, daily_smma_200)
        elif STRATEGY == "gap_breakout":
            # Debug specific date/time if requested
            debug_gap_breakout(data_5min, i, vwap, rsi, data_1min)
            entry_signal = entry_signal_gap_breakout(data_5min, i, vwap, rsi, data_1min)
        
        if entry_signal:
            # Wait for next 1-minute candle and enter at its close if green
            current_5min_time = current_row['timestamp']
            
            # Find the next 1-minute candle after the current 5-minute candle
            next_1min_candles = data_1min[data_1min['timestamp'] > current_5min_time]
            
            if next_1min_candles.empty:
                continue  # No next 1-minute candle available
            
            next_1min_candle = next_1min_candles.iloc[0]
            
            # Check if the next 1-minute candle closes green
            if next_1min_candle['close'] > next_1min_candle['open']:
                # Enter at the close of the next 1-minute candle
                entry_time = next_1min_candle['timestamp']
                entry_price = next_1min_candle['close']
            else:
                # Skip this signal - next 1-minute candle was red
                continue
            
            # Calculate position size
            shares = calculate_position_size(entry_price)
            
            # Mark this day as traded
            daily_trades[current_date] = True
            
            logging.info(f"ENTRY SIGNAL at {to_et_time(entry_time)}: {shares} shares at ${entry_price:.2f}")
            
            # Monitor for exit using pre-fetched 1-minute data
            exit_price, exit_time, exit_reason = monitor_exit_1min(data_1min, entry_price, entry_time)
            
            # Calculate profit/loss
            profit_loss = (exit_price - entry_price) * shares
            profit_loss_pct = (exit_price - entry_price) / entry_price
            
            # Log the trade
            trade = {
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_time': exit_time,
                'exit_price': exit_price,
                'shares': shares,
                'profit_loss': profit_loss,
                'profit_loss_pct': profit_loss_pct,
                'exit_reason': exit_reason
            }
            
            trades.append(trade)
            log_trade(entry_time, entry_price, exit_time, exit_price, shares, profit_loss, exit_reason)
    
    # Generate summary report
    generate_summary_report(trades)

# =============================================================================
# RESULTS AND REPORTING
# =============================================================================

def log_trade(entry_time, entry_price, exit_time, exit_price, shares, profit_loss, exit_reason):
    """
    Log individual trade details.
    """
    profit_loss_pct = (exit_price - entry_price) / entry_price * 100
    
    logging.info(f"TRADE COMPLETED:")
    logging.info(f"  Entry: {to_et_time(entry_time)} at ${entry_price:.2f}")
    logging.info(f"  Exit:  {to_et_time(exit_time)} at ${exit_price:.2f} ({exit_reason})")
    logging.info(f"  Shares: {shares}")
    logging.info(f"  P&L: ${profit_loss:,.2f} ({profit_loss_pct:+.2f}%)")
    logging.info("-" * 50)

def generate_summary_report(trades):
    """
    Generate final summary report.
    """
    if not trades:
        logging.info("No trades executed during backtest period")
        return
    
    total_trades = len(trades)
    total_profit_loss = sum(trade['profit_loss'] for trade in trades)
    total_profit_loss_pct = (total_profit_loss / POSITION_SIZE) * 100
    
    winning_trades = [t for t in trades if t['profit_loss'] > 0]
    losing_trades = [t for t in trades if t['profit_loss'] < 0]
    
    win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
    
    avg_win = sum(t['profit_loss'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(t['profit_loss'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
    
    # Exit reason breakdown
    exit_reasons = {}
    for trade in trades:
        reason = trade['exit_reason']
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
    
    logging.info("=" * 60)
    logging.info("BACKTEST SUMMARY REPORT")
    logging.info("=" * 60)
    logging.info(f"Symbol: {SYMBOL}")
    logging.info(f"Total Trades: {total_trades}")
    logging.info(f"Winning Trades: {len(winning_trades)}")
    logging.info(f"Losing Trades: {len(losing_trades)}")
    logging.info(f"Win Rate: {win_rate:.1f}%")
    logging.info(f"Total P&L: ${total_profit_loss:,.2f}")
    logging.info(f"Total Return: {total_profit_loss_pct:+.2f}%")
    logging.info(f"Average Win: ${avg_win:,.2f}")
    logging.info(f"Average Loss: ${avg_loss:,.2f}")
    logging.info("")
    logging.info("Exit Reasons:")
    for reason, count in exit_reasons.items():
        logging.info(f"  {reason}: {count}")
    logging.info("=" * 60)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print(f"Starting backtest for {SYMBOL}")
    print(f"Position size: ${POSITION_SIZE:,}")
    print(f"Profit target: {PROFIT_TARGET*100}%")
    print(f"Stop loss: {STOP_LOSS*100}%")
    print(f"Backtest period: {BACKTEST_DAYS} days")
    print("-" * 50)
    
    # Run the backtest
    run_backtest()
