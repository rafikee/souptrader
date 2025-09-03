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
STOP_LOSS = -0.005                # -0.5% stop loss
BACKTEST_DAYS = 7                   # Number of days to backtest (1 week for testing)

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

def entry_signal_5min(data_5min, current_index, smma_21, body_sizes):
    """
    Determine if we should enter a long position based on 5-minute data.
    
    Entry criteria:
    - Previous candle closed above 21 SMMA
    - Body of previous candle is at least 2X the average body size of last 3 candles
    - Current candle is green
    - Next candle opens green then we enter
    
    Args:
        data_5min (pd.DataFrame): 5-minute OHLCV data
        current_index (int): Current index in the dataframe
        smma_21 (pd.Series): Pre-calculated 21-period SMMA
        body_sizes (pd.Series): Pre-calculated body sizes
    
    Returns:
        bool: True if we should enter a position
    """
    # Need at least 25 candles for SMMA calculation and lookback
    if current_index < 25:
        return False
    
    # Get previous candle index
    prev_index = current_index - 1
    
    # Criterion 1: Previous candle closed above 21 SMMA
    if data_5min.iloc[prev_index]['close'] <= smma_21.iloc[prev_index]:
        return False
    
    # Criterion 2: Body of previous candle is at least 2X average body size of last 3 candles
    avg_body_size_3 = body_sizes.iloc[prev_index-3:prev_index].mean()
    prev_body_size = body_sizes.iloc[prev_index]
    
    if prev_body_size < (2 * avg_body_size_3):
        return False
    
    # Criterion 3: Current candle is green
    if data_5min.iloc[current_index]['close'] <= data_5min.iloc[current_index]['open']:
        return False
    
    # Criterion 4: Next candle opens green
    if current_index + 1 >= len(data_5min):
        return False
    
    next_open = data_5min.iloc[current_index + 1]['open']
    current_close = data_5min.iloc[current_index]['close']
    
    if next_open <= current_close:
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
    Monitor 1-minute data for exit signals using pre-fetched data.
    
    Exit conditions:
    - 1% profit target
    - -0.5% stop loss
    - Market close
    
    Args:
        data_1min_all (pd.DataFrame): All 1-minute OHLCV data
        entry_price (float): Price at which we entered the position
        entry_time (datetime): Time we entered the position
    
    Returns:
        tuple: (exit_price, exit_time, exit_reason)
    """
    if data_1min_all.empty:
        return entry_price, entry_time, "No 1-minute data"
    
    # Filter 1-minute data to start from entry time
    data_1min = data_1min_all[data_1min_all['timestamp'] >= entry_time].copy()
    
    if data_1min.empty:
        return entry_price, entry_time, "No 1-minute data after entry"
    
    profit_target_price = entry_price * (1 + PROFIT_TARGET)
    stop_loss_price = entry_price * (1 + STOP_LOSS)
    
    for index, row in data_1min.iterrows():
        current_time = row['timestamp']
        high = row['high']
        low = row['low']
        close = row['close']
        
        # Check if we hit profit target (high touched target price)
        if high >= profit_target_price:
            return profit_target_price, current_time, "Profit Target"
        
        # Check if we hit stop loss (low touched stop price)
        if low <= stop_loss_price:
            return stop_loss_price, current_time, "Stop Loss"
        
        # Check if it's market close using exchange calendar
        if not is_regular_trading_hours(current_time):
            return close, current_time, "Market Close"
    
    # If we get here, exit at the last available price
    last_row = data_1min.iloc[-1]
    return last_row['close'], last_row['timestamp'], "End of Data"

# =============================================================================
# MAIN BACKTESTING FUNCTION
# =============================================================================

def run_backtest():
    """
    Main backtesting function.
    """
    logging.info(f"Starting backtest for {SYMBOL}")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=BACKTEST_DAYS)
    
    logging.info(f"Date range: {start_date.date()} to {end_date.date()}")
    
    # Fetch 5-minute data
    logging.info("Fetching 5-minute data...")
    data_5min = fetch_5min_data(SYMBOL, start_date, end_date)
    if data_5min.empty:
        logging.error("No 5-minute data available for backtesting")
        return
    
    # Fetch all 1-minute data upfront
    logging.info("Fetching 1-minute data...")
    data_1min = fetch_1min_data(SYMBOL, start_date, end_date)
    if data_1min.empty:
        logging.error("No 1-minute data available for backtesting")
        return
    
    # 1-minute data is already filtered for trading hours in fetch_1min_data
    data_1min['timestamp'] = pd.to_datetime(data_1min['timestamp'])
    data_1min = data_1min.sort_values('timestamp').reset_index(drop=True)
    
    # Pre-calculate indicators once (major performance optimization)
    logging.info("Calculating indicators...")
    smma_21 = calculate_smma(data_5min, 21)
    body_sizes = calculate_body_size(data_5min)
    logging.info("Indicators calculated, starting backtest...")
    
    # Track trades and results
    trades = []
    daily_trades = {}  # Track trades per day to ensure only one per day
    
    # Main backtesting loop
    for i in range(25, len(data_5min) - 1):  # Start after SMMA calculation, leave room for next candle
        current_row = data_5min.iloc[i]
        current_date = current_row['timestamp'].date()
        
        # Skip if we already traded today
        if current_date in daily_trades:
            continue
        
        # Check for entry signal
        if entry_signal_5min(data_5min, i, smma_21, body_sizes):
            entry_time = current_row['timestamp']
            entry_price = current_row['close']  # Enter at current candle's close
            
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
