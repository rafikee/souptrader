"""
Backtest Runner - Bridge between Dash UI and backtesting engine

This module coordinates backtest execution with configurable parameters.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import pytz
import exchange_calendars as xcals

# Timezone configuration
UTC = pytz.UTC
ET = pytz.timezone('US/Eastern')
NYSE = xcals.get_calendar('XNYS')


def format_et_time(timestamp):
    """Convert timestamp to Eastern Time string for logging"""
    if timestamp.tzinfo is None:
        timestamp = UTC.localize(timestamp)
    elif timestamp.tzinfo != UTC:
        timestamp = timestamp.astimezone(UTC)
    
    et_time = timestamp.astimezone(ET)
    return et_time.strftime('%Y-%m-%d %H:%M:%S %Z')

# Constants
POSITION_SIZE = 100000  # $100k position size
PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKTEST_DATA_PATH = PROJECT_ROOT / 'data' / 'backtest_data'


class BacktestConfig:
    """Configuration for a backtest run"""
    def __init__(self, ticker, strategy, filters, stop_loss_pct, trailing_stop_pct, 
                 take_profit_pct=None):
        self.ticker = ticker
        self.strategy = strategy  # 'sma_alignment' or 'gap_breakout'
        self.filters = filters  # Dict of filter name -> enabled (bool)
        self.stop_loss_pct = stop_loss_pct  # e.g., -0.005 for -0.5%
        self.trailing_stop_pct = trailing_stop_pct  # e.g., 0.0025 for 0.25%
        self.take_profit_pct = take_profit_pct  # e.g., 0.01 for 1% or None


class BacktestResults:
    """Results from a backtest run"""
    def __init__(self):
        self.trades = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.win_rate = 0.0
        self.total_pnl = 0.0
        self.total_return_pct = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.exit_reasons = {}
        self.log_messages = []
        self.error = None


def is_regular_trading_hours(timestamp):
    """Check if timestamp is during regular trading hours"""
    if timestamp.tzinfo is None:
        timestamp = UTC.localize(timestamp)
    
    et_time = timestamp.astimezone(ET)
    
    if not NYSE.is_session(et_time.date()):
        return False
    
    session_start = NYSE.session_open(et_time.date())
    session_end = NYSE.session_close(et_time.date())
    
    return session_start <= timestamp <= session_end


def load_bar_data(ticker, timeframe='5Min'):
    """
    Load bar data from local parquet files and filter to regular trading hours.
    
    Args:
        ticker (str): Stock symbol
        timeframe (str): '5Min' or 'Daily'
    
    Returns:
        pd.DataFrame: Concatenated bar data sorted by timestamp, filtered to RTH
    """
    ticker_path = BACKTEST_DATA_PATH / ticker / timeframe
    
    if not ticker_path.exists():
        raise FileNotFoundError(f"No {timeframe} data found for {ticker} at {ticker_path}")
    
    parquet_files = sorted(ticker_path.glob('*.parquet'))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {ticker_path}")
    
    dfs = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        dfs.append(df)
    
    # Concatenate all data
    data = pd.concat(dfs, ignore_index=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values('timestamp').reset_index(drop=True)
    
    # Filter to regular trading hours only for intraday data (5Min)
    # Daily data doesn't need RTH filtering since it represents the full trading day
    if timeframe == '5Min':
        data = data[data['timestamp'].apply(is_regular_trading_hours)].reset_index(drop=True)
    
    return data


def load_nbbo_data_for_day(ticker, trade_date):
    """
    Load NBBO 1-second data for a specific trading day.
    
    Args:
        ticker (str): Stock symbol
        trade_date (datetime.date): Trading date
    
    Returns:
        pd.DataFrame: NBBO data for that day
    """
    date_str = trade_date.strftime('%Y-%m-%d')
    nbbo_file = BACKTEST_DATA_PATH / ticker / 'nbbo1s' / f'{date_str}.parquet'
    
    if not nbbo_file.exists():
        raise FileNotFoundError(f"NBBO data not found for {ticker} on {date_str}")
    
    df = pd.read_parquet(nbbo_file)
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    df = df.sort_values('ts_event').reset_index(drop=True)
    
    return df


def calculate_smma(data, period, column='close'):
    """Calculate Smoothed Moving Average (SMMA)"""
    if len(data) == 0:
        raise ValueError(f"Cannot calculate SMMA: data is empty")
    
    if len(data) < period:
        raise ValueError(f"Cannot calculate SMMA: need at least {period} bars, but only have {len(data)}")
    
    smma = pd.Series(index=data.index, dtype=float)
    smma.iloc[0] = data[column].iloc[0]
    
    for i in range(1, len(data)):
        if i < period:
            smma.iloc[i] = (smma.iloc[i-1] * i + data[column].iloc[i]) / (i + 1)
        else:
            smma.iloc[i] = (smma.iloc[i-1] * (period - 1) + data[column].iloc[i]) / period
    
    return smma


def calculate_rsi(data, period=14):
    """Calculate RSI"""
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_body_size(data):
    """Calculate candle body size"""
    return abs(data['close'] - data['open'])


def check_sma_alignment(data_5min, index, smma_21, smma_50, smma_200):
    """Check if SMAs are aligned (21 > 50 > 200)"""
    return (smma_21.iloc[index] > smma_50.iloc[index] > smma_200.iloc[index])


def check_vwap_filter(data_5min, index):
    """Check if price is above VWAP"""
    if 'vwap' not in data_5min.columns:
        return True  # Skip filter if VWAP not available
    return data_5min.iloc[index]['close'] > data_5min.iloc[index]['vwap']


def check_rsi_filter(rsi, index, threshold=55):
    """Check if RSI > threshold"""
    return rsi.iloc[index] > threshold


def check_volume_filter(data_5min, index):
    """Check if volume > 1.5x average (simplified for now)"""
    # TODO: Implement proper 20-day average volume calculation
    return True


def check_body_size_filter(data_5min, index, body_sizes):
    """Check if body size >= 2x average of last 3 candles"""
    if index < 3:
        return False
    avg_body = body_sizes.iloc[index-3:index].mean()
    return body_sizes.iloc[index] >= (2 * avg_body)


def entry_signal_sma_alignment(data_5min, index, smma_21, smma_50, smma_200, 
                                body_sizes, daily_smma_21, daily_smma_50, 
                                daily_smma_200, filters):
    """
    Check entry signal for SMA alignment strategy with configurable filters.
    """
    min_candles = 200 * 78  # Need 200 days of data
    if index < min_candles:
        return False
    
    prev_index = index - 1
    
    # Core criteria (always required)
    # Previous candle closed above 21 SMMA and is green
    if data_5min.iloc[prev_index]['close'] <= smma_21.iloc[prev_index]:
        return False
    if data_5min.iloc[prev_index]['close'] <= data_5min.iloc[prev_index]['open']:
        return False
    
    # Current candle is green
    if data_5min.iloc[index]['close'] <= data_5min.iloc[index]['open']:
        return False
    
    # Next candle check (no more than 0.1% gap down)
    if index + 1 >= len(data_5min):
        return False
    next_open = data_5min.iloc[index + 1]['open']
    current_close = data_5min.iloc[index]['close']
    gap_pct = (next_open - current_close) / current_close
    if gap_pct < -0.001:
        return False
    
    # Configurable filters
    if filters.get('sma_alignment', False):
        if not check_sma_alignment(data_5min, prev_index, smma_21, smma_50, smma_200):
            return False
        # Daily SMA alignment (only if daily data is available)
        if (daily_smma_21 is not None and daily_smma_50 is not None and daily_smma_200 is not None):
            if (daily_smma_21.iloc[prev_index] <= daily_smma_50.iloc[prev_index] or 
                daily_smma_50.iloc[prev_index] <= daily_smma_200.iloc[prev_index]):
                return False
    
    if filters.get('vwap_filter', False):
        if not check_vwap_filter(data_5min, index):
            return False
    
    if filters.get('body_size', False):
        if not check_body_size_filter(data_5min, prev_index, body_sizes):
            return False
    
    return True


def entry_signal_gap_breakout(data_5min, index, vwap, rsi, filters):
    """
    Check entry signal for gap breakout strategy with configurable filters.
    """
    if index < 2:
        return False
    
    current_row = data_5min.iloc[index]
    current_time = current_row['timestamp']
    et_time = current_time.tz_convert('US/Eastern')
    
    # Only first hour (9:30-10:30 AM ET)
    if et_time.hour != 9:
        return False
    
    # Get previous day's close
    prev_day_close = None
    for i in range(index - 1, -1, -1):
        prev_candle = data_5min.iloc[i]
        prev_et_time = prev_candle['timestamp'].tz_convert('US/Eastern')
        if prev_et_time.date() < et_time.date():
            prev_day_close = prev_candle['close']
            break
    
    if prev_day_close is None:
        return False
    
    # Core criteria: Gap up at least +1%
    gap_pct = (current_row['open'] - prev_day_close) / prev_day_close
    if gap_pct < 0.01:
        return False
    
    # Opening range breakout
    day_start_index = None
    for j in range(index, -1, -1):
        candle_time = data_5min.iloc[j]['timestamp'].tz_convert('US/Eastern')
        if candle_time.date() < et_time.date():
            day_start_index = j + 1
            break
    
    if day_start_index is None or day_start_index + 1 >= index:
        return False
    
    first_candle = data_5min.iloc[day_start_index]
    second_candle = data_5min.iloc[day_start_index + 1]
    
    # Both opening candles must be green
    if (first_candle['close'] <= first_candle['open'] or 
        second_candle['close'] <= second_candle['open']):
        return False
    
    opening_range_high = max(first_candle['high'], second_candle['high'])
    
    # Break above opening range
    if current_row['close'] <= opening_range_high:
        return False
    
    # Configurable filters
    if filters.get('vwap_filter', False):
        if not check_vwap_filter(data_5min, index):
            return False
    
    if filters.get('rsi_filter', False):
        if not check_rsi_filter(rsi, index):
            return False
    
    # Volume filter (simplified for now)
    if filters.get('volume_confirmation', False):
        if not check_volume_filter(data_5min, index):
            return False
    
    return True


def execute_trade_with_nbbo(ticker, entry_signal_time, entry_signal_price, 
                             stop_loss_pct, trailing_stop_pct, take_profit_pct, 
                             log_func):
    """
    Execute a trade using NBBO data for realistic entry/exit.
    
    Args:
        log_func: Function to call for logging (takes a string message)
    
    Returns:
        dict: Trade results or None if trade not executed
    """
    trade_date = entry_signal_time.date()
    
    try:
        nbbo_data = load_nbbo_data_for_day(ticker, trade_date)
    except FileNotFoundError as e:
        log_func(f"ERROR: {str(e)}")
        return None
    
    # Calculate buy stop price (0.1% above signal price)
    buy_stop_price = entry_signal_price * 1.001
    
    # Find entry point: look for NBBO tick where ask >= buy stop price
    # Start from the signal time onwards
    nbbo_after_signal = nbbo_data[nbbo_data['ts_event'] > entry_signal_time]
    
    entry_tick = None
    for idx, row in nbbo_after_signal.iterrows():
        if row['ask_px_00'] >= buy_stop_price:
            entry_tick = row
            break
    
    if entry_tick is None:
        log_func(f"  Buy stop not filled (price never reached ${buy_stop_price:.2f})")
        return None
    
    # Entry executed at ask price
    entry_price = entry_tick['ask_px_00']
    entry_time = entry_tick['ts_event']
    shares = int(POSITION_SIZE / entry_price)
    
    log_func(f"ENTRY: {format_et_time(entry_time)} at ${entry_price:.2f} ({shares} shares)")
    log_func(f"  Buy stop triggered at ${buy_stop_price:.2f}")
    
    # Initialize stop loss tracking
    initial_stop_price = entry_price * (1 + stop_loss_pct)
    trailing_stop_price = initial_stop_price
    current_peak = entry_price
    trailing_activated = False
    activation_threshold = entry_price * (1 + trailing_stop_pct)
    
    log_func(f"  Initial stop: ${initial_stop_price:.2f} ({stop_loss_pct*100:.2f}%)")
    log_func(f"  Trailing activates at: ${activation_threshold:.2f} (+{trailing_stop_pct*100:.2f}%)")
    if take_profit_pct:
        take_profit_price = entry_price * (1 + take_profit_pct)
        log_func(f"  Take profit: ${take_profit_price:.2f} (+{take_profit_pct*100:.2f}%)")
    
    # Monitor exit using every NBBO tick after entry
    nbbo_after_entry = nbbo_data[nbbo_data['ts_event'] > entry_time]
    
    exit_price = None
    exit_time = None
    exit_reason = None
    
    for idx, row in nbbo_after_entry.iterrows():
        bid_price = row['bid_px_00']
        ask_price = row['ask_px_00']
        tick_time = row['ts_event']
        
        # Check if we've exited regular trading hours
        if not is_regular_trading_hours(tick_time):
            # Exit at market (bid price)
            exit_price = bid_price
            exit_time = tick_time
            exit_reason = "EOD"
            break
        
        # Check trailing stop hit (sell at bid)
        if bid_price <= trailing_stop_price:
            exit_price = bid_price
            exit_time = tick_time
            exit_reason = "Trailing Stop"
            break
        
        # Check take profit hit (if enabled)
        if take_profit_pct and bid_price >= take_profit_price:
            exit_price = bid_price
            exit_time = tick_time
            exit_reason = "Take Profit"
            break
        
        # Update peak and trailing stop
        if ask_price > current_peak:
            current_peak = ask_price
            
            # Activate trailing stop
            if not trailing_activated and ask_price >= activation_threshold:
                trailing_activated = True
                log_func(f"  TRAILING ACTIVATED at ${ask_price:.2f}")
            
            # Update trailing stop
            if trailing_activated:
                new_trailing = current_peak * (1 - trailing_stop_pct)
                if new_trailing > trailing_stop_price:
                    trailing_stop_price = new_trailing
    
    # If we didn't exit, use last NBBO tick
    if exit_price is None:
        last_tick = nbbo_after_entry.iloc[-1]
        exit_price = last_tick['bid_px_00']
        exit_time = last_tick['ts_event']
        exit_reason = "End of Data"
    
    # Calculate P&L
    pnl = (exit_price - entry_price) * shares
    pnl_pct = (exit_price - entry_price) / entry_price
    
    log_func(f"EXIT: {format_et_time(exit_time)} at ${exit_price:.2f} ({exit_reason})")
    log_func(f"  P&L: ${pnl:,.2f} ({pnl_pct*100:+.2f}%)")
    log_func("-" * 60)
    
    return {
        'entry_time': entry_time,
        'entry_price': entry_price,
        'exit_time': exit_time,
        'exit_price': exit_price,
        'shares': shares,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'exit_reason': exit_reason
    }


def run_backtest(config: BacktestConfig) -> BacktestResults:
    """
    Main backtest execution function.
    
    Args:
        config: BacktestConfig object with all parameters
    
    Returns:
        BacktestResults object with trades and summary
    """
    results = BacktestResults()
    logger = results.log_messages
    
    # Setup log file
    log_file_path = PROJECT_ROOT / 'logs' / 'backtest_results.log'
    log_file_path.parent.mkdir(exist_ok=True)
    
    # Clear log file
    with open(log_file_path, 'w') as f:
        f.write('')
    
    def log(message):
        """Log to both memory and file"""
        logger.append(message)
        with open(log_file_path, 'a') as f:
            f.write(message + '\n')
    
    try:
        log("=" * 60)
        log(f"BACKTEST: {config.ticker}")
        log(f"Strategy: {config.strategy}")
        log(f"Filters: {config.filters}")
        log(f"Stop Loss: {config.stop_loss_pct*100:.2f}%")
        log(f"Trailing Stop: {config.trailing_stop_pct*100:.2f}%")
        if config.take_profit_pct:
            log(f"Take Profit: {config.take_profit_pct*100:.2f}%")
        log("=" * 60)
        
        # Load 5-minute bar data
        log("Loading 5-minute bar data...")
        data_5min = load_bar_data(config.ticker, '5Min')
        log(f"Loaded {len(data_5min)} 5-minute candles")
        
        # Calculate indicators based on strategy
        log("Calculating indicators...")
        
        if config.strategy == 'sma_alignment':
            smma_21 = calculate_smma(data_5min, 21)
            smma_50 = calculate_smma(data_5min, 50)
            smma_200 = calculate_smma(data_5min, 200)
            body_sizes = calculate_body_size(data_5min)
            
            # Daily SMAs - load actual daily data
            try:
                daily_data = load_bar_data(config.ticker, 'Daily')
                log(f"Loaded daily data: {len(daily_data)} bars")
                
                if len(daily_data) < 200:
                    log(f"WARNING: Only {len(daily_data)} daily bars available, need 200 for daily SMA alignment. Skipping daily SMA filter.")
                    daily_smma_21 = daily_smma_50 = daily_smma_200 = None
                else:
                    daily_smma_21 = calculate_smma(daily_data, 21)
                    daily_smma_50 = calculate_smma(daily_data, 50)
                    daily_smma_200 = calculate_smma(daily_data, 200)
            except FileNotFoundError:
                log("WARNING: No daily data found, skipping daily SMA alignment filter")
                daily_smma_21 = daily_smma_50 = daily_smma_200 = None
            except ValueError as e:
                log(f"WARNING: Cannot calculate daily SMAs: {e}")
                daily_smma_21 = daily_smma_50 = daily_smma_200 = None
            
            vwap = None
            rsi = None
            
        elif config.strategy == 'gap_breakout':
            # Use VWAP from data if available
            vwap = data_5min['vwap'] if 'vwap' in data_5min.columns else None
            rsi = calculate_rsi(data_5min, 14)
            
            smma_21 = smma_50 = smma_200 = None
            body_sizes = None
            daily_smma_21 = daily_smma_50 = daily_smma_200 = None
        
        log("Indicators calculated, starting backtest...")
        
        # Track trades
        trades = []
        daily_trades = {}
        
        # Determine loop start
        if config.strategy == 'sma_alignment':
            loop_start = 200 * 78  # Need 200 days
        else:
            loop_start = 20 * 78  # Need 20 days for RSI
        
        log(f"Processing {len(data_5min) - loop_start} candles...")
        
        # Main backtest loop
        for i in range(loop_start, len(data_5min) - 1):
            current_row = data_5min.iloc[i]
            current_date = current_row['timestamp'].date()
            
            # Skip if already traded today
            if current_date in daily_trades:
                continue
            
            # Check for entry signal
            entry_signal = False
            
            if config.strategy == 'sma_alignment':
                entry_signal = entry_signal_sma_alignment(
                    data_5min, i, smma_21, smma_50, smma_200, body_sizes,
                    daily_smma_21, daily_smma_50, daily_smma_200, config.filters
                )
            elif config.strategy == 'gap_breakout':
                entry_signal = entry_signal_gap_breakout(
                    data_5min, i, vwap, rsi, config.filters
                )
            
            if entry_signal:
                log(f"\nENTRY SIGNAL: {format_et_time(current_row['timestamp'])} at ${current_row['close']:.2f}")
                
                # Execute trade with NBBO data
                trade = execute_trade_with_nbbo(
                    config.ticker,
                    current_row['timestamp'],
                    current_row['close'],
                    config.stop_loss_pct,
                    config.trailing_stop_pct,
                    config.take_profit_pct,
                    log
                )
                
                if trade:
                    trades.append(trade)
                    daily_trades[current_date] = True
        
        # Calculate summary statistics
        results.trades = trades
        results.total_trades = len(trades)
        
        if trades:
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]
            
            results.winning_trades = len(winning_trades)
            results.losing_trades = len(losing_trades)
            results.win_rate = (results.winning_trades / results.total_trades * 100) if results.total_trades > 0 else 0
            results.total_pnl = sum(t['pnl'] for t in trades)
            results.total_return_pct = (results.total_pnl / POSITION_SIZE) * 100
            results.avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
            results.avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
            
            # Exit reasons
            for trade in trades:
                reason = trade['exit_reason']
                results.exit_reasons[reason] = results.exit_reasons.get(reason, 0) + 1
        
        # Generate summary
        log("\n" + "=" * 60)
        log("BACKTEST SUMMARY")
        log("=" * 60)
        log(f"Total Trades: {results.total_trades}")
        log(f"Winning Trades: {results.winning_trades}")
        log(f"Losing Trades: {results.losing_trades}")
        log(f"Win Rate: {results.win_rate:.1f}%")
        log(f"Total P&L: ${results.total_pnl:,.2f}")
        log(f"Total Return: {results.total_return_pct:+.2f}%")
        if results.winning_trades > 0:
            log(f"Average Win: ${results.avg_win:,.2f}")
        if results.losing_trades > 0:
            log(f"Average Loss: ${results.avg_loss:,.2f}")
        
        if results.exit_reasons:
            log("\nExit Reasons:")
            for reason, count in results.exit_reasons.items():
                log(f"  {reason}: {count}")
        
        log("=" * 60)
        
    except Exception as e:
        results.error = str(e)
        log(f"\nERROR: {str(e)}")
        import traceback
        log(traceback.format_exc())
    
    return results

