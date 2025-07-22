import sqlite3
import pandas as pd
from collections import defaultdict
from datetime import datetime
import numpy as np
import logging
import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    filename=os.path.join(PROJECT_ROOT, 'logs', 'summary_update.log'),
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def calculate_yearly_profits(df):
    """
    Calculate realized profits and returns per year from trading data.
    
    Returns:
        Tuple of (profit_df, yearly_metrics, quarterly_metrics, all_trades)
    """
    
    # Strip time down to just the date
    df['filled_time'] = df['filled_time'].str[:10]
    df['filled_time'] = pd.to_datetime(df['filled_time'])

    # Add year column
    df['year'] = df['filled_time'].dt.year

    # filtering out long_term and rsu for now
    df = df[~df['category'].str.contains('long_term|rsu', na=False)]
    
    # Create dictionaries to track positions and profits
    positions = defaultdict(list)
    monthly_profits = defaultdict(float)
    quarterly_profits = defaultdict(float)
    yearly_profits = defaultdict(float)
    yearly_metrics = defaultdict(dict)
    quarterly_metrics = defaultdict(dict)
    yearly_fees = defaultdict(float)
    yearly_divs = defaultdict(float)
    yearly_options = defaultdict(float)
    yearly_stock_pnl = defaultdict(lambda: defaultdict(float))
    
    # Track open positions and their P&L
    open_positions = defaultdict(lambda: {'shares': 0, 'cost_basis': 0.0, 'realized_pnl': 0.0})
    
    # Enhanced metrics tracking
    all_trades = []
    all_dividends = []
    all_options = []
    yearly_trades = defaultdict(list)
    yearly_position_value = defaultdict(float)
    yearly_trade_counts = defaultdict(int)
    yearly_winning_trades = defaultdict(int)
    
    # Process equity trades day by day
    for date in sorted(df['filled_time'].unique()):
        day_trades = df[df['filled_time'] == date]
        month_key = date.strftime('%Y-%m')
        year = date.year
        quarter = f"{year}-Q{(date.month-1)//3 + 1}"
        
        # Process all buys first
        df_buys = day_trades[(day_trades['side'] == 'buy') & (day_trades['asset_class'] == 'us_equity')]
        for _, row in df_buys.iterrows():
            symbol = row['symbol']
            amount = float(row['amount'])
            qty = int(row['qty'])
            
            # Update open position tracking
            open_positions[symbol]['shares'] += qty
            open_positions[symbol]['cost_basis'] += amount
            
            positions[symbol].append({
                'qty': qty,
                'amount': amount,
                'filled_time': date
            })
        
        # Then process all sells
        df_sells = day_trades[(day_trades['side'] == 'sell') & (day_trades['asset_class'] == 'us_equity')]
        for _, row in df_sells.iterrows():
            symbol = row['symbol']
            category = row['category']
            amount = float(row['amount'])
            qty = int(row['qty'])
            
            remaining_qty = qty
            sell_amount = amount
            total_position_profit = 0
            trade_dates = []
            
            # Verify we have enough shares to sell (long-only check)
            total_owned = sum(pos['qty'] for pos in positions[symbol])
            if total_owned < qty:
                print(f"Warning: Attempted to sell {qty} shares of {symbol} but only owned {total_owned}. Skipping trade.")
                continue
        
            while remaining_qty > 0 and positions[symbol]:
                buy_position = positions[symbol][0]
                buy_qty = buy_position['qty']
                
                shares_to_sell = min(remaining_qty, buy_qty)
                portion_ratio = shares_to_sell / qty
                sell_portion = sell_amount * portion_ratio
                buy_portion = buy_position['amount'] * (shares_to_sell / buy_position['qty'])
                
                position_profit = sell_portion - buy_portion
                position_return_pct = (position_profit / buy_portion) * 100
                
                # Update open position tracking
                open_positions[symbol]['shares'] -= shares_to_sell
                open_positions[symbol]['realized_pnl'] += position_profit
                
                # If position is fully closed, record the P&L
                if open_positions[symbol]['shares'] == 0:
                    yearly_stock_pnl[symbol][year] += open_positions[symbol]['realized_pnl']
                    open_positions[symbol]['realized_pnl'] = 0.0
                
                monthly_profits[month_key] += position_profit
                quarterly_profits[quarter] += position_profit
                yearly_profits[year] += position_profit
                total_position_profit += position_profit
                
                trade_dates.append({
                    'buy_date': buy_position['filled_time'],
                    'sell_date': date,
                    'shares': shares_to_sell,
                    'profit': position_profit,
                    'position_size': buy_portion,
                    'return_pct': position_return_pct
                })
                
                yearly_position_value[year] += buy_portion
                
                remaining_qty -= shares_to_sell
                buy_position['qty'] -= shares_to_sell
                buy_position['amount'] -= buy_portion
                
                if buy_position['qty'] == 0:
                    positions[symbol].pop(0)
            
            yearly_trade_counts[year] += 1
            if total_position_profit > 0:
                yearly_winning_trades[year] += 1
            
            for trade in trade_dates:
                duration = (trade['sell_date'] - trade['buy_date']).days
                trade['duration'] = duration
                trade['year'] = year
                trade['ticker'] = symbol
                trade['category'] = category
                all_trades.append(trade)
                yearly_trades[year].append(trade)


    # Process all non-equity transactions after the main loop
    for date in sorted(df['filled_time'].unique()):
        day_trades = df[df['filled_time'] == date]
        month_key = date.strftime('%Y-%m')
        year = date.year
        quarter = f"{year}-Q{(date.month-1)//3 + 1}"
        
        # Process options
        df_options = day_trades[day_trades['asset_class'] == 'us_option'].copy()  # Create explicit copy
        df_options.loc[:, 'amount'] = pd.to_numeric(df_options['amount']) * np.where(df_options['side'] == 'buy', -1, 1)
        
        for _, row in df_options.iterrows():
            all_options.append({
                'date': row['filled_time'],
                'category': row['category'],
                'ticker': row['symbol'],
                'profit': row['amount'],
            })
        options = df_options['amount'].astype(float).sum()
        
        # Process fees
        df_fees = day_trades[day_trades['asset_class'] == 'fee']
        fees = df_fees['amount'].astype(float).sum()
        
        # Process interest
        df_int = day_trades[day_trades['asset_class'] == 'interest']
        interest = df_int['amount'].astype(float).sum()
        
        # Process dividends
        df_div = day_trades[day_trades['asset_class'] == 'dividend']
        for _, row in df_div.iterrows():
            all_dividends.append({
                'date': row['filled_time'],
                'category': row['category'],
                'ticker': row['symbol'],
                'profit': row['amount'],
            })
        dividends = df_div['amount'].astype(float).sum()
        
        # Update all profit metrics
        if options != 0:
            monthly_profits[month_key] += options
            quarterly_profits[quarter] += options
            yearly_profits[year] += options
            yearly_options[year] += options
            
        if fees != 0:
            monthly_profits[month_key] -= fees
            quarterly_profits[quarter] -= fees
            yearly_profits[year] -= fees
            yearly_fees[year] -= fees
            
        if interest != 0:
            monthly_profits[month_key] += interest
            quarterly_profits[quarter] += interest
            yearly_profits[year] += interest
            
        if dividends != 0:
            monthly_profits[month_key] += dividends
            quarterly_profits[quarter] += dividends
            yearly_profits[year] += dividends
            yearly_divs[year] += dividends
        
    
    # Remove QUARTERLY_DEPOSITS and get_cumulative_deposits
    # Remove all code that uses total_deposits or annualized_return
    # Only calculate and update realized profit and trading metrics

    # Calculate quarterly metrics (profit only)
    for quarter in sorted(quarterly_profits.keys()):
        quarterly_metrics[quarter] = {
            'total_profit': quarterly_profits[quarter],
        }

    # Calculate yearly metrics (no annualized return or deposits)
    for year in sorted(yearly_profits.keys()):
        year_trades = yearly_trades[year]
        year_df = df[df['year'] == year]
        if not year_df.empty:
            total_trades = yearly_trade_counts[year]
            win_rate = round((yearly_winning_trades[year] / total_trades * 100), 2) if total_trades > 0 else 0
            avg_trade_duration = round(sum(t['duration'] for t in year_trades) / len(year_trades), 1) if year_trades else 0
            avg_profit_per_trade = round(yearly_profits[year] / total_trades, 2) if total_trades > 0 else 0
            avg_position_size = round(yearly_position_value[year] / total_trades, 2) if total_trades > 0 else 0
            avg_return_pct = round(sum(t['return_pct'] for t in year_trades) / len(year_trades), 2) if year_trades else 0
            position_weighted_return = 0
            if year_trades and yearly_position_value[year] > 0:
                position_weighted_return = round(
                    sum(t['return_pct'] * (t['position_size'] / yearly_position_value[year]) for t in year_trades), 
                    2
                )
            yearly_metrics[year] = {
                'total_profit': yearly_profits[year],
                'total_fees': yearly_fees[year],
                'total_divs': yearly_divs[year],
                'total_options': yearly_options[year],
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_trade_duration': avg_trade_duration,
                'avg_profit_per_trade': avg_profit_per_trade,
                'avg_position_size': avg_position_size,
                'avg_return_pct': avg_return_pct,
                'position_weighted_return': position_weighted_return
            }
            winning_trades = [t for t in year_trades if t['profit'] > 0]
            losing_trades = [t for t in year_trades if t['profit'] <= 0]
            if winning_trades:
                yearly_metrics[year].update({
                    'avg_winning_position_size': round(sum(t['position_size'] for t in winning_trades) / len(winning_trades), 2),
                    'avg_winning_return_pct': round(sum(t['return_pct'] for t in winning_trades) / len(winning_trades), 2)
                })
            if losing_trades:
                yearly_metrics[year].update({
                    'avg_losing_position_size': round(sum(t['position_size'] for t in losing_trades) / len(losing_trades), 2),
                    'avg_losing_return_pct': round(sum(t['return_pct'] for t in losing_trades) / len(losing_trades), 2)
                })

    # Convert monthly profits to DataFrame
    profit_df = pd.DataFrame([
        {'Month': month, 'Profit': profit}
        for month, profit in monthly_profits.items()
    ])
    
    if not profit_df.empty:
        profit_df = profit_df.sort_values('Month')
        profit_df['Profit'] = profit_df['Profit'].round(2)
    
    return profit_df, yearly_metrics, quarterly_metrics, all_trades, all_dividends, all_options, yearly_stock_pnl

# Connect to database and get data
try:
    logging.info('Starting summary update')
    conn = sqlite3.connect(os.path.join(PROJECT_ROOT, 'data', 'souptrader.db'))
    df = pd.read_sql_query("SELECT * FROM trades", conn)
    conn.close()
    monthly_summary, yearly_metrics, quarterly_metrics, all_trades, all_dividends, all_options, yearly_stock_pnl = calculate_yearly_profits(df)
    logging.info('Calculated yearly profits and metrics')

    # connect to database for updating tables
    conn = sqlite3.connect(os.path.join(PROJECT_ROOT, 'data', 'souptrader.db'))
    cursor = conn.cursor()

    # populate monthly_summary
    # Upsert each row from DataFrame
    for _, row in monthly_summary.iterrows():
        cursor.execute('''
        INSERT INTO monthly_summary (month, realized_profit)
        VALUES (?, ?)
        ON CONFLICT(month) DO UPDATE SET realized_profit = excluded.realized_profit
        ''', (row['Month'], row['Profit']))
    logging.info('Updated monthly summary table')

    # quarterly_summary table
    # Write or update each row
    for quarter, metrics in quarterly_metrics.items():
        cursor.execute('''
            INSERT INTO quarterly_summary (
                quarter, realized_profit
            ) VALUES (?, ?)
            ON CONFLICT(quarter) DO UPDATE SET
                realized_profit = excluded.realized_profit
        ''', (
            quarter,
            round(metrics['total_profit'], 2)
        ))
    logging.info('Updated quarterly summary table')

    # yearly_summary table
    for year, metrics in yearly_metrics.items():
        cursor.execute('''
            INSERT INTO yearly_summary (
                year,
                realized_profit,
                total_fees,
                total_dividends,
                options_profit,
                total_trades_closed,
                win_rate,
                avg_trade_duration,
                avg_profit_per_trade,
                avg_position_size,
                simple_avg_return,
                position_weighted_return,
                avg_winning_position_size,
                avg_winning_return_pct,
                avg_losing_position_size,
                avg_losing_return_pct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(year) DO UPDATE SET
                realized_profit = excluded.realized_profit,
                total_fees = excluded.total_fees,
                total_dividends = excluded.total_dividends,
                options_profit = excluded.options_profit,
                total_trades_closed = excluded.total_trades_closed,
                win_rate = excluded.win_rate,
                avg_trade_duration = excluded.avg_trade_duration,
                avg_profit_per_trade = excluded.avg_profit_per_trade,
                avg_position_size = excluded.avg_position_size,
                simple_avg_return = excluded.simple_avg_return,
                position_weighted_return = excluded.position_weighted_return,
                avg_winning_position_size = excluded.avg_winning_position_size,
                avg_winning_return_pct = excluded.avg_winning_return_pct,
                avg_losing_position_size = excluded.avg_losing_position_size,
                avg_losing_return_pct = excluded.avg_losing_return_pct
        ''', (
            year,
            round(metrics.get('total_profit', 0), 2),
            round(metrics.get('total_fees', 0), 2),
            round(metrics.get('total_divs', 0), 2),
            round(metrics.get('total_options', 0), 2),
            int(metrics.get('total_trades', 0)),
            round(metrics.get('win_rate', 0), 2),
            round(metrics.get('avg_trade_duration', 0), 2),
            round(metrics.get('avg_profit_per_trade', 0), 2),
            round(metrics.get('avg_position_size', 0), 2),
            round(metrics.get('avg_return_pct', 0), 2),
            round(metrics.get('position_weighted_return', 0), 2),
            round(metrics.get('avg_winning_position_size', 0), 2),
            round(metrics.get('avg_winning_return_pct', 0), 2),
            round(metrics.get('avg_losing_position_size', 0), 2),
            round(metrics.get('avg_losing_return_pct', 0), 2)
        ))
    logging.info('Updated yearly summary table')

    # Insert or update each ticker/year combo
    for ticker, yearly_data in yearly_stock_pnl.items():
        for year, profit in yearly_data.items():
            cursor.execute('''
                INSERT INTO yearly_stock_pnl (ticker, year, profit)
                VALUES (?, ?, ?)
                ON CONFLICT(ticker, year) DO UPDATE SET
                    profit = excluded.profit
            ''', (
                ticker,
                int(year),
                round(profit, 2)
            ))
    logging.info('Updated yearly stock P&L table')

    # Commit and close
    conn.commit()
    conn.close()
    logging.info('Summary update completed successfully')
except Exception as e:
    logging.error(f'Error in summary update: {str(e)}')
    if 'conn' in locals():
        conn.close()