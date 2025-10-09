# Download Configuration

## Overview

The `download_config.yaml` file in this directory (`src/backtesting/`) controls the date ranges for all market data downloads.

## How It Works

### Single Configuration File

Instead of hardcoding dates in multiple scripts, all date ranges are now defined in `download_config.yaml`:

```yaml
trading_start_date: "2025-01-01"  # When you want to start trading/backtesting
trading_end_date: "today"          # End date (use "today" or specific date)
warmup_periods: 200                # Number of periods needed for indicators
```

### Automatic Date Calculation

The download scripts automatically calculate the appropriate start dates for each data type:

#### Daily Bars
- **Purpose**: Calculate 200-day SMMA and other daily indicators
- **Offset**: ~10 months before trading_start_date (200 trading days ≈ 286 calendar days)
- **Example**: If trading_start_date = 2025-01-01, downloads from ~2024-03-12

#### 5-Minute Bars
- **Purpose**: Calculate 200-period SMMA on 5-minute timeframe
- **Offset**: 3 calendar days before trading_start_date (200 periods ≈ 2.5 trading days)
- **Example**: If trading_start_date = 2025-01-01, downloads from 2024-12-29

#### 1-Minute Bars
- **Purpose**: Calculate 200-period SMMA on 1-minute timeframe
- **Offset**: 1 calendar day before trading_start_date (200 periods ≈ 0.5 trading days)
- **Example**: If trading_start_date = 2025-01-01, downloads from 2024-12-31

#### NBBO 1-Second Data
- **Purpose**: Execution and spread data (no indicators)
- **Offset**: None - starts exactly at trading_start_date
- **Example**: If trading_start_date = 2025-01-01, downloads from 2025-01-01

### End Date

All data types download until `trading_end_date`:
- Use `"today"` to always download up to the current date
- Or specify a date like `"2025-12-31"` for a fixed end date

## Usage

### Modifying Dates

1. Edit `src/backtesting/download_config.yaml`
2. Change the `trading_start_date` and/or `trading_end_date`
3. Run the download scripts - they'll automatically use the new dates

### Example Configurations

#### Backtest Full Year 2025
```yaml
trading_start_date: "2025-01-01"
trading_end_date: "2025-12-31"
warmup_periods: 200
```
This downloads:
- Daily: 2024-03-12 → 2025-12-31
- 5Min: 2024-12-29 → 2025-12-31
- 1Min: 2024-12-31 → 2025-12-31
- NBBO: 2025-01-01 → 2025-12-31

#### Live Trading (Always Current)
```yaml
trading_start_date: "2025-01-01"
trading_end_date: "today"
warmup_periods: 200
```
This downloads up to today's date, automatically updating each day.

## Benefits

1. **Single Source of Truth**: All date configuration in one file
2. **Automatic Calculation**: Scripts calculate appropriate offsets
3. **Easy Updates**: Change dates once, affects all downloads
4. **Clear Documentation**: Config file is self-documenting
5. **Flexible**: Use "today" for always-current data or fixed dates for reproducible backtests

