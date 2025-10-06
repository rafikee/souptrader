import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import set_with_dataframe
import datetime as dt
import pandas as pd
from finvizfinance.screener.overview import Overview
import json
import pytz
import os
from aatinaa import sharia_status
import time
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Alpaca API configuration
APCA_API_KEY_ID = os.getenv('APCA_API_KEY_ID')
APCA_API_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')

# Create Alpaca client
alpaca_client = StockHistoricalDataClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY)


def get_stock_data(ticker, start_date, end_date, max_retries=3):
    """Get stock data using Alpaca API (much more reliable and higher rate limits)"""
    
    for attempt in range(max_retries):
        try:
            # Add small delay between requests to be respectful
            if attempt > 0:
                time.sleep(0.1)  # Very small delay since Alpaca has high rate limits
            
            print(f"Fetching data for {ticker} from Alpaca...")
            
            # Create request for daily bars
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            
            # Get the data
            bars = alpaca_client.get_stock_bars(request)
            

            
            if not bars or len(bars.data) == 0:
                print(f"Warning: No data available for {ticker}")
                return None
            
            # Try to access the raw data directly
            try:
                # Get the raw data from the bars object
                raw_data = bars.data
                
                # Alpaca returns data as a dict with ticker as key
                if isinstance(raw_data, dict) and ticker in raw_data:
                    # Extract the actual bar data for this ticker
                    ticker_data = raw_data[ticker]
                    
                    # Convert raw data to DataFrame manually
                    data_list = []
                    for item in ticker_data:
                        if hasattr(item, 'timestamp') and hasattr(item, 'close'):
                            # This is a bar object
                            data_list.append({
                                'timestamp': item.timestamp,
                                'open': item.open,
                                'high': item.high,
                                'low': item.low,
                                'close': item.close,
                                'volume': item.volume
                            })
                        elif isinstance(item, dict):
                            # This is already a dict
                            data_list.append(item)
                        else:
                            print(f"Debug: Unexpected item type: {type(item)}, value: {item}")
                    
                    if not data_list:
                        print(f"Warning: No valid data found for {ticker}")
                        return None
                    
                    # Create DataFrame from the data
                    df = pd.DataFrame(data_list)
                    
                    # Set timestamp as index
                    if 'timestamp' in df.columns:
                        df['Date'] = pd.to_datetime(df['timestamp'])
                        df.set_index('Date', inplace=True)
                        df = df.drop('timestamp', axis=1)
                    
                    # Rename columns to match our expected format
                    column_mapping = {
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume'
                    }
                    
                    for alpaca_col, our_col in column_mapping.items():
                        if alpaca_col in df.columns:
                            df[our_col] = df[alpaca_col]
                    
                    # Ensure we have the Close column
                    if 'Close' not in df.columns:
                        print(f"Warning: No Close price data for {ticker}")
                        return None
                    
                    # Validate that we have enough data
                    if len(df) < 50:
                        print(f"Warning: Insufficient data for {ticker} (only {len(df)} days)")
                        return None
                    
                    print(f"Successfully got data for {ticker} ({len(df)} days)")
                    return df
                else:
                    print(f"Warning: No data found for ticker {ticker} in response")
                    return None
                
            except Exception as df_error:
                print(f"DataFrame conversion error for {ticker}: {str(df_error)}")
                if attempt < max_retries - 1:
                    continue
                else:
                    return None
            
        except Exception as e:
            error_msg = str(e)
            print(f"Full error for {ticker}: {error_msg}")
            print(f"Error type: {type(e)}")
            
            if "not found" in error_msg.lower() or "404" in error_msg:
                print(f"Warning: {ticker} not found in Alpaca database")
                return None
            elif "rate limit" in error_msg.lower() or "429" in error_msg:
                if attempt < max_retries - 1:
                    print(f"Rate limited for {ticker}, attempt {attempt + 1}/{max_retries}")
                    time.sleep(1 * (attempt + 1))  # Small delay for rate limiting
                else:
                    print(f"Failed to get data for {ticker} after {max_retries} attempts: Rate limited")
                    return None
            else:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed for {ticker}: {error_msg}")
                    time.sleep(0.5 * (attempt + 1))
                else:
                    print(f"Failed to get data for {ticker} after {max_retries} attempts: {error_msg}")
                    return None
    
    return None


def main():
    # Load service account JSON locally
    with open("src/screener/service_account.json") as json_file:
        service_account_json = json.load(json_file)

    # To access Google Sheets
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        service_account_json, scope  # type: ignore
    )

    # we need to get roughly a year's worth of data to evaluate a stock for all the metrics
    now = dt.datetime.now()
    start = now - dt.timedelta(days=400)

    # Choose either Adjusted Close or Regular Close
    closing_types = ["Adj Close", "Close"]
    close = closing_types[1]

    # Use the FinViz API to get the first set of filters out of the way
    foverview = Overview()
    filters_dict = {
        "Market Cap.": "+Mid (over $2bln)",
        "Average Volume": "Over 1M",
        "200-Day Simple Moving Average": "Price above SMA200",
        "50-Day Simple Moving Average": "Price above SMA50",
        "52-Week High/Low": "30% or more above Low",
        "EPS growthqtr over qtr": "Over 20%",
        "Sales growthqtr over qtr": "Over 20%",
    }

    foverview.set_filter(filters_dict=filters_dict)
    finviz = foverview.screener_view()

    if isinstance(finviz, pd.DataFrame):
        finviz = finviz.drop(columns=["P/E"])  # we don't care about P/E

        # Get the DataFrame as list of dicts to loop through
        stocks = finviz.to_dict("records")
        print(f"\nFound {len(stocks)} stocks from Finviz")

    else:
        return "We couldn't get the stocks from FinViz"

    output = pd.DataFrame()  # Create the empty output DataFrame

    # Loop through all the stocks we got from FinViz
    stock_data = []
    
    # For testing, you can limit the number of stocks processed
    # Set to None to process all stocks, or set to a number (e.g., 5) for testing
    MAX_STOCKS_TO_PROCESS = None  # Process all stocks
    
    stocks_to_process = stocks[:MAX_STOCKS_TO_PROCESS] if MAX_STOCKS_TO_PROCESS else stocks
    print(f"Processing {len(stocks_to_process)} stocks (out of {len(stocks)} total)")
    
    for i, stock in enumerate(stocks_to_process):
        # added this short if statement because finviz changed Ticker to Ticker\n\n
        # if they ever fix it this shoould still work
        if "Ticker\n\n" in stock:
            stock["Ticker"] = stock.pop("Ticker\n\n")

        ticker = stock["Ticker"]
        print(f"\nProcessing {ticker}... ({i+1}/{len(stocks_to_process)})")
        
        # Add minimal delay between requests since Alpaca has high rate limits
        if i > 0:
            time.sleep(0.1)  # 0.1 second delay between requests

        try:
            # Get stock data using our new function
            df = get_stock_data(ticker, start, now)

            if df is None:
                continue

            # Get most recently close (last row in df)
            currentClose = df[close].iloc[-1]

            # Get SMA for 50, 150, 200 and round to 2 decimals
            moving_average_50 = round(df.tail(50)[close].mean(), 2)
            moving_average_150 = round(df.tail(150)[close].mean(), 2)
            moving_average_200 = round(df.tail(200)[close].mean(), 2)

            # Get the SMA_200 20 trading days ago to check if it's been trending up since
            moving_average_200_20 = round(df[close][-221:-21].mean(), 2)

            # Go back a year's worth of trading days and get the min/max
            low_of_52week = round(min(df.tail(255)[close]), 2)
            high_of_52week = round(max(df.tail(255)[close]), 2)

            # Some of the following conditions are already checked in Finviz but I'll do a double check here in case the code is updated
            conditions = []
            # Condition 1: Current Price > 150 SMA and > 200 SMA
            cond_1 = currentClose > moving_average_150 > moving_average_200
            conditions.append(cond_1)

            # Condition 2: 200 SMA trending up for at least 1 month (ideally 4-5 months)
            cond_2 = True if (moving_average_200 > moving_average_200_20) else False
            conditions.append(cond_2)

            # Condition 3: 50 SMA > 150 SMA and 50 SMA > 200 SMA
            cond_3 = (
                True
                if (moving_average_50 > moving_average_150 > moving_average_200)
                else False
            )
            conditions.append(cond_3)

            # Condition 4: Current Price > 50 SMA
            cond_4 = True if (currentClose > moving_average_50) else False
            conditions.append(cond_4)

            # Condition 5: Current Price is at least 30% above 52 week low (Many of the best are up 100-300% before coming out of consolidation)
            cond_5 = True if (currentClose >= (1.3 * low_of_52week)) else False
            conditions.append(cond_5)

            # Condition 6: Current Price is within 25% of 52 week high
            cond_6 = True if (currentClose >= (0.75 * high_of_52week)) else False
            conditions.append(cond_6)

            # count how many conditions are true
            count = sum(1 for cond in conditions if cond)

            # Set values and add to output DataFrame
            stock["cond count"] = count
            stock["cond 1"] = cond_1
            stock["cond 2"] = cond_2
            stock["cond 3"] = cond_3
            stock["cond 4"] = cond_4
            stock["cond 5"] = cond_5
            stock["cond 6"] = cond_6

            stock_data.append(stock)
            print(f"Successfully processed {ticker}")

        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")

    if not stock_data:
        print("\nNo stocks were successfully processed!")
        return "No stocks were successfully processed"

    output = pd.DataFrame(stock_data)  # type: ignore
    print(f"\nSuccessfully processed {len(output)} stocks")

    # set the type of each column for formatting
    output = output.astype(
        {
            "cond count": "int",
            "cond 1": "int",
            "cond 2": "int",
            "cond 3": "int",
            "cond 4": "int",
            "cond 5": "int",
            "cond 6": "int",
            "Volume": "int",
            "Market Cap": "int",
        }
    )

    # Add the sharia status to the output
    for index, row in output.iterrows():
        if row["cond count"] > 0:
            ticker = row["Ticker"]
            sharia = sharia_status(ticker)
            output.at[index, "Sharia"] = sharia

    # Round for appearance
    output = output.round({"Volume": 0, "Market Cap": 0, "Price": 2, "Change": 4})
    col_count = len(output.columns)  # how many columns are in the df
    # convert col count to column letter in sheet for formatting
    letter = chr(col_count + 64)
    first_row = f"A1:{letter}1"  # the range for first row in the sheet

    # get New York time to put in the sheet name
    newYorkTz = pytz.timezone("America/New_York")
    timeInNewYork = dt.datetime.now(newYorkTz)
    newYorkTz = timeInNewYork.strftime("%m-%d-%Y")

    # open up the Google Sheets page delete and create a new one
    gc = gspread.authorize(creds)
    gs = gc.open("Stock Screener")
    worksheet_list = gs.worksheets()
    for sheet in worksheet_list:
        if "Screener" in sheet.title:
            sheet = gs.worksheet(sheet.title)
            gs.del_worksheet(sheet)
    sheet = gs.add_worksheet(title=f"Screener {newYorkTz}", rows=1, cols=1)

    # Drop in the data from the data frame to the sheet
    set_with_dataframe(
        worksheet=sheet,
        dataframe=output,
        include_index=False,
        include_column_header=True,
        resize=True,
    )

    # Format the header row
    sheet.format(
        first_row,
        {
            "backgroundColor": {
                "red": 216 / 255,
                "green": 229 / 255,
                "blue": 252 / 255,
            },
            "horizontalAlignment": "CENTER",
            "textFormat": {"fontSize": 12, "bold": True},
        },
    )

    # Format cells for their numbers
    cell = sheet.find("Change")
    if cell:
        letter = chr(cell.col + 64)  # convert col num to letter
        sheet.format(f"{letter}:{letter}", {"numberFormat": {"type": "PERCENT"}})

    cell = sheet.find("Market Cap")
    if cell:
        letter = chr(cell.col + 64)  # convert col num to letter
        sheet.format(
            f"{letter}:{letter}",
            {"numberFormat": {"type": "NUMBER", "pattern": '0,,"M"'}},
        )

    cell = sheet.find("Price")
    if cell:
        letter = chr(cell.col + 64)  # convert col num to letter
        sheet.format(f"{letter}:{letter}", {"numberFormat": {"type": "CURRENCY"}})

    cell = sheet.find("Volume")
    if cell:
        letter = chr(cell.col + 64)  # convert col num to letter
        sheet.format(
            f"{letter}:{letter}",
            {"numberFormat": {"type": "NUMBER", "pattern": '0.0,,"M"'}},
        )

    # Sort and Filter. Filter out to only show the ones that met all 6 conditions, and sort by Volume

    # Here we identify the 3 different columns we need to filter and sort by
    cond_count_cell = sheet.find("cond count")
    if cond_count_cell:
        cond_count_cell = cond_count_cell.col

    vol_cell = sheet.find("Volume")
    if vol_cell:
        vol_cell = vol_cell.col

    sharia_cell = sheet.find("Sharia")
    if sharia_cell:
        sharia_cell = sharia_cell.col

    if vol_cell and cond_count_cell and sharia_cell:
        filterSettings = {
            "range": {"sheetId": sheet.id},
            "filterSpecs": [
                {
                    "filterCriteria": {"hiddenValues": ["0,", "1", "2", "3", "4", "5"]},
                    "columnIndex": cond_count_cell - 1,
                },
                {
                    "filterCriteria": {"hiddenValues": ["FAILED"]},
                    "columnIndex": sharia_cell - 1,
                },
            ],
            "sortSpecs": [
                {
                    "sortOrder": "DESCENDING",
                    "dimensionIndex": vol_cell - 1,
                }
            ],
        }

        gs.batch_update({"requests": [{"setBasicFilter": {"filter": filterSettings}}]})
        sheet.columns_auto_resize(0, col_count)  # resize the columns

    # hide the conditions columns (we have 6 of them)
    # we added the sharia column at the end so hide the 6 before it
    sheet.hide_columns(col_count - 7, col_count - 1)
    return "Yay Stocks!"


# running locally to test
if __name__ == "__main__":
    import tracemalloc

    tracemalloc.start()
    main()
    print(tracemalloc.get_traced_memory())
    tracemalloc.stop()
