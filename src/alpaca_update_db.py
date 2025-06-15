import logging
import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    filename=os.path.join(PROJECT_ROOT, 'logs', 'alpaca_update.log'),
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

from requests import get
import json
import pandas as pd
import sqlite3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

APCA_API_KEY_ID = os.getenv('APCA_API_KEY_ID')
APCA_API_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')

try:
    logging.info('Starting Alpaca DB update')
    conn = sqlite3.connect(os.path.join(PROJECT_ROOT, 'data', 'souptrader.db'))

    # get ALL existing order IDs
    order_ids_db = pd.read_sql_query("""
        SELECT id FROM trades 
        WHERE LOWER(asset_class) NOT LIKE '%fee%'
        """, conn)['id'].tolist()

    # get ALL existing fee IDs
    fee_ids_db = pd.read_sql_query("""
        SELECT id FROM trades 
        WHERE LOWER(asset_class) LIKE '%fee%'
        """, conn)['id'].tolist()

    # get ALL existing fund IDs
    funds_ids_db = pd.read_sql_query("""
        SELECT id FROM funds 
        """, conn)['id'].tolist()

    HEADER = {
        'APCA-API-SECRET-KEY': APCA_API_SECRET_KEY,
        'APCA-API-KEY-ID': APCA_API_KEY_ID,
        'Accept': 'application/json'
        }

    # Getting the last 100 orders
    PARAMS = {
        'status': "closed",
        'limit': 100
    }

    url = "https://api.alpaca.markets/v2/orders"
    response = get(url, headers=HEADER, params=PARAMS)
    data = response.json()
    df = pd.DataFrame(data)

    df = df[df['status'] == 'filled']

    # list of ids from alpaca
    id_list = df['id'].tolist()

    # compare the two lists of ids
    missing_ids = set(id_list) - set(order_ids_db)

    # now filter to the rows we need to add
    df = df[df['id'].isin(missing_ids)]

    cols = ['id', 'symbol', 'asset_class', 'qty', 'filled_avg_price', 'side', 'status', 'filled_at']

    df = df[cols]

    # Convert qty and filled_avg_price to numeric values
    df['qty'] = pd.to_numeric(df['qty'], errors='coerce')
    df['filled_avg_price'] = pd.to_numeric(df['filled_avg_price'], errors='coerce')

    # Add a new column for the multiplication
    df['amount'] = df['qty'] * df['filled_avg_price']

    # multiply options by 100
    df['amount'] = df.apply(lambda x: x['amount'] * 100 if x['asset_class'] == 'us_option' else x['amount'], axis=1)

    df['platform'] = 'alpaca'
    df['category'] = 'short_term'

    df_orders = df.rename(columns={
        'filled_avg_price': 'price',
        'filled_at': 'filled_time'
        # Add any other column renames here
    })

    # remove the extra characters with options contracts
    df_orders['symbol'] = df_orders['symbol'].str.extract(r'([A-Z]+)(?:\d+|$)')[0]

    ##### Moving on to fees here
    PARAMS = {
        'activity_types': "FEE",
    }
    url = "https://api.alpaca.markets/v2/account/activities"

    response = get(url, headers=HEADER, params=PARAMS)
    data = response.json()
    df = pd.DataFrame(data)

    # only keep executed status (extra step)
    df = df[df["status"] == "executed"]

    # list of ids from alpaca
    id_list = df['id'].tolist()

    # compare the two lists of ids
    missing_ids = set(id_list) - set(fee_ids_db)

    # now filter to the rows we need to add
    df = df[df['id'].isin(missing_ids)]

    cols = ['id', 'date', 'net_amount', 'description']

    df = df[cols]

    # Convert amount to numeric and then take absolute value
    df['net_amount'] = pd.to_numeric(df['net_amount'])
    df['net_amount'] = df['net_amount'].abs()
    df['platform'] = 'alpaca'
    df['asset_class'] = 'fee'

    df_fees = df.rename(columns={
        'date': 'filled_time',
        'net_amount': 'amount',
    })

    ###### stack the two dataframes fees and orders
    df = pd.concat([df_orders, df_fees], ignore_index=True)
    df['amount'] = df['amount'].round(4)
    df['price'] = df['price'].round(4)
    df['qty'] = df['qty'].round(4)
    df_all = df

    ##### Moving on to transfers here
    PARAMS = {
        'activity_types': 'CSD,CSW'
    }
    url = "https://api.alpaca.markets/v2/account/activities"

    response = get(url, headers=HEADER, params=PARAMS)
    data = response.json()
    df = pd.DataFrame(data)
    df = df[df["status"] == "executed"]

    # list of ids from alpaca
    id_list = df['id'].tolist()

    # compare the two lists of ids
    missing_ids = set(id_list) - set(funds_ids_db)

    # now filter to the rows we need to add
    df = df[df['id'].isin(missing_ids)]

    cols = ['id', 'activity_type', 'net_amount', 'date']
    df = df[cols]
    df['net_amount'] = pd.to_numeric(df['net_amount'])
    df['net_amount'] = df['net_amount'].abs()
    df['platform'] = 'alpaca'

    df_funds = df.rename(columns={
        'activity_type': 'type',
        'net_amount': 'amount',
    })

    # Insert trades with conflict handling
    if not df_all.empty:
        df_all.to_sql('trades', conn, if_exists='append', index=False)
        logging.info(f'Added {len(df_all)} new trades')
    else:
        logging.info('No new trades to add')

    # Insert funds with conflict handling
    if not df_funds.empty:
        df_funds.to_sql('funds', conn, if_exists='append', index=False)
        logging.info(f'Added {len(df_funds)} new fund entries')
    else:
        logging.info('No new fund entries to add')

    conn.close()
    logging.info('Alpaca DB update completed successfully')
except Exception as e:
    logging.error(f'Error in Alpaca DB update: {str(e)}')
    if 'conn' in locals():
        conn.close()