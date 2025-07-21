import sqlite3
import pandas as pd

# Load CSV into DataFrame
df = pd.read_csv('uploadme.csv')

# Connect to SQLite database
conn = sqlite3.connect('souptrader.db')

# Append data to 'trades' table
df.to_sql('trades', conn, if_exists='append', index=False)

# Close connection
conn.close()