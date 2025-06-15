import pandas as pd
import sqlite3

# Read the CSV file
df = pd.read_csv('funds_import.csv', dtype=object)  # Replace with your CSV filename

# Connect to the database
conn = sqlite3.connect('souptrader.db')  # Replace with your database name

# Import the data into the trades table
# if_exists options: 'fail', 'replace', or 'append'
df.to_sql('funds', conn, if_exists='append', index=False)

# Close the connection
conn.close()

print(f"Imported {len(df)} rows into trades table")