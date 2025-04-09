import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect('database/traffic_data.db')

# View available tables
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
print("Tables in database:\n", tables)

table_name = 'traffic_data'

# Get column names
cursor = conn.execute(f"PRAGMA table_info({table_name});")
columns = [row[1] for row in cursor.fetchall()]
print(f"\nColumns in '{table_name}' table:\n", columns)


# Read a specific table, e.g., 'traffic_data'
df = pd.read_sql_query("SELECT * FROM traffic_data LIMIT 5;", conn)
print("\nSample data from traffic_data table:\n", df)

conn.close()
