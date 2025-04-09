import sqlite3
import pandas as pd 


def create_database(data):
    """Create SQLite database and store preprocessed data"""
    conn = sqlite3.connect('traffic_data.db')
    
    # Store the full dataset
    data.to_sql('traffic_data', conn, if_exists='replace', index=False)
    
    # Create aggregated views for visualization
    hourly_traffic = data.groupby('hour')['traffic_volume'].mean().reset_index()
    hourly_traffic.to_sql('hourly_traffic', conn, if_exists='replace', index=False)
    
    weekday_traffic = data.groupby('weekday')['traffic_volume'].mean().reset_index()
    weekday_traffic.to_sql('weekday_traffic', conn, if_exists='replace', index=False)
    
    monthly_traffic = data.groupby('month')['traffic_volume'].mean().reset_index()
    monthly_traffic.to_sql('monthly_traffic', conn, if_exists='replace', index=False)
    
    day_traffic = data.groupby('month_day')['traffic_volume'].mean().reset_index()
    day_traffic.to_sql('day_traffic', conn, if_exists='replace', index=False)
    
    conn.close()
    print("Database created successfully!")

if __name__ == "__main__":
    # Process data
    data = pd.read_csv("cleaned_data.csv")
    
    # Create database
    create_database(data)