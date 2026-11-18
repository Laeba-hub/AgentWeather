import requests
import pandas as pd
from datetime import datetime, timedelta

# Karachi coordinates
LAT = 24.8607
LON = 67.0011

# Fetch 1 year of data
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

print(f"Fetching historical data from {start_str} to {end_str}...")

# Open-Meteo Historical API (No key required!)
url = f"https://archive-api.open-meteo.com/v1/archive?latitude={LAT}&longitude={LON}&start_date={start_str}&end_date={end_str}&hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m&timezone=auto"

response = requests.get(url)
data = response.json()

# Convert to Pandas DataFrame
df = pd.DataFrame(data['hourly'])
df.rename(columns={
    'time': 'timestamp',
    'temperature_2m': 'temperature_c',
    'relative_humidity_2m': 'humidity',
    'precipitation': 'precipitation_mm',
    'wind_speed_10m': 'wind_speed'
}, inplace=True)

# Add city and placeholders for AQI/UV
df['city'] = 'Karachi'
df['aqi'] = 100 
df['uv_index'] = 0.0 

df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"Successfully fetched {len(df)} rows of historical data.")
print(df.head())

# Save to CSV (This is your local Feature Store)
df.to_csv("historical_data.csv", index=False)
print("\nSaved to historical_data.csv")