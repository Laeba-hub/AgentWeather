import requests
import pandas as pd
import hopsworks
import datetime

# --- 1. KEYS ---
OPENWEATHER_KEY = "MY_SECRET_KEY"
AQICN_KEY = "MY_SECRET_KEY"
HOPSWORKS_KEY = "MY_SECRET_KEY"
# Choose a city (Must be a major city for AQICN to find it, e.g., London, Karachi, New York)
CITY = "Karachi"

print(f"Fetching data for {CITY}...\n")

# --- 2. FETCH DATA ---
weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={OPENWEATHER_KEY}&units=metric"
weather_data = requests.get(weather_url).json()

aqi_url = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_KEY}"
aqi_data = requests.get(aqi_url).json()

# --- 3. EXTRACT METRICS ---
temp = weather_data['main']['temp']
humidity = weather_data['main']['humidity']
wind_speed = weather_data['wind']['speed']
rain = weather_data.get('rain', {}).get('1h', 0.0) 
uv_index = 0.0 
aqi = aqi_data['data']['aqi']

# --- 4. CREATE TABLE (Add Timestamp!) ---
# Hopsworks requires a timestamp column to know when data was collected
now = datetime.datetime.now()

data = {
    "timestamp": [now], # <-- We added this
    "city": [CITY],
    "temperature_c": [temp],
    "humidity": [humidity],
    "wind_speed": [wind_speed],
    "precipitation_mm": [rain],
    "uv_index": [uv_index],
    "aqi": [aqi]
}
df = pd.DataFrame(data)

print("Data collected successfully. Connecting to Hopsworks...")

# --- 5. CONNECT TO HOPSWORKS ---
# Connect directly
project = hopsworks.login(
    project="AgentWeather_L", 
    host="eu-west.cloud.hopsworks.ai", 
    port=443,
    api_key_value=HOPSWORKS_KEY
)
fs = project.get_feature_store()



# --- 6. CREATE THE FEATURE GROUP (The Cloud Table) ---
# This creates the table structure in the cloud
fg = fs.get_or_create_feature_group(
    name="weather_features",
    version=1,
    primary_key=["city", "timestamp"], 
    description="Weather and AQI data for AgentWeather",
    event_time="timestamp" # Hopsworks uses this to sort time-series data
)

# --- 7. INSERT DATA ---
# --- 7. INSERT DATA (WITH LOCAL FALLBACK) ---
try:
    print("Attempting to push to Hopsworks...")
    fg.insert(df)
    print("Success! Data pushed to the Hopsworks Cloud.")
except Exception as e:
    print(f"\nHopsworks insert failed (Windows HDFS issue).")
    print("Executing local fallback...")
    
    # Save to a local CSV file to keep the project moving
    df.to_csv("weather_history.csv", mode='a', header=False, index=False)
    print("Success! Data saved locally to weather_history.csv")

# --- 8. SAVE MODEL/GRAPH LOCALLY (Simulating Model Registry) ---
import os
if not os.path.exists("models"):
    os.makedirs("models")