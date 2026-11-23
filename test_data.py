import requests
import pandas as pd
import datetime
import os
from dotenv import load_dotenv
import hopsworks

load_dotenv()

OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")
AQICN_KEY = os.getenv("AQICN_KEY")
HOPSWORKS_KEY = os.getenv("HOPSWORKS_KEY")

CITY = "Karachi"
LAT = 24.8607
LON = 67.0011

print(f"--- Fetching live data for {CITY} ---")

# --- 1. FETCH LIVE WEATHER (Open-Meteo Primary, OpenWeather Fallback) ---
temp, humidity, wind_speed, rain, uv_index = None, None, None, 0.0, 0.0

try:
    print("Attempting to fetch from Open-Meteo (Primary)...")
    meteo_url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,uv_index&timezone=auto"
    response = requests.get(meteo_url).json()
    
    current = response['current']
    temp = current['temperature_2m']
    humidity = current['relative_humidity_2m']
    wind_speed = current['wind_speed_10m']
    rain = current['precipitation']
    uv_index = current['uv_index']
    print("Open-Meteo Success!")

except Exception as e:
    print(f"Open-Meteo failed: {e}. Falling back to OpenWeather...")
    
    weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={OPENWEATHER_KEY}&units=metric"
    weather_data = requests.get(weather_url).json()
    
    temp = weather_data['main']['temp']
    humidity = weather_data['main']['humidity']
    wind_speed = weather_data['wind']['speed']
    rain = weather_data.get('rain', {}).get('1h', 0.0)
    uv_index = 0.0
    print("OpenWeather Fallback Success!")

# --- 2. FETCH AQI (AQICN) ---
print("Fetching AQI from AQICN...")
aqi_url = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_KEY}"
aqi_data = requests.get(aqi_url).json()
aqi = aqi_data['data']['aqi']

# --- 3. CREATE TABLE ---
now = datetime.datetime.now()
data = {
    "timestamp": [now],
    "city": [CITY],
    "temperature_c": [temp],
    "humidity": [humidity],
    "wind_speed": [wind_speed],
    "precipitation_mm": [rain],
    "uv_index": [uv_index],
    "aqi": [aqi]
}
df = pd.DataFrame(data)
print("\n--- CLEAN DATA ---")
print(df)

# --- 4. PUSH TO HOPSWORKS (With Local Fallback) ---
print("\nConnecting to Hopsworks...")
try:
    project = hopsworks.login(
        project="AgentWeather_L", 
        host="eu-west.cloud.hopsworks.ai", 
        port=443,
        api_key_value=HOPSWORKS_KEY
    )
    fs = project.get_feature_store()
    
    fg = fs.get_or_create_feature_group(
        name="weather_features",
        version=1,
        primary_key=["city", "timestamp"], 
        description="Live Weather and AQI data",
        event_time="timestamp"
    )
    
    print("Attempting to push to Hopsworks...")
    fg.insert(df, write_options={"start_offline_materialization": False})
    print("Success! Data pushed to Hopsworks Cloud.")

except Exception as e:
    print(f"Hopsworks failed (Windows HDFS issue). Saving locally...")
    df.to_csv("weather_history.csv", mode='a', header=False, index=False)
    print("Success! Data saved to weather_history.csv")