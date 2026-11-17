import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os

print("--- Starting Training Pipeline ---")

# 1. Try to load the real data
try:
    df = pd.read_csv("weather_history.csv")
except FileNotFoundError:
    print("Error: weather_history.csv not found. Run test_data.py first.")
    exit()

# 2. Check if we have enough data. If not, generate dummy data for testing.
if len(df) < 50:
    print(f"Warning: Only {len(df)} row(s) of real data found.")
    print("Generating 100 rows of dummy data just so we can test the ML code...\n")
    
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='h')
    
    # Create fake data for Karachi
    df = pd.DataFrame({
        'humidity': np.random.normal(60, 10, 100), 
        'aqi': np.random.normal(150, 30, 100),
        'temperature_c': np.random.normal(28, 5, 100) 
    })
    df['timestamp'] = dates
    
    # Create the Target: What will the temperature be in 24 hours?
    df['target_temp_24h'] = df['temperature_c'].shift(-24)
    
    # Drop the last 24 rows because we don't know the future target for them
    df = df.dropna()

# 3. Setup Features (X) and Target (y)
# We will use Humidity and AQI to predict the Temperature 24h from now
X = df[['humidity', 'aqi']] 
y = df['target_temp_24h']

# 4. Time-based split (Train on first 80%, test on last 20%)
split_index = int(len(df) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 5. Train Models
print("Training models...")

# Model 1: Naive Baseline (Just guess 28 degrees every time)
naive_pred = [28.0] * len(y_test)

# Model 2: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Model 3: Random Forest
rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# 6. Evaluate (Calculate Errors)
def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name} -> MAE: {mae:.2f} | RMSE: {rmse:.2f}")

print("\n--- Model Results (Lower is better) ---")
evaluate(y_test, naive_pred, "Naive Baseline ")
evaluate(y_test, lr_pred,    "Linear Regression")
evaluate(y_test, rf_pred,    "Random Forest  ")

# 7. Save the best model
if not os.path.exists("models"):
    os.makedirs("models")

joblib.dump(rf_model, "models/best_temperature_model.pkl")
print("\nModel saved successfully to /models/best_temperature_model.pkl")