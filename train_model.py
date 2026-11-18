import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
import joblib
import os

print("--- Starting Professional Training Pipeline ---")

# 1. Load the REAL historical data
df = pd.read_csv("historical_data.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp') # Ensure it's in time order

# 2. FEATURE ENGINEERING (The Secret Sauce)
# Create the target
df['target_temp_24h'] = df['temperature_c'].shift(-24)

# Add Lag Features (What was the temp 24h, 48h, 72h ago?)
df['temp_lag_24h'] = df['temperature_c'].shift(24)
df['temp_lag_48h'] = df['temperature_c'].shift(48)
df['temp_lag_72h'] = df['temperature_c'].shift(72)

# Add Rolling Averages (Average of the last 24 hours and last 72 hours)
df['temp_rolling_24h'] = df['temperature_c'].rolling(window=24).mean()
df['temp_rolling_72h'] = df['temperature_c'].rolling(window=72).mean()

# Add Time-based features
df['hour'] = df['timestamp'].dt.hour
df['month'] = df['timestamp'].dt.month
df['day_of_year'] = df['timestamp'].dt.dayofyear

# Drop rows with NaN (because shifting creates empty rows at the start/end)
df = df.dropna()
print(f"Loaded {len(df)} rows after feature engineering.")

# 3. Setup Features and Target
features = ['humidity', 'wind_speed', 'precipitation_mm', 
            'temp_lag_24h', 'temp_lag_48h', 'temp_lag_72h', 
            'temp_rolling_24h', 'temp_rolling_72h', # <-- NEW
            'hour', 'month']
X = df[features]
y = df['target_temp_24h']

# 4. Time-based split
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

# ==========================================
# MODEL 1: NAIVE BASELINE
# ==========================================
# Baseline: Predict tomorrow = today's temp (which is the 24h lag)
naive_pred = test_df['temp_lag_24h'].values

# ==========================================
# MODEL 2: STATISTICAL (Prophet)
# ==========================================
print("Training Prophet...")
prophet_train = train_df[['timestamp', 'temperature_c']].rename(columns={'timestamp': 'ds', 'temperature_c': 'y'})
prophet_model = Prophet(daily_seasonality=True, yearly_seasonality=True)
prophet_model.fit(prophet_train)

future = pd.DataFrame({'ds': test_df['timestamp']})
prophet_pred = prophet_model.predict(future)['yhat'].values

# ==========================================
# MODEL 3: CLASSICAL ML (Random Forest)
# ==========================================
print("Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# ==========================================
# MODEL 4: DEEP LEARNING (Neural Network)
# ==========================================
print("Training Neural Network...")
nn_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
nn_model.fit(X_train, y_train)
nn_pred = nn_model.predict(X_test)

# ==========================================
# EVALUATION
# ==========================================
def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name:20} -> MAE: {mae:.2f} | RMSE: {rmse:.2f}")
    return mae

print("\n--- Model Results (Lower is better) ---")
mae_naive = evaluate(y_test, naive_pred, "Naive Baseline")
mae_prophet = evaluate(y_test, prophet_pred, "Prophet (Statistical)")
mae_rf = evaluate(y_test, rf_pred, "Random Forest (ML)")
mae_nn = evaluate(y_test, nn_pred, "Neural Net (Deep Learning)")

# ==========================================
# SAVE THE BEST MODEL
# ==========================================
model_scores = {
    "Prophet": (mae_prophet, prophet_model),
    "Random Forest": (mae_rf, rf_model),
    "Neural Network": (mae_nn, nn_model)
}

best_model_name = min(model_scores, key=lambda k: model_scores[k][0])
best_model_obj = model_scores[best_model_name][1]
best_mae = model_scores[best_model_name][0]

print(f"\nThe best trained model is: {best_model_name} (MAE: {best_mae:.2f})")

if best_mae < mae_naive:
    print("SUCCESS: The trained model has beaten the Naive Baseline!")
else:
    print("WARNING: The Naive Baseline is still better.")

if not os.path.exists("models"):
    os.makedirs("models")

joblib.dump(best_model_obj, "models/best_temperature_model.pkl")
print(f"Best model saved to /models/best_temperature_model.pkl")