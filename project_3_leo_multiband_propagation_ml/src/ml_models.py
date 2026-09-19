import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# -----------------------------
# Project paths
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "multiband_ml_dataset.csv")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "rf_multiband_excess_loss.pkl")

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)

# -----------------------------
# Temporal split (important)
# -----------------------------
df = df.sort_values("time_index").reset_index(drop=True)

split_ratio = 0.7
split_idx = int(len(df) * split_ratio)

train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

# -----------------------------
# Features & target
# -----------------------------
features = [
    "elevation_deg",
    "slant_range_km",
    "doppler_proxy",
    "time_index",
    "frequency_ghz",
    "rain_mm_hr",
    "humidity_pct",
    "temperature_c"
]

X_train = train_df[features]
y_train = train_df["excess_loss_db"]

X_test = test_df[features]
y_test = test_df["excess_loss_db"]

# -----------------------------
# Model definition
# -----------------------------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=14,
    random_state=42,
    n_jobs=-1
)

# -----------------------------
# Train
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# Predict & evaluate
# -----------------------------
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"Temporal + Multi-band Test RMSE: {rmse:.4f} dB")
print(f"Temporal + Multi-band Test MAE : {mae:.4f} dB")

# -----------------------------
# Save model
# -----------------------------
joblib.dump(model, MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")
