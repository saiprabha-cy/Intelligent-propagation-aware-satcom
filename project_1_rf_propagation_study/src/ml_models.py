import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("data/propagation_dataset.csv")

# -------------------------
# Define EXCESS LOSS target
# -------------------------
df["excess_loss_db"] = df["total_path_loss_db"] - df["fspl_db"]

# -------------------------
# Features (no leakage)
# -------------------------
X = df[[
    "distance_km",
    "rain_mm_hr",
    "humidity_pct",
    "temperature_c"
]]

y = df["excess_loss_db"]

# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Models
# -------------------------
lr = LinearRegression()
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# -------------------------
# Train
# -------------------------
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# -------------------------
# Predict
# -------------------------
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)

# -------------------------
# Evaluation
# -------------------------
def evaluate(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{name} → RMSE: {rmse:.3f} dB | MAE: {mae:.3f} dB")

evaluate(y_test, y_pred_lr, "Linear Regression")
evaluate(y_test, y_pred_rf, "Random Forest")

# -------------------------
# Baseline: COST-231 excess loss
# -------------------------
cost231_excess = df.loc[y_test.index, "cost231_db"]
cost231_rmse = np.sqrt(mean_squared_error(y_test, cost231_excess))

print(f"COST-231 Excess Loss RMSE: {cost231_rmse:.3f} dB")

# -------------------------
# Save best model
# -------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/rf_excess_loss_model.pkl")

print("Corrected ML training completed.")
