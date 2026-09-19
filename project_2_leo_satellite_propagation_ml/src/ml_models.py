import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "propagation_dataset.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)

# -----------------------------
# Temporal train-test split
# -----------------------------
df = df.sort_values("time_index").reset_index(drop=True)

split_ratio = 0.7
split_idx = int(len(df) * split_ratio)

train_df = df.iloc[:split_idx]
test_df  = df.iloc[split_idx:]

features = [
    "elevation_deg",
    "slant_range_km",
    "doppler_proxy",
    "rain_mm_hr",
    "humidity_pct",
    "temperature_c"
]

X_train = train_df[features]
y_train = train_df["excess_loss_db"]

X_test = test_df[features]
y_test = test_df["excess_loss_db"]

# Store feature names safely
feature_names = features.copy()

# -----------------------------
# ML model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)

print(f"Temporal Test RMSE: {rmse:.3f} dB")
print(f"Temporal Test MAE : {mae:.3f} dB")

# -----------------------------
# Save evaluation data (CRITICAL FIX)
# -----------------------------
eval_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": y_pred
})

eval_df.to_csv(
    os.path.join(RESULTS_DIR, "evaluation_data.csv"),
    index=False
)

# -----------------------------
# Save model
# -----------------------------
joblib.dump(
    model,
    os.path.join(MODEL_DIR, "rf_excess_loss_model.pkl")
)

print("ML model saved.")

# -----------------------------
# Feature Importance Plot
# -----------------------------
importances = model.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(7, 5))
plt.barh(importance_df["feature"], importance_df["importance"])
plt.xlabel("Importance Score")
plt.title("Feature Importance for Excess Loss Prediction")
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "feature_importance.png"),
    dpi=300
)
plt.close()
