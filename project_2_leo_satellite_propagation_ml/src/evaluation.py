import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib

os.makedirs("results", exist_ok=True)

eval_df = pd.read_csv("results/evaluation_data.csv")
y_test = eval_df["y_true"].values
y_pred = eval_df["y_pred"].values

# -----------------------------
# Load dataset & model
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

data_path = os.path.join(PROJECT_ROOT, "data", "propagation_dataset.csv")
model_path = os.path.join(PROJECT_ROOT, "models", "rf_excess_loss_model.pkl")

df = pd.read_csv(data_path)
model = joblib.load(model_path)

# -----------------------------
# Temporal split (same as training)
# -----------------------------
df = df.sort_values("time_index")
split_idx = int(0.7 * len(df))

test_df = df.iloc[split_idx:]

features = [
    "elevation_deg",
    "slant_range_km",
    "doppler_proxy",
    "rain_mm_hr",
    "humidity_pct",
    "temperature_c"
]

X_test = test_df[features]
y_true = test_df["excess_loss_db"]
y_pred = model.predict(X_test)

error = y_pred - y_true

# -----------------------------
# Plot directory
# -----------------------------
plot_dir = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(plot_dir, exist_ok=True)

# -----------------------------
# 1. True vs Predicted
# -----------------------------
plt.figure()
plt.scatter(y_true, y_pred, alpha=0.5)
plt.plot([y_true.min(), y_true.max()],
         [y_true.min(), y_true.max()],
         linestyle="--")
plt.xlabel("True Excess Loss (dB)")
plt.ylabel("Predicted Excess Loss (dB)")
plt.title("True vs Predicted Excess Loss")
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "true_vs_predicted.png"), dpi=300)
plt.close()

# -----------------------------
# 2. Error vs Elevation
# -----------------------------
plt.figure()
plt.scatter(test_df["elevation_deg"], error, alpha=0.5)
plt.xlabel("Elevation Angle (deg)")
plt.ylabel("Prediction Error (dB)")
plt.title("Prediction Error vs Elevation Angle")
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "error_vs_elevation.png"), dpi=300)
plt.close()

# -----------------------------
# 3. Temporal error drift
# -----------------------------
plt.figure()
plt.plot(test_df["time_index"], error, alpha=0.7)
plt.xlabel("Time Index (Satellite Pass)")
plt.ylabel("Prediction Error (dB)")
plt.title("Temporal Error Drift")
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "temporal_error_drift.png"), dpi=300)
plt.close()

print("Evaluation plots generated successfully.")
print("Mean Absolute Error:", np.mean(np.abs(error)))
print("Max Absolute Error:", np.max(np.abs(error)))

import matplotlib.pyplot as plt
import numpy as np

# Residuals
residuals = y_test - y_pred

plt.figure(figsize=(7,5))
plt.hist(residuals, bins=50)
plt.xlabel("Prediction Error (dB)")
plt.ylabel("Frequency")
plt.title("Residual Error Distribution")
plt.grid(True)
plt.tight_layout()

plt.savefig("results/residual_histogram.png", dpi=300)
plt.close()
