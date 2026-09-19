import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("data/propagation_dataset.csv")

# Define excess loss
df["excess_loss_db"] = df["total_path_loss_db"] - df["fspl_db"]

# -------------------------
# Load trained model
# -------------------------
model = joblib.load("models/rf_excess_loss_model.pkl")

# -------------------------
# Features
# -------------------------
X = df[[
    "distance_km",
    "rain_mm_hr",
    "humidity_pct",
    "temperature_c"
]]

y_true = df["excess_loss_db"]

# -------------------------
# Predictions
# -------------------------
y_pred = model.predict(X)
residuals = y_true - y_pred

# -------------------------
# Output directory
# -------------------------
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

# -------------------------
# 1. Prediction vs True
# -------------------------
plt.figure()
plt.scatter(y_true, y_pred, s=5)
plt.xlabel("True Excess Loss (dB)")
plt.ylabel("Predicted Excess Loss (dB)")
plt.title("Predicted vs True Excess Loss")
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "predicted_vs_true_excess_loss.png"), dpi=300)
plt.show()

# -------------------------
# 2. Residual Distribution
# -------------------------
plt.figure()
plt.hist(residuals, bins=50)
plt.xlabel("Residual Error (dB)")
plt.ylabel("Frequency")
plt.title("Residual Error Distribution")
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "residual_distribution.png"), dpi=300)
plt.show()

# -------------------------
# 3. Error vs Distance
# -------------------------
plt.figure()
plt.scatter(df["distance_km"], np.abs(residuals), s=5)
plt.xlabel("Distance (km)")
plt.ylabel("Absolute Error (dB)")
plt.title("Prediction Error vs Distance")
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "error_vs_distance.png"), dpi=300)
plt.show()

print("Evaluation plots generated successfully.")
print("Mean Absolute Error:", np.mean(np.abs(residuals)))
print("Max Absolute Error:", np.max(np.abs(residuals)))
