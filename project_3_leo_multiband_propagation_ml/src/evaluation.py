import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "multiband_ml_dataset.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "rf_multiband_excess_loss.pkl")
PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# -----------------------------
# Load data and model
# -----------------------------
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

# -----------------------------
# Temporal split (same as training)
# -----------------------------
df = df.sort_values("time_index").reset_index(drop=True)
split_idx = int(len(df) * 0.7)

test_df = df.iloc[split_idx:]

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

X_test = test_df[features]
y_test = test_df["excess_loss_db"]
y_pred = model.predict(X_test)

errors = y_pred - y_test

# -----------------------------
# 1. True vs Predicted
# -----------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, s=5, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel("True Excess Loss (dB)")
plt.ylabel("Predicted Excess Loss (dB)")
plt.title("True vs Predicted Excess Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "fig2_true_vs_predicted.png"), dpi=300)
plt.close()

# -----------------------------
# 2. Error vs Elevation
# -----------------------------
plt.figure(figsize=(7,5))
plt.scatter(test_df["elevation_deg"], errors, s=5, alpha=0.6)
plt.xlabel("Elevation Angle (deg)")
plt.ylabel("Prediction Error (dB)")
plt.title("Prediction Error vs Elevation Angle")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "fig3_error_vs_elevation.png"), dpi=300)
plt.close()

# -----------------------------
# 3. Error vs Frequency
# -----------------------------
plt.figure(figsize=(7,5))
plt.scatter(test_df["frequency_ghz"], errors, s=5, alpha=0.6)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Prediction Error (dB)")
plt.title("Prediction Error vs Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "fig4_error_vs_frequency.png"), dpi=300)
plt.close()

# -----------------------------
# 4. Feature Importance
# -----------------------------
importances = model.feature_importances_

importance_df = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(7,5))
plt.barh(importance_df["feature"], importance_df["importance"])
plt.xlabel("Importance Score")
plt.title("Feature Importance for Multi-band Excess Loss Prediction")
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "fig5_feature_importance.png"), dpi=300)
plt.close()

# -----------------------------
# Print summary stats
# -----------------------------
mae = np.mean(np.abs(errors))
max_err = np.max(np.abs(errors))

print("Evaluation completed.")
print(f"Mean Absolute Error: {mae:.4f} dB")
print(f"Max Absolute Error : {max_err:.4f} dB")
print(f"Plots saved to: {PLOT_DIR}")
