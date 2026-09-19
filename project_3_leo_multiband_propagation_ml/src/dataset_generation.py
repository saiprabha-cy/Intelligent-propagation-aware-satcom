import pandas as pd
import os

# -----------------------------
# Project paths
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "multiband_propagation_dataset.csv"
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR, "multiband_ml_dataset.csv"
)

# -----------------------------
# Load processed dataset
# -----------------------------
df = pd.read_csv(INPUT_PATH)

# -----------------------------
# Sort temporally
# -----------------------------
df = df.sort_values(by=["time_index", "frequency_ghz"]).reset_index(drop=True)

# -----------------------------
# Select ML-relevant columns
# -----------------------------
final_columns = [
    "elevation_deg",
    "slant_range_km",
    "doppler_proxy",
    "time_index",
    "frequency_ghz",
    "rain_mm_hr",
    "humidity_pct",
    "temperature_c",
    "excess_loss_db"
]

df_final = df[final_columns]

# -----------------------------
# Save final dataset
# -----------------------------
df_final.to_csv(OUTPUT_PATH, index=False)

print("Final multi-band ML dataset generated.")
print(f"Saved to: {OUTPUT_PATH}")
print(f"Shape: {df_final.shape}")
print(df_final.head())
