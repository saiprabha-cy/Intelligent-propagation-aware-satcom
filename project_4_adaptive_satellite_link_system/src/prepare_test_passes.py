import os
import pandas as pd

# ------------------------------------
# Paths
# ------------------------------------
SOURCE_PROJECT = r"D:\Project\leo_multiband_propagation_ml\data\multiband_ml_dataset.csv"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "test_passes.csv")

# ------------------------------------
# Load dataset
# ------------------------------------
df = pd.read_csv(SOURCE_PROJECT)

# ------------------------------------
# Select required columns ONLY
# (FSPL will be computed later)
# ------------------------------------
required_columns = [
    "elevation_deg",
    "slant_range_km",
    "doppler_proxy",
    "time_index",
    "frequency_ghz",
    "rain_mm_hr",
    "humidity_pct",
    "temperature_c"
]

df = df[required_columns]

# ------------------------------------
# Temporal split (unseen passes)
# ------------------------------------
split_idx = int(0.7 * len(df))
test_df = df.iloc[split_idx:].reset_index(drop=True)

# ------------------------------------
# Save
# ------------------------------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
test_df.to_csv(OUTPUT_PATH, index=False)

print("test_passes.csv created successfully.")
print("Shape:", test_df.shape)
print("Columns:", list(test_df.columns))
print("Saved to:", OUTPUT_PATH)
