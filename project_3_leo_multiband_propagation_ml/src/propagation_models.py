import numpy as np
import pandas as pd
import os

# -----------------------------
# Project paths
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "geometry_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "multiband_propagation_dataset.csv")

# -----------------------------
# Load geometry dataset
# -----------------------------
df = pd.read_csv(INPUT_PATH)

# -----------------------------
# Constants
# -----------------------------
C = 3e8  # speed of light (m/s)

# -----------------------------
# FSPL calculation (frequency-aware)
# -----------------------------
def compute_fspl(distance_km, frequency_ghz):
    distance_m = distance_km * 1e3
    frequency_hz = frequency_ghz * 1e9
    return 20 * np.log10(distance_m) + 20 * np.log10(frequency_hz) - 147.55

df["fspl_db"] = compute_fspl(
    df["slant_range_km"].values,
    df["frequency_ghz"].values
)

# -----------------------------
# Atmospheric attenuation model
# (simplified but frequency-aware)
# -----------------------------
def atmospheric_loss(freq_ghz, rain, humidity, elevation):
    rain_factor = 0.02 * rain * (freq_ghz / 2.2)
    humidity_factor = 0.01 * humidity * (freq_ghz / 10)
    elevation_factor = 1 / np.sin(np.deg2rad(elevation))
    return (rain_factor + humidity_factor) * elevation_factor * 0.1

df["atmospheric_loss_db"] = atmospheric_loss(
    df["frequency_ghz"].values,
    df["rain_mm_hr"].values,
    df["humidity_pct"].values,
    df["elevation_deg"].values
)

# -----------------------------
# Excess loss definition
# -----------------------------
df["excess_loss_db"] = df["atmospheric_loss_db"]

# -----------------------------
# Save processed dataset
# -----------------------------
df.to_csv(OUTPUT_PATH, index=False)

print("Multi-band propagation modeling completed.")
print(f"Saved to: {OUTPUT_PATH}")
print(f"Shape: {df.shape}")
print(df[[
    "frequency_ghz",
    "slant_range_km",
    "fspl_db",
    "atmospheric_loss_db",
    "excess_loss_db"
]].head())
