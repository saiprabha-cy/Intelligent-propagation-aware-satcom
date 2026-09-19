import numpy as np
import pandas as pd
import os

# -----------------------------
# Project paths
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "geometry_dataset.csv")

# -----------------------------
# Simulation parameters
# -----------------------------
NUM_SAMPLES = 5000

# Frequency bands (GHz)
FREQUENCIES_GHZ = [2.2, 8.4, 12.5]  # S, X, Ku bands

# Earth & orbit assumptions
EARTH_RADIUS_KM = 6371
SAT_ALTITUDE_KM = 600  # LEO typical

# -----------------------------
# Generate base geometry
# -----------------------------
np.random.seed(42)

elevation_deg = np.random.uniform(5, 90, NUM_SAMPLES)
elevation_rad = np.deg2rad(elevation_deg)

# Slant range approximation
slant_range_km = np.sqrt(
    (EARTH_RADIUS_KM + SAT_ALTITUDE_KM)**2 -
    (EARTH_RADIUS_KM * np.cos(elevation_rad))**2
) - EARTH_RADIUS_KM * np.sin(elevation_rad)

# Doppler proxy (normalized orbital dynamics indicator)
doppler_proxy = np.abs(np.cos(elevation_rad)) * 7.5  # scaled, unitless

# Normalized time index (represents satellite pass evolution)
time_index = np.linspace(0, 1, NUM_SAMPLES)

# Weather parameters (simplified but realistic)
rain_mm_hr = np.random.uniform(0, 50, NUM_SAMPLES)
humidity_pct = np.random.uniform(30, 90, NUM_SAMPLES)
temperature_c = np.random.uniform(-10, 40, NUM_SAMPLES)

# -----------------------------
# Expand across frequencies
# -----------------------------
records = []

for freq in FREQUENCIES_GHZ:
    for i in range(NUM_SAMPLES):
        records.append({
            "elevation_deg": elevation_deg[i],
            "slant_range_km": slant_range_km[i],
            "doppler_proxy": doppler_proxy[i],
            "time_index": time_index[i],
            "frequency_ghz": freq,
            "rain_mm_hr": rain_mm_hr[i],
            "humidity_pct": humidity_pct[i],
            "temperature_c": temperature_c[i]
        })

df = pd.DataFrame(records)

# -----------------------------
# Save dataset
# -----------------------------
df.to_csv(OUTPUT_PATH, index=False)

print("Multi-band geometry dataset generated.")
print(f"Saved to: {OUTPUT_PATH}")
print(f"Shape: {df.shape}")
print(df.head())
