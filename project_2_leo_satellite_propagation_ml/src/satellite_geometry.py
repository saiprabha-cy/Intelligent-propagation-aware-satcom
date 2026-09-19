import numpy as np
import pandas as pd
import os

# -----------------------------
# Simulation parameters
# -----------------------------
NUM_SAMPLES = 5000

EARTH_RADIUS_KM = 6371
SAT_ALTITUDE_KM = 600          # Typical LEO
C = 3e8                        # Speed of light (m/s)
SAT_VELOCITY_KM_S = 7.5        # Approx LEO orbital speed

np.random.seed(42)

# -----------------------------
# Generate satellite geometry
# -----------------------------
elevation_deg = np.random.uniform(5, 90, NUM_SAMPLES)     # degrees
elevation_rad = np.deg2rad(elevation_deg)

# Slant range approximation
slant_range_km = np.sqrt(
    (EARTH_RADIUS_KM + SAT_ALTITUDE_KM)**2 -
    (EARTH_RADIUS_KM * np.cos(elevation_rad))**2
) - EARTH_RADIUS_KM * np.sin(elevation_rad)

# Doppler proxy (normalized)
doppler_proxy = SAT_VELOCITY_KM_S * np.cos(elevation_rad)

# Time index of satellite pass (0–1 normalized)
time_index = np.random.uniform(0, 1, NUM_SAMPLES)

# Simplified weather parameters
rain_mm_hr = np.random.uniform(0, 50, NUM_SAMPLES)
humidity_pct = np.random.uniform(30, 100, NUM_SAMPLES)
temperature_c = np.random.uniform(-10, 40, NUM_SAMPLES)

# -----------------------------
# Create DataFrame (THIS is df)
# -----------------------------
df = pd.DataFrame({
    "elevation_deg": elevation_deg,
    "slant_range_km": slant_range_km,
    "doppler_proxy": doppler_proxy,
    "time_index": time_index,
    "rain_mm_hr": rain_mm_hr,
    "humidity_pct": humidity_pct,
    "temperature_c": temperature_c
})

# -----------------------------
# Safe path handling
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(PROJECT_ROOT, "data")
os.makedirs(data_dir, exist_ok=True)

output_path = os.path.join(data_dir, "satellite_geometry_dataset.csv")

# -----------------------------
# Save
# -----------------------------
df.to_csv(output_path, index=False)

print(f"Dataset saved to: {output_path}")
print("Shape:", df.shape)
print(df.head())
