import numpy as np
import pandas as pd
import os

# -----------------------------
# Constants
# -----------------------------
FREQ_GHZ = 2.2       # S-band frequency
C = 3e8              # Speed of light (m/s)

# -----------------------------
# Load geometry dataset
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(PROJECT_ROOT, "data", "satellite_geometry_dataset.csv")

df = pd.read_csv(data_path)

# -----------------------------
# Free Space Path Loss (FSPL)
# -----------------------------
slant_range_m = df["slant_range_km"] * 1e3
frequency_hz = FREQ_GHZ * 1e9

fspl_db = (
    20 * np.log10(slant_range_m) +
    20 * np.log10(frequency_hz) -
    147.55
)

# -----------------------------
# Simplified atmospheric loss
# (Elevation + rain dependent)
# -----------------------------
elevation_rad = np.deg2rad(df["elevation_deg"])

# Elevation loss (low elevation → higher loss)
elevation_loss_db = 2.0 / np.sin(elevation_rad)

# Rain attenuation proxy
rain_loss_db = 0.02 * df["rain_mm_hr"]

atmospheric_loss_db = elevation_loss_db + rain_loss_db

# -----------------------------
# Total & excess loss
# -----------------------------
total_path_loss_db = fspl_db + atmospheric_loss_db
excess_loss_db = total_path_loss_db - fspl_db

# -----------------------------
# Append to dataframe
# -----------------------------
df["fspl_db"] = fspl_db
df["atmospheric_loss_db"] = atmospheric_loss_db
df["total_path_loss_db"] = total_path_loss_db
df["excess_loss_db"] = excess_loss_db

# -----------------------------
# Save updated dataset
# -----------------------------
output_path = os.path.join(PROJECT_ROOT, "data", "propagation_dataset.csv")
df.to_csv(output_path, index=False)

print("Propagation modeling completed.")
print("Saved to:", output_path)
print(df[[
    "elevation_deg",
    "slant_range_km",
    "fspl_db",
    "atmospheric_loss_db",
    "excess_loss_db"
]].head())
