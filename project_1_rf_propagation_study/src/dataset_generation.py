import numpy as np
import pandas as pd
import os

from src.propagation_models import fspl, cost231, rain_attenuation

# -------------------------
# Configuration
# -------------------------
N = 8000
frequency_mhz = 2200  # S-band

# -------------------------
# Dataset container
# -------------------------
data = []

# -------------------------
# Data generation loop
# -------------------------
for _ in range(N):
    distance_km = np.random.uniform(0.5, 20)
    rain_mm_hr = np.random.uniform(0, 50)
    humidity_pct = np.random.uniform(40, 95)
    temperature_c = np.random.uniform(20, 40)

    fspl_db = fspl(distance_km, frequency_mhz)
    cost_db = cost231(distance_km, frequency_mhz)
    rain_db = rain_attenuation(distance_km, rain_mm_hr)

    total_loss_db = fspl_db + cost_db + rain_db

    data.append([
        distance_km,
        rain_mm_hr,
        humidity_pct,
        temperature_c,
        fspl_db,
        cost_db,
        rain_db,
        total_loss_db
    ])

# -------------------------
# Create DataFrame
# -------------------------
columns = [
    "distance_km",
    "rain_mm_hr",
    "humidity_pct",
    "temperature_c",
    "fspl_db",
    "cost231_db",
    "rain_atten_db",
    "total_path_loss_db"
]

df = pd.DataFrame(data, columns=columns)

# -------------------------
# Save dataset
# -------------------------
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "propagation_dataset.csv")
df.to_csv(output_path, index=False)

print("Dataset saved to:", output_path)
print("Shape:", df.shape)
print(df.head())
