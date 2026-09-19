import os
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "test_passes.csv")
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "..",
    "leo_multiband_propagation_ml",
    "models",
    "rf_multiband_excess_loss.pkl"
)

# -----------------------------
# Constants (S / C / X band safe)
# -----------------------------
TX_POWER_DBM = 30.0          # dBm
TX_GAIN_DB = 35.0            # dBi
RX_GAIN_DB = 35.0            # dBi
SYSTEM_LOSS_DB = 2.0         # dB
NOISE_FLOOR_DBM = -100.0     # dBm
REQUIRED_SNR_DB = 10.0       # dB

# -----------------------------
# FSPL computation
# -----------------------------
def compute_fspl(distance_km, frequency_ghz):
    return 92.45 + 20 * np.log10(distance_km) + 20 * np.log10(frequency_ghz)

# -----------------------------
# Link margin computation
# -----------------------------
def compute_link_margin(distance_km, frequency_ghz, excess_loss_db):
    fspl = compute_fspl(distance_km, frequency_ghz)

    received_power = (
        TX_POWER_DBM
        + TX_GAIN_DB
        + RX_GAIN_DB
        - fspl
        - excess_loss_db
        - SYSTEM_LOSS_DB
    )

    snr = received_power - NOISE_FLOOR_DBM
    margin = snr - REQUIRED_SNR_DB

    return margin

# -----------------------------
# Load data and model
# -----------------------------
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

# -----------------------------
# Feature set (MUST match training)
# -----------------------------
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

X = df[features]

# -----------------------------
# Predict excess loss
# -----------------------------
df["predicted_excess_loss_db"] = model.predict(X)

# -----------------------------
# Compute link margin
# -----------------------------
df["link_margin_db"] = df.apply(
    lambda row: compute_link_margin(
        row["slant_range_km"],
        row["frequency_ghz"],
        row["predicted_excess_loss_db"]
    ),
    axis=1
)

# -----------------------------
# Link state classification
# -----------------------------
def classify_link(margin):
    if margin > 10:
        return "GOOD"
    elif margin > 0:
        return "DEGRADED"
    else:
        return "OUTAGE"

df["link_state"] = df["link_margin_db"].apply(classify_link)

# =====================================================
# 🔥 ADAPTIVE MCS LOGIC (ADD HERE)
# =====================================================
def select_mcs(link_margin_db):
    if link_margin_db > 15:
        return "64-QAM"
    elif link_margin_db > 8:
        return "16-QAM"
    elif link_margin_db > 2:
        return "QPSK"
    else:
        return "LINK-OFF"

df["adaptive_mcs"] = df["link_margin_db"].apply(select_mcs)

def throughput_class(mcs):
    if mcs == "64-QAM":
        return "HIGH"
    elif mcs == "16-QAM":
        return "MEDIUM"
    elif mcs == "QPSK":
        return "LOW"
    else:
        return "NONE"

df["throughput_class"] = df["adaptive_mcs"].apply(throughput_class)

# -----------------------------
# Output summary
# -----------------------------
print(f"Sample link margin: {df['link_margin_db'].iloc[0]:.2f} dB\n")

print("Link state distribution:")
print(df["link_state"].value_counts())

print("\nAdaptive MCS distribution:")
print(df["adaptive_mcs"].value_counts())

# -----------------------------
# Save results
# -----------------------------
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "results", "adaptive_link_results.csv")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("\nAdaptive link simulation completed.")
print(f"Results saved to: {OUTPUT_PATH}")
