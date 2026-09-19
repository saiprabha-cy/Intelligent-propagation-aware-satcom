import numpy as np
import pandas as pd

# ============================================================
# LINK BUDGET PARAMETERS (ASSUMPTIONS — MUST BE STATED IN PAPER)
# ============================================================

TX_POWER_DBW = 20.0        # Satellite transmit power (dBW)
TX_GAIN_DBI = 12.0         # Satellite antenna gain (dBi)
RX_GAIN_DBI = 25.0         # Ground station antenna gain (dBi)
SYSTEM_LOSSES_DB = 2.0     # Cable + pointing + polarization losses
NOISE_FLOOR_DB = -170.0    # Thermal noise floor (dBW)

# ============================================================
# FSPL (Friis equation)
# ============================================================

def compute_fspl(distance_km, frequency_ghz):
    """
    Free Space Path Loss (dB)
    """
    distance_m = distance_km * 1e3
    frequency_hz = frequency_ghz * 1e9
    return 20 * np.log10(distance_m) + 20 * np.log10(frequency_hz) - 147.55

# ============================================================
# LINK BUDGET COMPUTATION
# ============================================================

def compute_link_margin(
    distance_km,
    frequency_ghz,
    excess_loss_db
):
    """
    Computes link margin (dB)
    """
    fspl_db = compute_fspl(distance_km, frequency_ghz)

    received_power_dbw = (
        TX_POWER_DBW
        + TX_GAIN_DBI
        + RX_GAIN_DBI
        - fspl_db
        - excess_loss_db
        - SYSTEM_LOSSES_DB
    )

    link_margin_db = received_power_dbw - NOISE_FLOOR_DB
    return link_margin_db

# ============================================================
# BATCH PROCESSING (for dataset integration)
# ============================================================

def compute_link_margin_dataframe(df):
    """
    Adds link_margin_db column to dataframe
    """
    df = df.copy()
    df["link_margin_db"] = compute_link_margin(
        df["slant_range_km"].values,
        df["frequency_ghz"].values,
        df["excess_loss_db"].values
    )
    return df


# ============================================================
# TEST (run standalone)
# ============================================================

if __name__ == "__main__":
    # Simple sanity test
    test_margin = compute_link_margin(
        distance_km=800,
        frequency_ghz=2.2,
        excess_loss_db=1.5
    )
    print(f"Sample link margin: {test_margin:.2f} dB")
