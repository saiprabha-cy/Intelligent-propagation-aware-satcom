import pandas as pd
import numpy as np
import os

ELEVATIONS = np.linspace(5, 90, 4500)
results = []

for elev in ELEVATIONS:
    predicted_margin = 5 + 0.18 * elev
    safe_margin = max(predicted_margin, 3.0)

    if safe_margin < 2:
        mcs = "QPSK"
    elif safe_margin < 6:
        mcs = "16-QAM"
    else:
        mcs = "64-QAM"

    results.append({
        "elevation_deg": elev,
        "link_margin_db": safe_margin,
        "hybrid_mcs": mcs
    })

df = pd.DataFrame(results)
os.makedirs("results", exist_ok=True)
df.to_csv("results/hybrid_results.csv", index=False)

print("Hybrid evaluation results saved.")
