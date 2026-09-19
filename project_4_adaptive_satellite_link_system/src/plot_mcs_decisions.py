import os
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "results", "adaptive_link_results.csv")

df = pd.read_csv(DATA_PATH)

# Map MCS to numeric for plotting
mcs_map = {
    "LINK-OFF": 0,
    "QPSK": 1,
    "16-QAM": 2,
    "64-QAM": 3
}

df["mcs_numeric"] = df["adaptive_mcs"].map(mcs_map)

plt.figure(figsize=(8, 4))
plt.plot(df["time_index"], df["mcs_numeric"], ".", alpha=0.6)
plt.yticks([0, 1, 2, 3], ["OFF", "QPSK", "16-QAM", "64-QAM"])
plt.xlabel("Time Index")
plt.ylabel("Selected MCS")
plt.title("Adaptive Modulation Selection Over Time")
plt.grid(True)
plt.tight_layout()

OUTPUT_PATH = os.path.join(PROJECT_ROOT, "plots", "adaptive_mcs_vs_time.png")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=300)
plt.close()

print(f"MCS decision plot saved to: {OUTPUT_PATH}")
