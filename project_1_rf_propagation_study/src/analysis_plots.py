import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------------
# Load dataset
# -------------------------
data_path = os.path.join("data", "propagation_dataset.csv")
df = pd.read_csv(data_path)

# -------------------------
# Create output directory
# -------------------------
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

# -------------------------
# 1. Path Loss vs Distance
# -------------------------
plt.figure()
plt.scatter(df["distance_km"], df["total_path_loss_db"], s=3)
plt.xlabel("Distance (km)")
plt.ylabel("Total Path Loss (dB)")
plt.title("Total Path Loss vs Distance (S-band)")
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "path_loss_vs_distance.png"), dpi=300)
plt.show()

# -------------------------
# 2. FSPL vs Distance
# -------------------------
plt.figure()
plt.scatter(df["distance_km"], df["fspl_db"], s=3)
plt.xlabel("Distance (km)")
plt.ylabel("FSPL (dB)")
plt.title("Free Space Path Loss vs Distance")
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "fspl_vs_distance.png"), dpi=300)
plt.show()

# -------------------------
# 3. COST-231 vs FSPL comparison
# -------------------------
plt.figure()
plt.scatter(df["distance_km"], df["cost231_db"], s=3, label="COST-231")
plt.scatter(df["distance_km"], df["fspl_db"], s=3, label="FSPL")
plt.xlabel("Distance (km)")
plt.ylabel("Path Loss (dB)")
plt.title("COST-231 vs FSPL (S-band)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "cost231_vs_fspl.png"), dpi=300)
plt.show()

# -------------------------
# 4. Rain Attenuation vs Rain Rate
# -------------------------
plt.figure()
plt.scatter(df["rain_mm_hr"], df["rain_atten_db"], s=3)
plt.xlabel("Rain Rate (mm/hr)")
plt.ylabel("Rain Attenuation (dB)")
plt.title("Rain Attenuation vs Rain Rate")
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "rain_atten_vs_rain.png"), dpi=300)
plt.show()

print("All validation plots generated and saved in 'plots/' directory.")
