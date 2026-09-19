import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/hybrid_results.csv")

plt.figure()
plt.plot(df["elevation_deg"], df["link_margin_db"])
plt.xlabel("Elevation Angle (deg)")
plt.ylabel("Link Margin (dB)")
plt.title("Hybrid Controller: Link Margin vs Elevation")
plt.grid(True)

plt.savefig("results/hybrid_margin_vs_elevation.png", dpi=300)
plt.show()

