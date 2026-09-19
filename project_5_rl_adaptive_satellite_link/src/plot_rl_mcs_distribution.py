import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/rl_policy_results.csv")

counts = df["rl_mcs"].value_counts()

plt.figure()
counts.plot(kind="bar")
plt.xlabel("MCS")
plt.ylabel("Count")
plt.title("RL-Based MCS Distribution")
plt.grid(axis="y")

plt.savefig("results/rl_mcs_distribution.png", dpi=300)
plt.show()

