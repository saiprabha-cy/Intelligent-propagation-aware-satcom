import pandas as pd
import matplotlib.pyplot as plt

rule = pd.read_csv(
    "../adaptive_satellite_link_system/results/adaptive_link_results.csv"
)
rl = pd.read_csv(
    "../project_5_rl_adaptive_satellite_link/results/rl_policy_results.csv"
)
hybrid = pd.read_csv("results/hybrid_results.csv")

rule_counts = rule["adaptive_mcs"].value_counts()
rl_counts = rl["rl_mcs"].value_counts()
hybrid_counts = hybrid["hybrid_mcs"].value_counts()

df = pd.DataFrame({
    "Rule-Based": rule_counts,
    "RL-Based": rl_counts,
    "Hybrid MPC+RL": hybrid_counts
}).fillna(0)

df.plot(kind="bar")
plt.xlabel("MCS")
plt.ylabel("Count")
plt.title("MCS Selection Comparison")
plt.grid(axis="y")

plt.savefig("results/mcs_comparison.png", dpi=300)
plt.show()
