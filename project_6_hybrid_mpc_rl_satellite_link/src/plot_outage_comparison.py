import pandas as pd
import matplotlib.pyplot as plt

rule = pd.read_csv(
    "../adaptive_satellite_link_system/results/adaptive_link_results.csv"
)
rl = pd.read_csv(
    "../project_5_rl_adaptive_satellite_link/results/rl_policy_results.csv"
)
hybrid = pd.read_csv("results/hybrid_results.csv")

def outage_prob(df, mcs_col):
    return (df[mcs_col] == "LINK-OFF").mean()

outages = {
    "Rule-Based": outage_prob(rule, "adaptive_mcs"),
    "RL-Based": outage_prob(rl, "rl_mcs"),
    "Hybrid MPC+RL": outage_prob(hybrid, "hybrid_mcs")
}

plt.figure()
plt.bar(outages.keys(), outages.values())
plt.ylabel("Outage Probability")
plt.title("Outage Probability Comparison")
plt.grid(axis="y")

plt.savefig("results/outage_comparison.png", dpi=300)
plt.show()

