import pandas as pd

df = pd.read_csv("results/rl_policy_output.csv")

print("\nRule-based MCS distribution:")
print(df["adaptive_mcs"].value_counts())

print("\nRL-based MCS distribution:")
print(df["rl_mcs"].value_counts())
