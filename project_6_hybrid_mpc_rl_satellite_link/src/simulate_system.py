import pandas as pd
import joblib
import os
from hybrid_controller import hybrid_mcs_selection

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "adaptive_link_results.csv")
QTABLE_PATH = os.path.join(PROJECT_ROOT, "models", "q_table.pkl")

df = pd.read_csv(DATA_PATH)
q_table = joblib.load(QTABLE_PATH)

df["hybrid_mcs"] = df.apply(
    lambda row: hybrid_mcs_selection(row, q_table),
    axis=1
)

OUTPUT_PATH = os.path.join(PROJECT_ROOT, "results", "hybrid_policy_results.csv")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("Hybrid MPC + RL simulation completed.")
print(df["hybrid_mcs"].value_counts())
