import pandas as pd
import numpy as np
import pickle
import os

# Load Q-table (DICT)
with open("models/q_table.pkl", "rb") as f:
    Q = pickle.load(f)

ACTIONS = ["QPSK", "16-QAM", "64-QAM", "LINK-OFF"]

def margin_to_state(margin):
    """
    Must match training discretization
    """
    return int(np.clip(np.floor(margin), -10, 20))

results = []
ELEVATIONS = np.linspace(5, 90, 4500)

for elev in ELEVATIONS:
    # Same proxy channel logic used earlier
    if elev < 20:
        margin = -2 + 0.1 * elev
    elif elev < 45:
        margin = 3 + 0.15 * elev
    else:
        margin = 8 + 0.2 * elev

    state = margin_to_state(margin)

    # SAFE fallback if state unseen during training
    if state not in Q:
        mcs = "QPSK"
    else:
        action_idx = int(np.argmax(Q[state]))
        mcs = ACTIONS[action_idx]

    results.append({
        "elevation_deg": elev,
        "link_margin_db": margin,
        "rl_mcs": mcs
    })

df = pd.DataFrame(results)

os.makedirs("results", exist_ok=True)
df.to_csv("results/rl_policy_results.csv", index=False)

print("RL evaluation results saved to results/rl_policy_results.csv")
