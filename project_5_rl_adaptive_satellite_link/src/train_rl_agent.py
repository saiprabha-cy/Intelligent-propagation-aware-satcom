import os
import numpy as np
import pandas as pd
from rl_agent import QLearningAgent

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "results", "adaptive_link_results.csv")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "q_table.pkl")

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)

# -----------------------------
# State bins
# -----------------------------
state_bins = [
    np.linspace(0, 90, 10),     # elevation
    np.linspace(2, 12, 5),      # frequency
    np.linspace(0, 50, 6),      # rain
    np.linspace(0, 10, 6),      # excess loss
    np.linspace(-10, 40, 10)    # link margin
]

ACTIONS = ["QPSK", "16-QAM", "64-QAM", "OFF"]

# -----------------------------
# Initialize agent
# -----------------------------
agent = QLearningAgent(
    state_bins=state_bins,
    action_size=len(ACTIONS)
)

# -----------------------------
# Reward function
# -----------------------------
def compute_reward(mcs, link_margin):
    if link_margin < 0:
        return -10

    if mcs == "64-QAM":
        return 5 if link_margin > 15 else -5
    if mcs == "16-QAM":
        return 3 if link_margin > 8 else -5
    if mcs == "QPSK":
        return 1 if link_margin > 2 else -5

    return 0  # OFF

# -----------------------------
# Training loop
# -----------------------------
EPISODES = 30

for ep in range(EPISODES):
    total_reward = 0

    for i in range(len(df) - 1):
        state = [
            df.loc[i, "elevation_deg"],
            df.loc[i, "frequency_ghz"],
            df.loc[i, "rain_mm_hr"],
            df.loc[i, "predicted_excess_loss_db"],
            df.loc[i, "link_margin_db"]
        ]

        next_state = [
            df.loc[i+1, "elevation_deg"],
            df.loc[i+1, "frequency_ghz"],
            df.loc[i+1, "rain_mm_hr"],
            df.loc[i+1, "predicted_excess_loss_db"],
            df.loc[i+1, "link_margin_db"]
        ]

        action_idx = agent.select_action(state)
        mcs = ACTIONS[action_idx]

        reward = compute_reward(mcs, state[-1])
        total_reward += reward

        agent.update(state, action_idx, reward, next_state, done=False)

    print(f"Episode {ep+1}/{EPISODES} | Total Reward: {total_reward:.2f}")

# -----------------------------
# Save trained policy
# -----------------------------
agent.save(MODEL_SAVE_PATH)

print("\nRL training completed.")
print(f"Q-table saved to: {MODEL_SAVE_PATH}")
