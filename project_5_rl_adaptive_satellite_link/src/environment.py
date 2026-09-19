import numpy as np

class SatelliteLinkEnv:
    """
    Reinforcement Learning environment for adaptive satellite link control
    """

    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.index = 0
        self.max_index = len(df) - 1

        # Action space: MCS levels
        self.action_space = {
            0: "QPSK",
            1: "16-QAM",
            2: "64-QAM"
        }

    def reset(self):
        self.index = 0
        return self._get_state()

    def _get_state(self):
        row = self.df.iloc[self.index]
        return np.array([
            row["elevation_deg"],
            row["frequency_ghz"],
            row["rain_mm_hr"],
            row["predicted_excess_loss_db"],
            row["link_margin_db"]
        ], dtype=np.float32)

    def step(self, action):
        row = self.df.iloc[self.index]
        mcs = self.action_space[action]

        margin = row["link_margin_db"]

        # -----------------------------
        # Reward model (physics-aware)
        # -----------------------------
        if margin < 0:
            reward = -5.0   # outage penalty
        else:
            if mcs == "64-QAM":
                reward = 3.0 if margin > 15 else -2.0
            elif mcs == "16-QAM":
                reward = 2.0 if margin > 8 else -1.0
            else:  # QPSK
                reward = 1.0 if margin > 2 else -0.5

        # Move forward in time
        self.index += 1
        done = self.index >= self.max_index

        next_state = self._get_state() if not done else None

        return next_state, reward, done, {"mcs": mcs}
