import numpy as np
import pickle
import os

class QLearningAgent:
    def __init__(
        self,
        state_bins,
        action_size,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995
    ):
        self.state_bins = state_bins
        self.action_size = action_size

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table as dictionary
        self.q_table = {}

    # -----------------------------
    # State discretization
    # -----------------------------
    def discretize_state(self, state):
        discrete_state = []
        for i, value in enumerate(state):
            bins = self.state_bins[i]
            discrete_state.append(np.digitize(value, bins))
        return tuple(discrete_state)

    # -----------------------------
    # Action selection
    # -----------------------------
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)

        state = self.discretize_state(state)
        return np.argmax(self.q_table.get(state, np.zeros(self.action_size)))

    # -----------------------------
    # Q-table update
    # -----------------------------
    def update(self, state, action, reward, next_state, done):
        state = self.discretize_state(state)
        next_state = self.discretize_state(next_state) if not done else None

        self.q_table.setdefault(state, np.zeros(self.action_size))

        q_predict = self.q_table[state][action]

        if done:
            q_target = reward
        else:
            self.q_table.setdefault(next_state, np.zeros(self.action_size))
            q_target = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[state][action] += self.alpha * (q_target - q_predict)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # -----------------------------
    # Save / Load
    # -----------------------------
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)
