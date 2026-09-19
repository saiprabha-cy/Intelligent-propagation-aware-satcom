import numpy as np
from mpc_predictor import predict_future_margin

# RL action space (MUST match Project 5)
ACTIONS = ["64-QAM", "16-QAM", "QPSK", "LINK-OFF"]

def margin_state(margin):
    if margin > 10:
        return 0
    elif margin > 5:
        return 1
    elif margin > 0:
        return 2
    else:
        return 3

def hybrid_mcs_selection(row, q_table):
    state = margin_state(row["link_margin_db"])

    # -----------------------------
    # SAFE RL ACTION SELECTION
    # -----------------------------
    if state in q_table:
        rl_action = ACTIONS[np.argmax(q_table[state])]
    else:
        # Unseen state → conservative default
        rl_action = "QPSK"

    # -----------------------------
    # MPC SAFETY CHECK
    # -----------------------------
    future_margins = predict_future_margin(
        row["link_margin_db"],
        row["doppler_proxy"]
    )

    if np.min(future_margins) < 0:
        return "QPSK"  # MPC override (safety)

    return rl_action

