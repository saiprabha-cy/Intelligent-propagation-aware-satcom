import numpy as np

def predict_future_margin(current_margin, doppler_proxy, steps=5):
    """
    Simple MPC-style prediction of link margin
    """
    predicted = []
    margin = current_margin

    for _ in range(steps):
        # Margin decay due to geometry + Doppler dynamics
        margin = margin - 0.2 * abs(doppler_proxy)
        predicted.append(margin)

    return np.array(predicted)
