import matplotlib.pyplot as plt

episodes = list(range(1, 31))
rewards = [
    9332,9608,9517,9605,9497,9591,9621,9596,9591,9658,
    9562,9593,9614,9624,9609,9575,9642,9651,9527,9599,
    9687,9668,9608,9604,9653,9610,9676,9588,9626,9632
]

plt.figure()
plt.plot(episodes, rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("RL Training Reward Convergence")
plt.grid(True)

plt.savefig("results/rl_training_reward.png", dpi=300)
plt.show()
