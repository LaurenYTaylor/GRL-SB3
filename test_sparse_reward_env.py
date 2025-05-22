import numpy as np
import matplotlib.pyplot as plt

ep_length = 10
episodes = 500
step_rewards = dict(
    zip(
        range(ep_length),
        [np.zeros(episodes) for _ in range(ep_length)],
    )
)
for ep in range(episodes):
    t = 0
    total_reward = 0
    done = False

    while not done:
        if np.random.random() < 0.3:
            total_reward = 1
            step_rewards[t][ep] = total_reward
            done = True
        else:
            total_reward = 1
            step_rewards[t][ep] = total_reward

        t += 1
        if t >= ep_length:
            done = True
        # if done and t < ep_length:
        #    for next_t in range(t, ep_length):
        #        step_rewards[next_t][ep] = total_reward

step_means = []
for k, v in step_rewards.items():
    step_means.append(np.array(v))
rewards_matrix = np.array(step_means)
per_episode = rewards_matrix.T

gamma = 0.99
returns = np.zeros_like(per_episode)
for i in range(per_episode.shape[1], 0, -1):
    if i == per_episode.shape[1]:
        returns[:, i - 1] = per_episode[:, i - 1]
    else:
        returns[:, i - 1] = per_episode[:, i - 1] + gamma * returns[:, i]

mean_returns = np.mean(returns, axis=0)
std_returns = np.std(returns, axis=0)
fig, ax = plt.subplots(figsize=(8, 8))
plt.scatter(range(len(mean_returns)), mean_returns, label="Mean")
plt.scatter(range(len(std_returns)), std_returns, label="Std")
plt.legend()
plt.show()

perc = np.percentile(returns, range(10, 110, 10))

perc_dict = {}
for i, p in enumerate(perc):
    idxs = np.where(returns <= p)[0]
    perc_dict[p] = idxs

print(perc_dict)
