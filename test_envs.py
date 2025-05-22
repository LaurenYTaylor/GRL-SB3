import gymnasium
import gymnasium_robotics
from stable_baselines3 import SAC, PPO
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import os
from goal_distance_fns import adroit_relocate
import torch
import matplotlib.pyplot as plt


def exponential_smoothing(data, alpha):
    smoothed_data = np.zeros_like(data, dtype=float)
    smoothed_data[0] = data[0]
    for i in range(1, len(data)):
        smoothed_data[i] = alpha * data[i] + (1 - alpha) * smoothed_data[i - 1]
    return smoothed_data


env_names = [
    # "AntMaze_UMaze-v2",
    # "AdroitHandPen-v1",
    "AdroitHandHammer-v1",
    "AdroitHandRelocate-v1",
    "AdroitHandDoor-v1",
    "Pusher-v5",
    "InvertedDoublePendulum-v5",
    "Hopper-v5",
]
training_steps = 1000000
episodes = 50
for e_i, env_name in enumerate(env_names):
    env = gymnasium.make(env_name, disable_env_checker=True)

    pretrained_path = f"{os.getcwd()}/pretrained_1million/{env_name}_sac.zip"
    try:
        guide_policy = SAC.load(pretrained_path, device="cpu")
    except FileNotFoundError:
        from pathlib import Path

        policy_file = Path(pretrained_path[:-8] + "/checkpoint_1999999.pt")
        policy = torch.load(policy_file)
    step_rewards = dict(
        zip(
            range(env.spec.max_episode_steps),
            [np.zeros(episodes) for _ in range(env.spec.max_episode_steps)],
        )
    )
    for episode in range(episodes):
        done = False
        obs, infos = env.reset()
        steps = 0
        total_reward = 0
        success_flag = False
        while not done:
            action = env.action_space.sample()
            if np.random.random() < 0.95:
                action = guide_policy.predict(obs)[0]
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            step_rewards[steps][episode] = reward
            done = term or trunc
            # env.render()
            steps += 1
        # print(f"Episode {episode + 1} finished with total reward: {total_reward}")
    env.close()

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

    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    # reward_diffs = np.std(rewards_matrix, axis=1)
    # reward_diffs = np.abs(reward_diffs[1:]-reward_diffs[:-1])
    # norm_diffs = 1-reward_diffs/max(reward_diffs)
    # ax[0].scatter(range(len(reward_diffs)), reward_diffs, linestyle="--")

    std_returns = np.std(returns, axis=0)
    mean_returns = np.mean(returns, axis=0)
    ax[1].scatter(range(len(returns[0])), mean_returns, label="Mean Returns")
    ax[1].scatter(range(len(returns[0])), std_returns, label="Std Returns")
    ax[1].scatter(range(len(returns[0])), returns[0], label="Returns")

    return_diff = mean_returns[1:] - mean_returns[:-1]
    ax[0].scatter(range(len(return_diff)), return_diff, label="Return Diff")
    plt.legend()

    # plt.plot(range(len(reward_diffs)), reward_diffs, linestyle="--")
    # plt.scatter(range(len(reward_diffs)), reward_diffs, marker="X", color="black")
    plt.xlabel("Step")
    ax[0].set_ylabel("Mean Return Difference")
    ax[1].set_ylabel("Mean Returns")
    plt.title(env_name)
    plt.tight_layout()
    # plt.savefig(f"stddev_diff_{env_name}.png")
    plt.show()

    perc = np.percentile(mean_returns, range(10, 110, 10))

    perc_dict = {}
    for i, p in enumerate(perc):
        if i == 0:
            idxs = np.where(mean_returns <= p)[0]
        else:
            idxs = np.where((mean_returns <= p) & (mean_returns > perc[i - 1]))[0]
        perc_dict[p] = idxs
    print(perc_dict)
    print(perc_dict.keys())
