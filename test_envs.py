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

env_names = [
    # "AntMaze_UMaze-v2",
    "AdroitHandPen-v1",
    "AdroitHandHammer-v1",
    "AdroitHandRelocate-v1",
    "AdroitHandDoor-v1",
    "Pusher-v5",
    "InvertedDoublePendulum-v5",
    "Hopper-v5",
]
training_steps = 1000000
episodes = 200
for e_i, env_name in enumerate(env_names):
    # if e_i != 2:
    # continue
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
            if np.random.random() < 1.0:
                action = guide_policy.predict(obs)[0]
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            step_rewards[steps][episode] = total_reward
            # distance = adroit_relocate(obs, env)
            # success = info.get("is_success", False)
            # success = False
            # nail_pos = env.unwrapped.data.site_xpos[
            #     env.unwrapped.target_obj_site_id
            # ].ravel()
            # goal_pos = env.unwrapped.data.site_xpos[env.unwrapped.goal_site_id].ravel()
            # if np.linalg.norm(goal_pos - nail_pos) < 0.01 and not success_flag:
            #     success = True
            #     print(f"Step: {steps}, success={success}, reward={total_reward}")
            #     success_flag = True
            done = term or trunc
            # env.render()
            steps += 1
        print(f"Episode {episode + 1} finished with total reward: {total_reward}")
    env.close()

    step_means = []
    for k, v in step_rewards.items():
        step_means.append(np.array(v))
    rewards_matrix = np.array(step_means)
    print(rewards_matrix.shape)
    import matplotlib.pyplot as plt

    reward_diffs = (
        np.mean(rewards_matrix, axis=1)[1:] - np.mean(rewards_matrix, axis=1)[:-1]
    )
    reward_diffs = reward_diffs[1:] - reward_diffs[:-1]
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.plot(range(len(reward_diffs)), reward_diffs, linestyle="--")
    plt.scatter(range(len(reward_diffs)), reward_diffs, marker="X", color="black")
    plt.xlabel("Step")
    plt.ylabel("Mean reward difference")
    plt.title(env_name)
    plt.tight_layout()
    plt.savefig(f"reward_diff_{env_name}.png")
    # plt.show()
