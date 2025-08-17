import gymnasium
from stable_baselines3 import TD3, SAC
import os
import numpy as np
import CombinationLockV1

training_steps = 10000
episodes = 1000
env_name = "CombinationLock-v1"
algo = SAC
env = gymnasium.make(
    env_name, disable_env_checker=True, reward="dense", max_steps=10, verbose=True
)

pretrained_path = f"{os.getcwd()}/pretrained_{training_steps}/{env_name}_td3.zip"
try:
    policy = algo.load(pretrained_path, device="cpu")
except (FileNotFoundError, IsADirectoryError):
    policy = algo("MlpPolicy", env, verbose=1, device="cpu", learning_rate=0.0001)
    policy.learn(total_timesteps=training_steps)
    policy.save(pretrained_path)
    policy = algo.load(pretrained_path, device="cpu")

total_rewards = []
for episode in range(episodes):
    done = False
    obs, infos = env.reset(seed=0)
    steps = 0
    total_reward = 0
    success_flag = False
    while not done:
        # action = env.action_space.sample()
        if np.random.random() < 1.0:
            action = policy.predict(obs, deterministic=False)[0]
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        done = term or trunc
        # env.render()
        steps += 1
    print(f"Episode {episode + 1} finished with total reward: {total_reward}")
    total_rewards.append(total_reward)
env.close()
print(f"Average reward over {episodes} episodes: {np.mean(total_rewards)}")
