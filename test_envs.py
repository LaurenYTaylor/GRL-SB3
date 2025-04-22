import gymnasium
import gymnasium_robotics
from stable_baselines3 import SAC, PPO
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import os

env_names = [
    "AdroitHandPen-v1",
    "AdroitHandHammer-v1",
    "AdroitHandRelocate-v1",
    "AdroitHandDoor-v1",
    "Pusher-v5",
    "InvertedDoublePendulum-v5",
    "Hopper-v5",
]
training_steps = 1000000
episodes = 100
for e_i, env_name in enumerate(env_names):
    if e_i < 6:
        continue
    env = gymnasium.make(env_name, render_mode="human")
    import pdb

    pdb.set_trace()
    pretrained_path = f"{os.getcwd()}/pretrained_1million/{env_name}_ppo.zip"
    guide_policy = PPO.load(pretrained_path, device="cpu")
    for episode in range(episodes):
        done = False
        obs, infos = env.reset()
        steps = 0
        while not done:
            action = env.action_space.sample()
            if np.random.random() < 0.2:
                action = guide_policy.predict(obs)[0]
            obs, reward, term, trunc, info = env.step(action)
            xpos = env.unwrapped.data.qpos[0]
            if xpos < -1:
                distance = np.abs(xpos) * 1 / obs[0]
            elif xpos < 1:
                distance = 1 * 1 / obs[0]
            else:
                distance = 1 / env.unwrapped.data.qpos[0]
            print(distance)
            done = term or trunc
            env.render()
            steps += 1
    env.close()
