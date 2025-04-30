import os
import copy

env_names = [
    "AdroitHandPen-v1",
    "AdroitHandHammer-v1",
    "AdroitHandRelocate-v1",
    "AdroitHandDoor-v1",
    "Pusher-v5",
    "InvertedDoublePendulum-v5",
    "Hopper-v5",
]
algorithms = ["SAC", "SAC", "PPO", "SAC", "SAC", "SAC", "PPO"]
algorithm_dict = dict(zip(env_names, algorithms))
paths_dict = dict(
    zip(
        env_names,
        [
            f"{os.getcwd()}/pretrained_1million/{env_names[i]}_{algorithms[i].lower()}.zip"
            for i in range(len(env_names))
        ],
    )
)

DEFAULT_CONFIG = {
    "eval_freq": 10000,
    "n_eval_episodes": 150,
    "pretrain_eval_episodes": 500,
    "training_steps": 1000000,
    "grl_config": {
        "horizon_fn": "variance",
        "n_curriculum_stages": 20,
        "variance_fn": None,
        "rolling_mean_n": 1,
        "tolerance": 0.05,
        "guide_randomness": 0.05,
    },
    "algo_config": {
        "buffer_size": 10000,
        "batch_size": 256,
        "learning_starts": 1000,
        "train_freq": 64,
        "gradient_steps": 8,
    },
}


def get_config(env_name, horizon_fn):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["algo"] = algorithm_dict[env_name]
    config["env_name"] = env_name
    config["pretrained_path"] = paths_dict[env_name]
    config["grl_config"]["horizon_fn"] = horizon_fn
    return config
