import os
import copy

env_names = [
    "AdroitHandPen-v1",
    "AdroitHandHammer-v1",
    "AdroitHandRelocate-v1",
    "AdroitHandDoor-v1",
    # "AntMaze_UMaze-v5"
    # "Pusher-v5",
    # "InvertedDoublePendulum-v5",
    "Hopper-v5",
]
algorithms = ["SAC", "SAC", "PPO", "SAC", "SAC"]  # , #"SAC", "SAC", "PPO", "SAC"]
# algorithms = ["SAC", "SAC", "PPO"]
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
    "pretrain_eval_episodes": 50,
    "training_steps": 1000000,
    "grl_config": {
        "horizon_fn": "variance",
        "n_curriculum_stages": 10,
        "variance_fn": None,
        "rolling_mean_n": 1,
        "tolerance": 0.1,
        "guide_randomness": 0.05,
        "exp_time_step_coeff": 0.001,
    },
    "algo_config": {
        "buffer_size": 10000,
        "batch_size": 256,
        "learning_starts": 0,
        "train_freq": 1,
        "gradient_steps": 1,
        "learning_rate": 0.0005,
        # "ent_coef": 5,
    },
}


def get_config(env_name, horizon_fn, guide_in, debug):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["algo"] = algorithm_dict.get(env_name, "SAC")
    config["env_name"] = env_name
    config["pretrained_path"] = paths_dict[env_name]
    config["grl_config"]["guide_in_buffer"] = guide_in
    config["grl_config"]["horizon_fn"] = horizon_fn
    config["debug"] = debug
    return config
