import os
import copy
from stable_baselines3.common.noise import ActionNoise
import numpy as np
from numpy.typing import DTypeLike
from GRLReplayBuffer import GRLReplayBuffer
from deepmerge import Merger

# Define the merge strategy
merger = Merger(
    [(dict, ["merge"])],  # merge dictionaries recursively
    ["override"],
    ["override"],  # fallback strategy
)


class NormalActionNoise(ActionNoise):
    """
    A Gaussian action noise.

    :param mean: Mean value of the noise
    :param sigma: Scale of the noise (std here)
    :param dtype: Type of the output noise
    """

    def __init__(
        self, mean: np.ndarray, sigma: np.ndarray, dtype: DTypeLike = np.float32
    ) -> None:
        self._mu = mean
        self._sigma = sigma
        self._dtype = dtype
        super().__init__()

    def __call__(self) -> np.ndarray:
        return self._dtype(np.random.normal(self._mu, self._sigma))

    def __repr__(self) -> str:
        return f"NormalActionNoise(mu={self._mu}, sigma={self._sigma})"


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
algorithms = [
    "SAC",
    "SAC",
    "PPO",
    "SAC",
    "SAC",
    "SAC",
]  # , #"SAC", "SAC", "PPO", "SAC"]
# algorithms = ["SAC", "SAC", "PPO"]
paths_dict = dict(
    zip(
        env_names,
        [
            f"{os.getcwd()}/pretrained_1million/{env_names[i]}_{algorithms[i].lower()}.zip"
            for i in range(len(env_names))
        ],
    )
)
env_names.append("CombinationLock-v1")
paths_dict["CombinationLock-v1"] = (
    f"{os.getcwd()}/pretrained_10000/CombinationLock-v1_td3.zip"
)
algorithm_dict = dict(zip(env_names, algorithms))

DEFAULT_CONFIG = {
    "env_name": "AdroitHandHammer-v1",
    "eval_freq": 10000,
    "n_eval_episodes": 500,
    "pretrain_eval_episodes": 500,
    "training_steps": 50000,
    "seed": 0,
    "debug": False,
    "tune": False,
    "multirun": False,
    "save_model_path": None,  # replace with desired path
    "tensorboard_log": None,  # replace with desired path
    "grl_config": {
        "horizon_fn": "agent_type",
        "n_curriculum_stages": 10,
        "variance_fn": None,
        "rolling_mean_n": 1,
        "tolerance": 0.01,
        "guide_randomness": 0.0,
        "exp_time_step_coeff": 0.001,
        "grl_buffer": True,
        "delay_training": True,
        "guide_in_actor_loss": False,
    },
    "algo_config": {
        "buffer_size": 100000,
        "batch_size": 128,
        "learning_starts": 0,
        "train_freq": 1,
        "gradient_steps": 1,
        "learning_rate": 0.00005,
        # "action_noise": NormalActionNoise(mean=0.0, sigma=0.1),
        "replay_buffer_class": "GRLReplayBuffer",
        "replay_buffer_kwargs": {"perc_guide_sampled": ("cs", "cs")},
    },
}


# def get_config(env_name, horizon_fn, grl_buffer, debug):
def get_config(**kwargs):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config = merger.merge(config, kwargs)
    config["pretrained_path"] = paths_dict[config["env_name"]]
    config["algo"] = algorithm_dict.get(config["env_name"], "SAC")
    # config["grl_config"]["guide_randomness"] = config["grl_config"]["guide_randomness"] + (1 / config["grl_config"]["n_curriculum_stages"])
    debug, tune = "", ""
    if config["debug"]:
        debug = "-debug"
    if config["tune"]:
        tune = "-tune"
    config["project_name"] = f"sb3-TD3{debug}{tune}"
    config["log_path"] = (
        f"./saved_models/{config['env_name']}_{config['algo']}_{config['seed']}"
    )
    config["algo_config"]["replay_buffer_class"] = globals()[
        config["algo_config"]["replay_buffer_class"]
    ]
    print(config)
    return config
