import os
import copy
from stable_baselines3.common.noise import ActionNoise
import numpy as np
from numpy.typing import DTypeLike


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
    "eval_freq": 3000,
    "n_eval_episodes": 150,
    "pretrain_eval_episodes": 500,
    "training_steps": 50000,
    "grl_config": {
        "horizon_fn": "time_step",
        "n_curriculum_stages": 10,
        "variance_fn": None,
        "rolling_mean_n": 1,
        "tolerance": 0.01,
        "guide_randomness": 0.0,
        "exp_time_step_coeff": 0.001,
        "grl_buffer": True,
    },
    "algo_config": {
        "buffer_size": 10000,
        "batch_size": 256,
        "learning_starts": 0,
        "train_freq": 1,
        "gradient_steps": 1,
        "learning_rate": 0.0005,
        "action_noise": NormalActionNoise(mean=0.0, sigma=1.5),
        # "target_policy_noise": 0.05
    },
}


def get_config(env_name, horizon_fn, grl_buffer, debug):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["algo"] = algorithm_dict.get(env_name, "SAC")
    config["env_name"] = env_name
    config["pretrained_path"] = paths_dict[env_name]
    config["grl_config"]["horizon_fn"] = horizon_fn
    config["grl_config"]["grl_buffer"] = grl_buffer
    config["debug"] = debug
    return config
