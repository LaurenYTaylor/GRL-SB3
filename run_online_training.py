from grl_utils import run_grl_training
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from pathlib import Path
import copy
import os
env_names = ['AdroitHandPen-v1', 'AdroitHandHammer-v1', 'AdroitHandRelocate-v1', 'AdroitHandDoor-v1', 'Pusher-v5','InvertedDoublePendulum-v5', 'Hopper-v5']
algorithms = ['SAC','SAC','PPO','SAC','SAC','SAC','PPO']
algorithm_dict = dict(zip(env_names, algorithms))
paths_dict = dict(zip(env_names, [f"{os.getcwd()}/pretrained_1million/{env_names[i]}_{algorithms[i].lower()}.zip" for i in range(len(env_names))]))

DEFAULT_CONFIG = {"eval_freq": 10000,
          "n_eval_episodes": 150,
          "training_steps": 1000000,
          "grl_config": {"horizon_fn": "agent_type",
                         "n_curriculum_stages": 10, "variance_fn": None, "rolling_mean_n": 5,
                         "tolerance": 0.01,
                         "guide_randomness": 0.1},
          "algo_config": {"buffer_size": 100000,
                          "batch_size": 256,"learning_starts": 0,
                         }
          }

seeds = [0]

def hyperparam_training(tune_config):
    tune_config["eval_freq"] = tune.choice([10000, 20000, 30000])
    tune_config["n_eval_episodes"] = tune.choice([100, 150, 200])
    tune_config["algo_config"]["buffer_size"] = tune.choice([10000, 100000, 1000000])
    tune_config["grl_config"]["n_curriculum_stages"] = tune.choice([10, 15, 20])
    tune_config["learning_rate"] = tune.loguniform(1e-7, 1e-3)
    tune_config["tau"] = tune.loguniform(1e-4, 1e-2)
    tune_config["train_freq"] = tune.choice([1, 8, 32, 64])

    tuner = tune.Tuner(
            run_grl_training,
            tune_config=tune.TuneConfig(
                num_samples=50,
                scheduler=ASHAScheduler(time_attr="global_timestep",grace_period=100000, metric="eval_return", mode="max"),
            ),
            param_space=tune_config,
            run_config=tune.RunConfig(storage_path=Path("./hyperparam_results").resolve(), name="tuning")
        )
    results = tuner.fit()

if __name__ == "__main__":
    for env_name in env_names[:2]:
        for seed in seeds:
            config = copy.deepcopy(DEFAULT_CONFIG)
            config["algo"] = algorithm_dict[env_name]
            config["env_name"] = env_name
            config["pretrained_path"] = paths_dict[env_name]
            config["seed"] = seed
            #run_grl_training(config)
            hyperparam_training(config)
            config = copy.deepcopy(DEFAULT_CONFIG)
            config["n_curriculum_stages"] = 0
            run_grl_training(config)
