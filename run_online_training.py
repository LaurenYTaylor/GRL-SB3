import argparse
import ray
from ray import tune
from configuration import get_config
from grl_utils import run_grl_training, ray_grl_training, hyperparam_training
import configuration as exp_config


def train(env_name, horizon_fn, seeds, tune=False, debug=False):
    if tune:
        hyperparam_training(get_config(env_name, horizon_fn))
    elif not debug:
        if env_name == "all":
            env_names = exp_config.env_names
        else:
            env_names = [env_name]
        if horizon_fn == "all":
            horizon_fns = ["agent_type", "time_step", "goal_dist", "variance"]
        else:
            horizon_fns = [horizon_fn]
        object_references = [
            ray_grl_training.remote(get_config(env_name, horizon_fn), seed)
            for env_name in env_names
            for seed in range(seeds)
            for horizon_fn in horizon_fns
        ]

        all_data = []
        while len(object_references) > 0:
            finished, object_references = ray.wait(object_references, timeout=7.0)
            data = ray.get(finished)
            all_data.extend(data)
    else:
        run_grl_training(get_config(env_name, horizon_fn), 0)


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument(
        "--env_name",
        type=str,
        help="Environment name",
        default="AdroitHandHammer-v1",
        required=False,
    )
    argparse.add_argument(
        "--horizon_fn",
        type=str,
        help="Currculum horizon function",
        default="agent_type",
        required=False,
    )
    argparse.add_argument(
        "--num_seeds",
        type=int,
        help="Number of experiments to run",
        default=1,
        required=False,
    )
    argparse.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning on this environment",
        required=False,
    )
    argparse.add_argument(
        "--debug", action="store_true", help="Run in debug (no Ray)", required=False
    )
    args = argparse.parse_args()
    train(args.env_name, args.horizon_fn, args.num_seeds, args.tune, args.debug)
