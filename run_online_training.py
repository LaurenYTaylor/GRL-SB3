import argparse
import ray
from ray import tune
from configuration import get_config
from grl_utils import run_grl_training, ray_grl_training, hyperparam_training
import configuration as exp_config


def train(
    env_name, horizon_fn, seeds, grl_buffer, multirun=False, tune=False, debug=False
):
    if tune:
        hyperparam_training(get_config(env_name, horizon_fn, grl_buffer, debug))
    elif not debug:
        if multirun:
            env_names = exp_config.env_names
            horizon_fns = ["agent_type", "time_step"]
            grl_buffer = [True, False]
        else:
            env_names = [env_name]
            horizon_fns = [horizon_fn]
            grl_buffer = [grl_buffer]
        object_references = [
            ray_grl_training.remote(get_config(env_name, horizon_fn, gb, debug), seed)
            for env_name in env_names
            for seed in range(seeds)
            for horizon_fn in horizon_fns
            for gb in grl_buffer
        ]

        all_data = []
        while len(object_references) > 0:
            finished, object_references = ray.wait(object_references, timeout=7.0)
            data = ray.get(finished)
            all_data.extend(data)
    else:
        run_grl_training(get_config(env_name, horizon_fn, grl_buffer, debug), 0)


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
        help="Curriculum horizon function",
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
        "--multirun",
        action="store_true",
        default=False,
        help="Run all config options",
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

    argparse.add_argument(
        "--grl_buffer",
        action="store_true",
        default=False,
        help="Use double buffer for guide and learner",
        required=False,
    )

    args = argparse.parse_args()

    train(
        args.env_name,
        args.horizon_fn,
        args.num_seeds,
        args.grl_buffer,
        args.multirun,
        args.tune,
        args.debug,
    )
