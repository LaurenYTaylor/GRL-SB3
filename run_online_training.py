import argparse
import ray
from ray import tune
from configuration import get_config, DEFAULT_CONFIG
from grl_utils import run_grl_training, ray_grl_training, hyperparam_training
import configuration as exp_config
import copy
import itertools


def collect_all_keys(d):
    keys = []
    for k, v in d.items():
        keys.append(k)
        if isinstance(v, dict):
            updated_dict = {}
            for sub_k, sub_v in v.items():
                sub_k = k + "/" + sub_k
                updated_dict[sub_k] = sub_v
            d[k] = updated_dict
            keys.extend(collect_all_keys(updated_dict))
    return keys


def parse_number(s):
    if "[" in s and "]" in s:
        return [parse_number(s) for s in s[1:-1].split(",")]
    elif s == "True":
        return True
    elif s == "False":
        return False
    elif '"' in s or "'" in s:
        return s[1:-1]
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def train(train_args, multi_run):
    if len(multi_run) == 0:
        config = get_config(**train_args)
        if "tune" in train_args and train_args["tune"]:
            hyperparam_training(config)
            return True
        if "debug" in train_args and train_args["debug"]:
            run_grl_training(config)
            return True
    object_references = []
    idx_combos = itertools.product(*[list(range(multi_run[k])) for k in multi_run])
    for idxs in idx_combos:
        current = copy.deepcopy(train_args)
        for n, name in enumerate(multi_run):
            if len(name) > 1:
                d = current
                for key in name[:-1]:
                    d = d.setdefault(key, {})
                d[name[-1]] = d[name[-1]][idxs[n]]
            else:
                current[name[0]] = current[name[0]][idxs[n]]
        print(current)
        config = get_config(**current)
        object_references.append(ray_grl_training.remote(config))

    all_data = []
    while len(object_references) > 0:
        finished, object_references = ray.wait(object_references, timeout=7.0)
        data = ray.get(finished)
        all_data.extend(data)
    return True


if __name__ == "__main__":
    DEFAULT_CONFIG_KEYS = collect_all_keys(copy.deepcopy(DEFAULT_CONFIG))

    argparse = argparse.ArgumentParser()
    argparse.add_argument(
        "--num_seeds", type=int, default=1, help="Number of seeds to run"
    )
    seed_arg, args = argparse.parse_known_args()

    arg_starts = [i for i, start in enumerate(args) if "--" in start]
    arg_dict = {}
    multi_run = {}
    for j, arg_idx in enumerate(arg_starts):
        if j == (len(arg_starts) - 1):
            vals = args[arg_idx + 1 :]
        else:
            vals = args[arg_idx + 1 : arg_starts[j + 1]]

        arg_name = args[arg_idx][2:]  # removes --
        parsed_vals = [parse_number(v) for v in vals]
        if len(vals) == 1:
            parsed_vals = parsed_vals[0]
        parts = arg_name.split("/")
        key_str = "\n".join(DEFAULT_CONFIG_KEYS)
        assert (
            arg_name in DEFAULT_CONFIG_KEYS
        ), f"Argument {arg_name} not in DEFAULT_CONFIG_KEYS. Config options are:\n {key_str}"
        if arg_name in DEFAULT_CONFIG_KEYS:
            current = arg_dict
            for p in parts[:-1]:
                current = current.setdefault(p, {})
            current[parts[-1]] = parsed_vals
            if len(vals) > 1:
                multi_run[tuple(parts)] = len(vals)

    for seed in range(seed_arg.num_seeds):
        arg_dict["seed"] = seed
        success = train(arg_dict, multi_run)
