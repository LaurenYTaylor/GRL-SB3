import numpy as np
from curriculum_action_choice_utils import (
    agent_type_action_choice,
    goal_distance_action_choice,
    timestep_action_choice,
    exp_timestep_action_choice,
    variance_action_choice,
    reward_var_action_choice,
)


def max_accumulator(v):
    return np.max(v)


def mean_accumulator(v):
    return np.mean(v)


def static_accumulator(v):
    return 1


def max_to_min_curriculum(guide_vals, n_curriculum_stages):
    """
    Generates a max to min curriculum (for time step).
    """
    # curric = np.linspace(guide_stage, 0, n_curriculum_stages + 1)
    curric = np.percentile(
        guide_vals, np.linspace(100 - 100 / n_curriculum_stages, 0, n_curriculum_stages)
    )
    return curric


def min_to_max_curriculum(guide_vals, n_curriculum_stages):
    """
    Generates a min to max curriculum (for goal distance, variance and agent type).
    """
    # curric = np.linspace(0, guide_stage, n_curriculum_stages + 1)
    # curric = np.percentile(
    #    guide_vals, np.linspace(100 / n_curriculum_stages, 100, n_curriculum_stages)
    # )
    curric = np.percentile(
        guide_vals,
        np.linspace(0, 100 - (100 / n_curriculum_stages), n_curriculum_stages),
    )
    return curric


def agent_type_curriculum(_, n_curriculum_stages):
    curric = np.linspace(0, 1, n_curriculum_stages + 1)[1:]
    return curric


def reward_var_curriculum(guide_vals, n_curriculum_stages):
    step_means = []
    for k, v in guide_vals.items():
        step_means.append(np.array(v))
    rewards_matrix = np.array(step_means)
    per_episode = rewards_matrix.T

    gamma = 0.99
    returns = np.zeros_like(per_episode)
    for i in range(per_episode.shape[1], 0, -1):
        if i == per_episode.shape[1]:
            returns[:, i - 1] = per_episode[:, i - 1]
        else:
            returns[:, i - 1] = per_episode[:, i - 1] + gamma * returns[:, i]
    # reward_diffs.append(0.0)
    # curric_dict = dict(zip(range(len(reward_diffs)), reward_diffs))
    var_returns = np.var(returns, axis=0)
    return_diff = np.abs(var_returns[1:] - var_returns[:-1])

    perc = np.percentile(
        return_diff[return_diff != 0],
        np.linspace((100 / n_curriculum_stages), 100, n_curriculum_stages),
    )
    perc_dict = {}
    for i, p in enumerate(perc):
        if i == 0:
            idxs = np.where(return_diff <= p)[0]
        else:
            idxs = np.where((return_diff <= p) & (return_diff > perc[i - 1]))[0]
        step_dict = dict(zip(list(idxs), [p] * len(idxs)))
        perc_dict.update(step_dict)
    perc_dict[max(perc_dict.keys()) + 1] = var_returns[-1]
    import curriculum_action_choice_utils

    curriculum_action_choice_utils.REWARD_VAR_MAP = perc_dict
    return perc


CURRICULUM_FNS = {
    "reward_var": {
        "action_choice_fn": reward_var_action_choice,
        "accumulator_fn": None,
        "generate_curriculum_fn": reward_var_curriculum,
    },
    "time_step": {
        "action_choice_fn": timestep_action_choice,
        "accumulator_fn": mean_accumulator,
        "generate_curriculum_fn": max_to_min_curriculum,
    },
    "exp_time_step": {
        "action_choice_fn": exp_timestep_action_choice,
        "accumulator_fn": mean_accumulator,
        "generate_curriculum_fn": min_to_max_curriculum,
    },
    "agent_type": {
        "action_choice_fn": agent_type_action_choice,
        "accumulator_fn": static_accumulator,
        "generate_curriculum_fn": agent_type_curriculum,
    },
    "goal_dist": {
        "action_choice_fn": goal_distance_action_choice,
        "accumulator_fn": max_accumulator,
        "generate_curriculum_fn": min_to_max_curriculum,
    },
    "variance": {
        "action_choice_fn": variance_action_choice,
        "accumulator_fn": mean_accumulator,
        "generate_curriculum_fn": min_to_max_curriculum,
    },
}
