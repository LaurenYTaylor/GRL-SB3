import numpy as np
from curriculum_action_choice_utils import (
    agent_type_action_choice,
    goal_distance_action_choice,
    timestep_action_choice,
    exp_timestep_action_choice,
    variance_action_choice,
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


CURRICULUM_FNS = {
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
