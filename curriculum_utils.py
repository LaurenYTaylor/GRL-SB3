import numpy as np
from curriculum_action_choice_utils import (
    agent_type_action_choice,
    goal_distance_action_choice,
    timestep_action_choice,
    variance_action_choice,
)


def max_accumulator(v):
    return np.max(v)


def mean_accumulator(v):
    return np.mean(v)


def static_accumulator(v):
    return 1


def max_to_min_curriculum(guide_stage, n_curriculum_stages):
    """
    Generates a max to min curriculum (for time step).
    """
    curric = np.linspace(guide_stage, 0, n_curriculum_stages + 1)
    return curric[1:]


def min_to_max_curriculum(guide_stage, n_curriculum_stages):
    """
    Generates a min to max curriculum (for goal distance, variance and agent type).
    """
    curric = np.linspace(0, guide_stage, n_curriculum_stages + 1)
    return curric[1:]


CURRICULUM_FNS = {
    "time_step": {
        "action_choice_fn": timestep_action_choice,
        "accumulator_fn": mean_accumulator,
        "generate_curriculum_fn": max_to_min_curriculum,
    },
    "agent_type": {
        "action_choice_fn": agent_type_action_choice,
        "accumulator_fn": static_accumulator,
        "generate_curriculum_fn": min_to_max_curriculum,
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
