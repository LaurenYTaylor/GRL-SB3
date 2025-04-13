import numpy as np
import torch
from goal_distance_fns import goal_dist_calc

def variance_action_choice(config):
    """
    Determine whether to use the learner or guide based on the state variance.
    Unused parameter placeholders ensure the horizon functions have the same signature.

    Parameters
    ----------
    _ : Any
        Time config["time_step"], not used.
    s : numpy.ndarray
        The current state of the environment.
    _e : Any
        Environment, not used.
    config : JsrlTrainConfig
        The configuration parameters for the JSRL training process.

    Returns
    -------
    Tuple[bool, float]
        A tuple containing a boolean indicating whether to use the learner and the calculated state
        variance.

    """
    use_learner = False
    var = config.vf(torch.Tensor(config["obs"]))

    if np.isnan(config["curriculum_stage"]):
        return True, var
    if (var <= config["curriculum_stage"]):
        use_learner = True
    return use_learner, var

def timestep_action_choice(config):
    """
    Determine whether to use the learner or guide based on the timestep.
    Unused parameter placeholders ensure the horizon functions have the same signature.

    Parameters
    ----------
    config["time_step"] : int
        The current timestep.
    _s : Any
        State, not used.
    _e : Any
        Env, not used.
    config : JsrlTrainConfig
        The configuration parameters for the JSRL training process.

    Returns
    -------
    Tuple[bool, int]
        A tuple containing a boolean indicating whether to use the learner and the current timestep.

    """
    use_learner = False
    if np.isnan(config["curriculum_stage"]):
        return True, config["time_step"]
    if (config["time_step"] >= config["curriculum_stage"]):
        use_learner = True
    return use_learner, config["time_step"]

def agent_type_action_choice(config):
    """
    Determine whether to use the learner or guide based on the current % use of the learner throughout the episode.
    Unused parameter placeholders ensure the horizon functions have the same signature.

    Parameters
    ----------
    _st : int
        Timestep, not used.
    _s : Any
        State, not used.
    _e : Any
        Env, not used.
    config : JsrlTrainConfig
        The configuration parameters for the JSRL training process.

    Returns
    -------
    Tuple[bool, int]
        A tuple containing a boolean indicating whether to use the learner and the current timestep.

    """
    use_learner = False
    if_learner_used = config["curriculum_val_ep"][:]
    if_learner_used.append(1)
    curriculum_val_ep = np.mean(if_learner_used)

    if np.isnan(config["curriculum_stage"]):
        return True,1
    if (curriculum_val_ep <= config["curriculum_stage"]):
        use_learner = (np.random.sample() < config["curriculum_stage"])
    return use_learner, use_learner


def goal_distance_action_choice(config):
    """
    Determine whether to use the learner or guide based on the distance from the goal.
    Unused parameter placeholders ensure the horizon functions have the same signature.

    Parameters
    ----------
    _t : Any
        Time config["time_step"], not used.
    s : numpy.ndarray
        The current state of the environment.
    env : gym.Env
        The environment.
    config : JsrlTrainConfig
        The configuration parameters for the JSRL training process.

    Returns
    -------
    Tuple[bool, float]
        A tuple containing a boolean indicating whether to use the learner and the calculated goal distance.

    """
    use_learner = False
    goal_dist = goal_dist_calc(config["obs"], config["env"])
    if np.isnan(config["curriculum_stage"]):
        return True, goal_dist
    if (goal_dist <= config["curriculum_stage"]):
        use_learner = True
    return use_learner, goal_dist

