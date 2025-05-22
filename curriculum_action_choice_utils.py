import numpy as np
import torch
from goal_distance_fns import goal_dist_calc

REWARD_VAR_MAP = {}
SAMPLE_PERC = 0.5


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
    var = config["variance_fn"](torch.Tensor(config["obs"])).detach().cpu()
    var = torch.clip(torch.exp(var), 1e-4, 100000000)
    var = var.item()
    if len(config["curriculum_stages"]) == 0:
        return False, var
    if var <= config["curriculum_stage"]:
        use_learner = True
    return use_learner, var


def reward_var_action_choice(config):
    if REWARD_VAR_MAP == {}:
        return False, None
    reward_var = REWARD_VAR_MAP[config["time_step"]]
    if reward_var <= config["curriculum_stages"][config["curriculum_stage_idx"]]:
        use_learner = True
    elif (config["curriculum_stage_idx"] != len(config["curriculum_stages"] - 1)) and (
        reward_var <= config["curriculum_stages"][config["curriculum_stage_idx"] + 1]
    ):
        if np.random.random() < SAMPLE_PERC:
            use_learner = True
        else:
            use_learner = False
    else:
        use_learner = False
    return use_learner, reward_var


def exp_timestep_action_choice(config):
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
    if len(config["curriculum_stages"]) == 0:
        return False, config["time_step"]

    remaining_stages = len(config["curriculum_stages"]) - config["curriculum_stage_idx"]
    # f_log = lambda c, n: c*np.emath.logn(remaining_stages, (remaining_stages**(1/c)-1)/remaining_stages+1)
    f_exp = lambda d, n, x: d * np.exp(np.log((1 + d) / d) / n * x) - d
    if remaining_stages == 1:
        compare_float = 1.0
    else:
        stages = config["curriculum_stages"]
        compare_float = np.clip(
            f_exp(
                config["exp_time_step_coeff"],
                stages[-config["curriculum_stage_idx"] - 1],
                config["time_step"],
            ),
            0.0,
            1.0,
        )
    use_learner = False
    if np.random.random() < compare_float:
        use_learner = True
    return use_learner, compare_float


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
    if len(config["curriculum_stages"]) == 0:
        return False, config["time_step"]
    if (
        config["time_step"]
        >= config["curriculum_stages"][config["curriculum_stage_idx"]]
    ):
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
    # check if threshold will be exceeded if the learner is chosen to be used
    if_learner_used = config["curriculum_val_ep"][:]
    if_learner_used.append(1.0)
    curriculum_val_ep = np.mean(if_learner_used)

    if len(config["curriculum_stages"]) == 0:
        return False, 0.0
    if curriculum_val_ep <= config["curriculum_stage"]:
        use_learner = np.random.sample() < config["curriculum_stage"]
    return use_learner, float(use_learner)


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
    if len(config["curriculum_stages"]) == 0:
        return False, goal_dist
    if goal_dist <= config["curriculum_stage"]:
        use_learner = True
    return use_learner, goal_dist
