import numpy as np
import torch

REWARD_VAR_MAP = {}
REWARD_VAR_CURRIC_MAP = {}
SAMPLE_PERC = 0.5


def reward_var_action_choice(config):
    if len(config["curriculum_stages"]) == 0:
        return False, None
    if REWARD_VAR_MAP == {}:
        return False, None
    all_ts = []
    for i in range(config["curriculum_stage_idx"] + 1):
        all_ts.extend(REWARD_VAR_CURRIC_MAP[i])
    if config["time_step"] in all_ts:
        use_learner = True
    # reward_var = REWARD_VAR_MAP[config["time_step"]]
    # if reward_var <= config["curriculum_stages"][config["curriculum_stage_idx"]]:
    #     use_learner = True
    # elif (config["curriculum_stage_idx"] != len(config["curriculum_stages"]) - 1) and (
    #     reward_var <= config["curriculum_stages"][config["curriculum_stage_idx"] + 1]
    # ):
    #     if np.random.random() < SAMPLE_PERC:
    #         use_learner = True
    #     else:
    #         use_learner = False
    else:
        use_learner = False
    return use_learner, REWARD_VAR_MAP[config["curriculum_stage_idx"]]


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
