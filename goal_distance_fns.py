import numpy as np


def adroit_hand(state, _):
    distance = np.linalg.norm(state[39:45])
    return distance


def adroit_hammer(state, env):
    if state.shape[0] == 1:
        state = state[0]
    nail_pos = env.unwrapped.data.site_xpos[env.unwrapped.target_obj_site_id].ravel()
    goal_pos = env.unwrapped.data.site_xpos[env.unwrapped.goal_site_id].ravel()
    distance = (
        (1 - int(np.linalg.norm(goal_pos - nail_pos) < 0.01))
        * np.linalg.norm(goal_pos - nail_pos)
        * (
            np.linalg.norm(state[42:45] - state[36:39])
            + np.linalg.norm(state[36:39] - state[33:36])
        )
    )
    return distance


def adroit_relocate(state, _):
    distance = np.linalg.norm(state[30:33]) + np.linalg.norm(state[33:36])
    return distance


def adroit_door(state, _):
    if state[28] == 0:
        mult = 0.00001
    else:
        mult = np.abs(state[28])
    distance = (1 / mult) * (1 - state[38])
    return distance


def pusher(state, _):
    distance = np.linalg.norm(state[20:] - state[17:20]) + np.linalg.norm(
        state[17:20] - state[14:17]
    )
    return distance


def inverted_double_pendulum(state, _):
    distance = np.linalg.norm(state[1]) + np.linalg.norm(state[2])
    return distance


def hopper(state, env):
    xpos = env.unwrapped.data.qpos[0]
    if xpos < -1:
        distance = np.abs(xpos) * 1 / state[0]
    elif xpos < 1:
        distance = 1 * 1 / state[0]
    else:
        distance = 1 / env.unwrapped.data.qpos[0]
    return distance


def antmaze(_, env):
    goal_state = np.array(env.target_goal)
    current_state = np.array(env.get_xy())
    goal_dist = np.linalg.norm(goal_state - current_state)
    return goal_dist


def lunar_lander(state, _):
    # Compares the x,y pos and whether the lander's
    # legs are touching the ground

    goal_state = np.array([0, 0, 0, 0, 0, 0, 1, 1])[:2][-2:]
    current_state = state[:2][-2:]
    goal_dist = np.linalg.norm(goal_state - current_state)
    return goal_dist


GOAL_MAP = {
    "antmaze-umaze-v2": antmaze,
    "antmaze-umaze-diverse-v2": antmaze,
    "antmaze-medium-play-v2": antmaze,
    "antmaze-medium-diverse-v2": antmaze,
    "antmaze-large-play-v2": antmaze,
    "antmaze-large-diverse-v2": antmaze,
    "LunarLander-v2": lunar_lander,
    "AdroitHandPen-v1": adroit_hand,
    "AdroitHandHammer-v1": adroit_hammer,
    "AdroitHandRelocate-v1": adroit_relocate,
    "AdroitHandDoor-v1": adroit_door,
    "Hopper-v5": hopper,
    "Pusher-v5": pusher,
    "InvertedDoublePendulum-v5": inverted_double_pendulum,
}


def goal_dist_calc(state, env):
    if isinstance(env.get_attr("spec"), list):
        env = env.envs[0]
    goal_dist_fn = GOAL_MAP[env.unwrapped.spec.id]
    if state.shape[0] == 1:
        state = state[0]
    return goal_dist_fn(state, env)
