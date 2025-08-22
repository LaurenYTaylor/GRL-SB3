from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np
from typing import Union, NamedTuple, Optional, Any
import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecNormalize
from curriculum_utils import reward_var_curriculum
import torch


def softmax_with_temperature(logits, temperature=1.0):
    """
    Computes the softmax with a temperature parameter.

    Args:
        logits (np.ndarray): An array of raw output scores (logits).
        temperature (float): The temperature parameter. Must be positive.

    Returns:
        np.ndarray: An array of probabilities.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be a positive value.")

    scaled_logits = logits / temperature
    exp_logits = np.exp(
        scaled_logits - np.max(scaled_logits)
    )  # Subtract max for numerical stability
    probabilities = exp_logits / np.sum(exp_logits)
    return probabilities


def reward_diff_distro(guide_vals):
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
    return_diff = np.concatenate((return_diff, [var_returns[-1]]))
    # return_diff_prob = softmax_with_temperature(return_diff, temperature=10000)
    return return_diff


class GRLReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    noiseless_actions: th.Tensor
    time_steps: th.Tensor
    used_learner: th.Tensor


class GRLSimpleReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    time_steps: th.Tensor
    used_learner: th.Tensor
    noiseless_actions: th.Tensor


class GRLReplayBuffer(ReplayBuffer):
    """
    Extended Replay Buffer that allows for additional functionality.
    Inherits from Stable Baselines3's ReplayBuffer.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        curric_vals: dict = None,
        perc_guide_sampled: list[Union[float, str, None], Union[float, str, None]] = [
            None,
            None,
        ],
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )
        self.noiseless_actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=self._maybe_cast_dtype(action_space.dtype),
        )
        self.time_step = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.reward_diffs = reward_diff_distro(curric_vals)
        self.guide_observations = np.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape),
            dtype=observation_space.dtype,
        )

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_guide_observations = np.zeros(
                (self.buffer_size, self.n_envs, *self.obs_shape),
                dtype=observation_space.dtype,
            )

        self.guide_actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=self._maybe_cast_dtype(action_space.dtype),
        )
        self.guide_noiseless_actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=self._maybe_cast_dtype(action_space.dtype),
        )
        self.guide_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.guide_dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.guide_timeouts = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.guide_timestep = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.guide_full = False
        self.guide_pos = 0
        self.guide_size = 0
        self.learner_frac = 0.1
        self.perc_guide_sampled = perc_guide_sampled

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))
        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))
        if self.recent_used_learner == 1:
            self.observations[self.pos] = np.array(obs)
            self.actions[self.pos] = np.array(action)
            self.rewards[self.pos] = np.array(reward)
            self.dones[self.pos] = np.array(done)
            if self.optimize_memory_usage:
                self.observations[(self.pos + 1) % self.buffer_size] = np.array(
                    next_obs
                )
            else:
                self.next_observations[self.pos] = np.array(next_obs)
            if self.handle_timeout_termination:
                self.timeouts[self.pos] = np.array(
                    [info.get("TimeLimit.truncated", False) for info in infos]
                )
            self.pos += 1
            if self.pos == self.buffer_size:
                self.full = True
                self.pos = 0
        else:
            self.guide_observations[self.guide_pos] = np.array(obs)
            self.guide_actions[self.guide_pos] = np.array(action)
            self.guide_rewards[self.guide_pos] = np.array(reward)
            self.guide_dones[self.guide_pos] = np.array(done)
            if self.optimize_memory_usage:
                self.guide_observations[(self.pos + 1) % self.buffer_size] = np.array(
                    next_obs
                )
            else:
                self.next_guide_observations[self.pos] = np.array(next_obs)
            if self.handle_timeout_termination:
                self.guide_timeouts[self.guide_pos] = np.array(
                    [info.get("TimeLimit.truncated", False) for info in infos]
                )
            self.guide_pos += 1
            if self.guide_pos == self.buffer_size:
                self.guide_full = True
                self.guide_pos = 0

    def add_grl_specific(self, grl_info: dict) -> None:
        self.recent_used_learner = grl_info.get("used_learner", 0)

        if self.recent_used_learner:
            self.time_step[self.pos] = grl_info.get("time_step", 0)
            self.noiseless_actions[self.pos] = grl_info.get("noiseless_action", None)
        else:
            self.guide_timestep[self.guide_pos] = grl_info.get("time_step", 0)
            self.guide_noiseless_actions[self.guide_pos] = grl_info.get(
                "noiseless_action", None
            )

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        guide_upper_bound = self.buffer_size if self.guide_full else self.guide_pos
        learner_upper_bound = self.buffer_size if self.full else self.pos
        lower_samp, upper_samp = self.perc_guide_sampled
        if upper_samp == "cs":
            if lower_samp == "cs":
                guide_batch_n = int((1 - self.learner_frac) * batch_size)
            else:
                guide_batch_n = min(
                    int((1 - lower_samp) * batch_size),
                    int((1 - self.learner_frac) * batch_size),
                )
        elif isinstance(upper_samp, float) or isinstance(upper_samp, int):
            guide_batch_n = int((1 - lower_samp) * batch_size)

        if self.pos == 0 and not self.full:
            guide_batch_n = batch_size
        elif self.guide_pos == 0 and not self.guide_full:
            guide_batch_n = 0

        guide_inds = np.array([])
        learner_inds = np.array([])

        if guide_batch_n < batch_size:
            # learner_samp_ts = self.time_step[:learner_upper_bound].astype("int64")
            # learner_samp_logits = np.take(self.reward_diffs, learner_samp_ts)
            # learner_samp_probs = softmax_with_temperature(
            #    learner_samp_logits, temperature=10000
            # )
            learner_inds = np.random.choice(
                list(range(learner_upper_bound)),
                size=batch_size - guide_batch_n,
                replace=True,
                # p=learner_samp_probs.flatten(),
            )

        if guide_batch_n > 0:
            ##dists = np.sum(diff**2, axis=-1)
            # Find the nearest guide index for each learner observation
            # guide_inds = np.argmin(dists, axis=1).flatten()

            guide_inds = np.random.choice(
                list(range(guide_upper_bound)),
                size=guide_batch_n,
                replace=True,
                # p=learner_samp_probs.flatten(),
            )

        return self._get_samples(learner_inds, guide_inds, env=env)

    def _get_guide_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> GRLReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.guide_observations[
                    (batch_inds + 1) % self.buffer_size, env_indices, :
                ],
                env,
            )
        else:
            next_obs = self._normalize_obs(
                self.next_guide_observations[batch_inds, env_indices, :], env
            )

        used_learner = np.zeros(self.guide_timestep.shape)
        data = (
            self._normalize_obs(
                self.guide_observations[batch_inds, env_indices, :], env
            ),
            self.guide_actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (
                self.guide_dones[batch_inds, env_indices]
                * (1 - self.guide_timeouts[batch_inds, env_indices])
            ).reshape(-1, 1),
            self._normalize_reward(
                self.guide_rewards[batch_inds, env_indices].reshape(-1, 1), env
            ),
            self.guide_noiseless_actions[batch_inds, env_indices, :],
            self.guide_timestep[batch_inds, env_indices].reshape(-1, 1),
            # self.guide_used_learner[batch_inds, env_indices].reshape(-1, 1),
            used_learner[batch_inds, env_indices].reshape(-1, 1),
        )
        return GRLReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def _get_learner_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> GRLReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :],
                env,
            )
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, env_indices, :], env
            )

        used_learner = np.ones(self.time_step.shape)
        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (
                self.dones[batch_inds, env_indices]
                * (1 - self.timeouts[batch_inds, env_indices])
            ).reshape(-1, 1),
            self._normalize_reward(
                self.rewards[batch_inds, env_indices].reshape(-1, 1), env
            ),
            self.noiseless_actions[batch_inds, env_indices, :],
            self.time_step[batch_inds, env_indices].reshape(-1, 1),
            used_learner[batch_inds, env_indices].reshape(-1, 1),
        )
        return GRLReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def _get_samples(
        self,
        learner_inds: np.ndarray,
        guide_inds: np.array,
        env: Optional[VecNormalize] = None,
    ) -> GRLReplayBufferSamples:
        # Sample randomly the env idx
        if len(learner_inds) == 0:
            return self._get_guide_samples(guide_inds, env)
        if len(guide_inds) == 0:
            return self._get_learner_samples(learner_inds, env)

        guide_data = self._get_guide_samples(guide_inds, env)
        learner_data = self._get_learner_samples(learner_inds, env)
        data = {}

        for i, k in enumerate(guide_data._asdict().keys()):
            data[k] = torch.concat((guide_data._asdict()[k], learner_data._asdict()[k]))
            if i == 0:
                idx = torch.randperm(len(data[k]))
            data[k] = data[k][idx]

        return GRLReplayBufferSamples(*tuple(map(self.to_torch, tuple(data.values()))))
