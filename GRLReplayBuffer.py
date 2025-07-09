from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np
from typing import Union, NamedTuple, Optional
import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecNormalize


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    time_steps: th.Tensor
    used_learner: th.Tensor


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
        self.used_learner = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.time_step = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add_grl_specific(self, grl_info: dict) -> None:
        if self.full:
            self.prev_pos = self.buffer_size - 1
        else:
            self.prev_pos = self.pos - 1
        self.used_learner[self.prev_pos] = grl_info.get("used_learner", 0)
        self.time_step[self.prev_pos] = grl_info.get("time_step", 0)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        learner_used = np.where(self.used_learner[:upper_bound])[0]
        guide_used = np.where(self.used_learner[:upper_bound] == 0)[0]
        if len(guide_used) == 0:
            n_learner = batch_size
        elif len(learner_used) == 0:
            n_learner = 0  # this batch will be ignored in training
        else:
            n_learner = int(batch_size * 0.5)
        learner_inds = np.random.choice(learner_used, size=n_learner, replace=True)
        guide_inds = np.random.choice(
            guide_used, size=batch_size - n_learner, replace=True
        )
        batch_inds = np.concatenate((learner_inds, guide_inds))
        return self._get_samples(batch_inds, env=env)

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
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
            self.time_step[batch_inds, env_indices].reshape(-1, 1),
            self.used_learner[batch_inds, env_indices].reshape(-1, 1),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
