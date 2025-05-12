from typing import Union, Any, Optional

import numpy as np
from copy import deepcopy
import torch as th
import torch.nn.functional as F


from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import polyak_update

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    RolloutReturn,
    Schedule,
    TrainFreq,
    TrainFrequencyUnit,
)
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer


def _store_transition_patch(
    self,
    replay_buffer: ReplayBuffer,
    buffer_action: np.ndarray,
    new_obs: Union[np.ndarray, dict[str, np.ndarray]],
    reward: np.ndarray,
    dones: np.ndarray,
    infos: list[dict[str, Any]],
) -> None:
    """
    Store transition in the replay buffer.
    We store the normalized action and the unnormalized observation.
    It also handles terminal observations (because VecEnv resets automatically).

    :param replay_buffer: Replay buffer object where to store the transition.
    :param buffer_action: normalized action
    :param new_obs: next observation in the current episode
        or first observation of the episode (when dones is True)
    :param reward: reward for the current transition
    :param dones: Termination signal
    :param infos: List of additional information about the transition.
        It may contain the terminal observations and information about timeout.
    """
    # Store only the unnormalized version
    if self._vec_normalize_env is not None:
        new_obs_ = self._vec_normalize_env.get_original_obs()
        reward_ = self._vec_normalize_env.get_original_reward()
    else:
        # Avoid changing the original ones
        self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

    # Avoid modification by reference
    next_obs = deepcopy(new_obs_)
    # As the VecEnv resets automatically, new_obs is already the
    # first observation of the next episode
    for i, done in enumerate(dones):
        if done and infos[i].get("terminal_observation") is not None:
            if isinstance(next_obs, dict):
                next_obs_ = infos[i]["terminal_observation"]
                # VecNormalize normalizes the terminal observation
                if self._vec_normalize_env is not None:
                    next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                # Replace next obs for the correct envs
                for key in next_obs.keys():
                    next_obs[key][i] = next_obs_[key]
            else:
                next_obs[i] = infos[i]["terminal_observation"]
                # VecNormalize normalizes the terminal observation
                if self._vec_normalize_env is not None:
                    next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])  # type: ignore[assignment]

    replay_buffer.add(
        self._last_original_obs,  # type: ignore[arg-type]
        next_obs,  # type: ignore[arg-type]
        buffer_action,
        reward_,
        dones,
        infos,
    )

    self._last_obs = new_obs
    # Save the unnormalized observation
    if self._vec_normalize_env is not None:
        self._last_original_obs = new_obs_


def train_patch(self, gradient_steps: int, batch_size: int = 64) -> None:
    # Switch to train mode (this affects batch norm / dropout)
    self.policy.set_training_mode(True)

    # Update optimizers learning rate
    optimizers = [self.actor.optimizer, self.critic.optimizer]
    if self.ent_coef_optimizer is not None:
        optimizers += [self.ent_coef_optimizer]

    # Update learning rate according to lr schedule
    self._update_learning_rate(optimizers)

    ent_coef_losses, ent_coefs = [], []
    actor_losses, critic_losses = [], []

    for gradient_step in range(gradient_steps):
        # Sample replay buffer
        replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
        # We need to sample because `log_std` may have changed between two gradient steps
        if self.use_sde:
            self.actor.reset_noise()

        # Action by the current actor for the sampled state
        actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
        log_prob = log_prob.reshape(-1, 1)

        ent_coef_loss = None
        if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            ent_coef = th.exp(self.log_ent_coef.detach())
            assert isinstance(self.target_entropy, float)
            ent_coef_loss = -(
                self.log_ent_coef * (log_prob + self.target_entropy).detach()
            ).mean()
            ent_coef_losses.append(ent_coef_loss.item())
        else:
            ent_coef = self.ent_coef_tensor

        ent_coefs.append(ent_coef.item())

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()

        with th.no_grad():
            # Select action according to policy
            next_actions, next_log_prob = self.actor.action_log_prob(
                replay_data.next_observations
            )
            # Compute the next Q values: min over all critics targets
            next_q_values = th.cat(
                self.critic_target(replay_data.next_observations, next_actions), dim=1
            )
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            # add entropy term
            next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
            # td error + entropy term
            target_q_values = (
                replay_data.rewards
                + (1 - replay_data.dones) * self.gamma * next_q_values
            )

        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        current_q_values = self.critic(replay_data.observations, replay_data.actions)

        # Compute critic loss
        critic_loss = 0.5 * sum(
            F.mse_loss(current_q, target_q_values) for current_q in current_q_values
        )
        assert isinstance(critic_loss, th.Tensor)  # for type checker
        critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

        # Optimize the critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Compute actor loss
        # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
        # Min over all critic networks
        q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
        min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
        actor_losses.append(actor_loss.item())

        # Optimize the actor
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Update target networks
        if gradient_step % self.target_update_interval == 0:
            polyak_update(
                self.critic.parameters(), self.critic_target.parameters(), self.tau
            )
            # Copy running stats, see GH issue #996
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

    self._n_updates += gradient_steps

    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
    self.logger.record("train/ent_coef", np.mean(ent_coefs))
    self.logger.record("train/actor_loss", np.mean(actor_losses))
    self.logger.record("train/critic_loss", np.mean(critic_losses))
    if len(ent_coef_losses) > 0:
        self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


def collect_rollouts_patch(
    self,
    env: VecEnv,
    callback: BaseCallback,
    train_freq: TrainFreq,
    replay_buffer: ReplayBuffer,
    action_noise: Optional[ActionNoise] = None,
    learning_starts: int = 0,
    log_interval: Optional[int] = None,
) -> RolloutReturn:
    """
    Collect experiences and store them into a ``ReplayBuffer``.

    :param env: The training environment
    :param callback: Callback that will be called at each step
        (and at the beginning and end of the rollout)
    :param train_freq: How much experience to collect
        by doing rollouts of current policy.
        Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
        or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
        with ``<n>`` being an integer greater than 0.
    :param action_noise: Action noise that will be used for exploration
        Required for deterministic policy (e.g. TD3). This can also be used
        in addition to the stochastic policy for SAC.
    :param learning_starts: Number of steps before learning for the warm-up phase.
    :param replay_buffer:
    :param log_interval: Log data every ``log_interval`` episodes
    :return:
    """
    # Switch to eval mode (this affects batch norm / dropout)
    self.policy.set_training_mode(False)

    num_collected_steps, num_collected_episodes = 0, 0

    assert isinstance(env, VecEnv), "You must pass a VecEnv"
    assert train_freq.frequency > 0, "Should at least collect one step or episode."

    if env.num_envs > 1:
        assert (
            train_freq.unit == TrainFrequencyUnit.STEP
        ), "You must use only one env when doing episodic training."

    if self.use_sde:
        self.actor.reset_noise(env.num_envs)  # type: ignore[operator]

    callback.on_rollout_start()
    continue_training = True
    self.used_learner = []
    while should_collect_more_steps(
        train_freq, num_collected_steps, num_collected_episodes
    ):
        if (
            self.use_sde
            and self.sde_sample_freq > 0
            and num_collected_steps % self.sde_sample_freq == 0
        ):
            # Sample a new noise matrix
            self.actor.reset_noise(env.num_envs)  # type: ignore[operator]

        # Select action randomly or according to policy
        actions, buffer_actions, use_learner = self._sample_action(
            learning_starts, action_noise, env.num_envs
        )
        self.used_learner.append(use_learner)
        new_obs, rewards, dones, infos = env.step(actions)
        try:
            last_used_learner = self.used_learner[-2]
        except IndexError:
            last_used_learner = None
        self.num_timesteps += env.num_envs
        if use_learner:
            num_collected_steps += 1

        # Give access to local variables
        callback.update_locals(locals())
        # Only stop training if return value is False, not when it is None.
        if not callback.on_step():
            return RolloutReturn(
                num_collected_steps * env.num_envs,
                num_collected_episodes,
                continue_training=False,
            )

        # Retrieve reward and episode length if using Monitor wrapper
        self._update_info_buffer(infos, dones)

        # Store data in replay buffer (normalized action and unnormalized observation)
        if use_learner:
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)  # type: ignore[arg-type]
        else:
            self._last_obs = new_obs
            self._last_original_obs = self._last_obs
        self._update_current_progress_remaining(
            self.num_timesteps, self._total_timesteps
        )

        # For DQN, check if the target network should be updated
        # and update the exploration schedule
        # For SAC/TD3, the update is dones as the same time as the gradient update
        # see https://github.com/hill-a/stable-baselines/issues/900
        self._on_step()
        for idx, done in enumerate(dones):
            if done:
                # Update stats
                num_collected_episodes += 1
                self._episode_num += 1
                self.used_learner = []
                if action_noise is not None:
                    kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                    action_noise.reset(**kwargs)

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self.dump_logs()
    callback.on_rollout_end()

    return RolloutReturn(
        num_collected_steps * env.num_envs, num_collected_episodes, continue_training
    )
