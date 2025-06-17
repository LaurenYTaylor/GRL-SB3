from typing import Optional, Union, Callable, Any

import ray
import torch
import wandb
import warnings
import gymnasium_robotics
import numpy as np
import gymnasium as gym

from ray import tune
from pathlib import Path
from stable_baselines3 import SAC, PPO
from gymnasium import spaces

from ray.tune.schedulers import ASHAScheduler
from stable_baselines3.common.noise import ActionNoise
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecMonitor,
    is_vecenv_wrapped,
)
from stable_baselines3.common import evaluation as sb3_eval
from grl_callbacks import (
    CurriculumMgmtCallback,
    CurriculumStageUpdateCallback,
    ModifiedEvalCallback,
)
from variance_trainer import StateDepFunction, VarianceLearner
from curriculum_utils import CURRICULUM_FNS
from experimental_utils import (
    collect_rollouts_patch,
)
from SACPolicy import SACPolicyPatch
from GRLReplayBuffer import GRLReplayBuffer

from typing import Any, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import SACPolicy

MODELS = {"SAC": SAC, "PPO": PPO}


def evaluate_imperfect_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    return_guide_vals: bool = False,
    warn: bool = True,
    randomness: float = 0.0,
    curriculum_fns: dict = None,
) -> Union[tuple[float, float], tuple[list[float], list[int]]]:
    """
    Runs the policy for ``n_eval_episodes`` episodes and outputs the average return
    per episode (sum of undiscounted rewards).
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a ``predict`` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to perform additional checks,
        called ``n_envs`` times after each step.
        Gets locals() and globals() passed as parameters.
        See https://github.com/DLR-RM/stable-baselines3/issues/1912 for more details.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean return per episode (sum of rewards), std of reward per episode.
        Returns (list[float], list[int]) when ``return_episode_rewards`` is True, first
        list containing per-episode return and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = (
        is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
    )

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    episode_reward_map = dict(
        zip(
            range(env.unwrapped.envs[0].spec.max_episode_steps),
            [
                np.zeros(n_eval_episodes)
                for _ in range(env.unwrapped.envs[0].spec.max_episode_steps)
            ],
        )
    )

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array(
        [(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int"
    )

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    ep_curric_vals = [0]  # initialise with dummy zero to avoid numpy empty mean warning
    curric_vals = []
    while (episode_counts < episode_count_targets).any():
        curric_config = {
            "curriculum_stages": [],
            "time_step": current_lengths[-1],
            "curriculum_val_ep": ep_curric_vals,
            "env": env,
            "obs": observations,
        }
        _, curric_val = curriculum_fns["action_choice_fn"](curric_config)
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        if current_lengths[0] > 0:
            ep_curric_vals.append(curric_val)
        else:
            # replace the dummy zero
            ep_curric_vals = [curric_val]
        if np.random.random() < randomness:
            actions = [env.action_space.sample()]
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        episode_reward_map[current_lengths[0]][episode_counts[0]] += rewards[0]
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    curric_vals.extend(ep_curric_vals)
                    ep_curric_vals = [0]
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            env.render()
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, (
            "Mean reward below threshold: "
            f"{mean_reward:.2f} < {reward_threshold:.2f}"
        )
    if return_guide_vals:
        if return_episode_rewards:
            if np.all(np.array(curric_vals) == None):
                curric_vals = episode_reward_map
            return episode_rewards, episode_lengths, curric_vals
        guide_val = curriculum_fns["accumulator_fn"](curric_vals)
        return mean_reward, std_reward, guide_val
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward


def evaluate_policy_patch(
    model: "type_aliases.PolicyPredictor",
    action_selector,
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[tuple[float, float], tuple[list[float], list[int]]]:
    """
    Runs the policy for ``n_eval_episodes`` episodes and outputs the average return
    per episode (sum of undiscounted rewards).
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a ``predict`` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to perform additional checks,
        called ``n_envs`` times after each step.
        Gets locals() and globals() passed as parameters.
        See https://github.com/DLR-RM/stable-baselines3/issues/1912 for more details.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean return per episode (sum of rewards), std of reward per episode.
        Returns (list[float], list[int]) when ``return_episode_rewards`` is True, first
        list containing per-episode return and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = (
        is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
    )

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    episode_successes = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array(
        [(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int"
    )
    episode_reward_map = dict(
        zip(
            range(env.unwrapped.envs[0].spec.max_episode_steps),
            [
                np.zeros(n_eval_episodes)
                for _ in range(env.unwrapped.envs[0].spec.max_episode_steps)
            ],
        )
    )
    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    ep_curriculum_values = [0]
    learner_usage_values = []
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        choice_config = {
            "curriculum_stages": model.curriculum_stages,
            "curriculum_stage_idx": model.curriculum_stage_idx,
            "time_step": current_lengths[-1],
            "curriculum_val_ep": ep_curriculum_values,
            "env": env,
            "obs": observations,
            "variance_fn": model.variance_fn,
            "exp_time_step_coeff": model.exp_time_step_coeff,
        }
        use_learner, curriculum_val = model.learner_or_guide_action(choice_config)
        # use_learner = True
        if use_learner and model.horizon_fn == "var_nn_adaptive":
            if action_selector.predict(observations)[0][0] == 0:
                use_learner = False
        if model.curriculum_stage_idx == len(model.curriculum_stages) - 1:
            use_learner = True
        curriculum_val = 0
        if current_lengths[0] > 0:
            ep_curriculum_values.append(curriculum_val)
            ep_learner_usage.append(int(use_learner))
        else:
            # replace the dummy zero
            ep_curriculum_values = [curriculum_val]
            ep_learner_usage = [int(use_learner)]
        if use_learner:
            actions, states = model.predict(
                observations,  # type: ignore[arg-type]
                state=states,
                episode_start=episode_starts,
                deterministic=deterministic,
            )
        else:
            actions, states = model.guide_policy.predict(
                observations,  # type: ignore[arg-type]
                state=states,
                episode_start=episode_starts,
                deterministic=deterministic,
            )
            if np.random.random() < model.guide_randomness:
                actions = [env.action_space.sample()]
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        episode_reward_map[current_lengths[0]][episode_counts[0]] += rewards[0]
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if "success" in info.keys():
                        episode_successes.append(info["success"])
                    elif "is_success" in info.keys():
                        episode_successes.append(info["is_success"])
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    ep_curriculum_values = [0]
                    learner_usage_values.append(np.mean(ep_learner_usage))

        observations = new_observations

        if render:
            env.render()

    mean_learner_usage = np.mean(learner_usage_values)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, (
            "Mean reward below threshold: "
            f"{mean_reward:.2f} < {reward_threshold:.2f}"
        )
    if return_episode_rewards:
        return (
            episode_rewards,
            episode_lengths,
            learner_usage_values,
            episode_reward_map,
            episode_successes,
        )
    return (
        mean_reward,
        std_reward,
        mean_learner_usage,
        episode_reward_map,
        episode_successes,
    )


def train_patch(self, gradient_steps: int, batch_size: int = 64) -> None:
    # Switch to train mode (this affects batch norm / dropout)
    self.policy.set_training_mode(True)
    # Update optimizers learning rate
    optimizers = [
        self.actor.optimizer,
        self.critic.optimizer,
        # self.action_selector.optimizer,
    ]
    if self.ent_coef_optimizer is not None:
        optimizers += [self.ent_coef_optimizer]

    # Update learning rate according to lr schedule
    self._update_learning_rate(optimizers)

    ent_coef_losses, ent_coefs = [], []
    actor_losses, critic_losses, action_selector_losses = [], [], []

    for gradient_step in range(gradient_steps):
        # Sample replay buffer
        replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
        # We need to sample because `log_std` may have changed between two gradient steps
        if self.use_sde:
            self.actor.reset_noise()

        # Action by the current actor for the sampled state
        original_actions_pi, log_prob = self.actor.action_log_prob(
            replay_data.observations
        )
        # print(log_prob[-10:].tolist())
        actions_taken = self.replay_buffer.to_torch(replay_data.actions)
        actions_taken_prob = torch.clip(
            self.actor.action_dist.log_prob(actions_taken), -100, 100
        )

        selected_guide = 1 - replay_data.used_learner
        selected_learner = replay_data.used_learner
        actions_pi = (
            original_actions_pi * selected_learner + actions_taken * selected_guide
        )
        log_prob = log_prob * torch.squeeze(
            selected_learner, 1
        ) + actions_taken_prob * torch.squeeze(selected_guide, 1)
        selected_q_values_log, guide_q_values_log = [], []
        policy_means = []
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

        action_selector, logits = self.action_selector(
            replay_data.observations, return_logits=True
        )
        action_selector = action_selector[:, 0]
        mean_learner_used = action_selector.mean()
        policy_means.append(mean_learner_used.item())
        with torch.no_grad():
            original_q_values = th.min(
                th.cat(
                    self.critic(replay_data.observations, original_actions_pi.detach()),
                    dim=1,
                ),
                dim=1,
                keepdim=True,
            )[0]
        selected_actions_pi = original_actions_pi.detach() * torch.unsqueeze(
            action_selector, 1
        ) + actions_taken.detach() * torch.unsqueeze((1 - action_selector), 1)
        selected_q_values = th.min(
            th.cat(self.critic(replay_data.observations, selected_actions_pi), dim=1),
            dim=1,
            keepdim=True,
        )[0]
        import pdb

        pdb.set_trace()
        # action_selector_losses = (action_selector-(selected_q_values>original_q_values).float()).mean()
        # self.action_selector.optimizer.zero_grad()
        # action_selector_losses.backward()
        # self.action_selector.optimizer.step()

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
        # from torchviz import make_dot
        # make_dot(actor_loss, params=dict(self.actor.named_parameters())).render("action_selector_loss", format="pdf")
        # exit()

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
    # self.logger.record("train/action_selector_loss", np.mean(action_selector_losses))
    # self.logger.record(
    #     "train/q_value_diff",
    #     np.mean(selected_q_values_log) - np.mean(guide_q_values_log),
    # )
    self.logger.record("train/mean_policy", np.mean(policy_means))
    if len(ent_coef_losses) > 0:
        self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


def _sample_action_patch(
    self,
    learning_starts: int,
    action_noise: Optional[ActionNoise] = None,
    n_envs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample an action according to the exploration policy.
    This is either done by sampling the probability distribution of the policy,
    or sampling a random action (from a uniform distribution over the action space)
    or by adding noise to the deterministic output.

    :param action_noise: Action noise that will be used for exploration
        Required for deterministic policy (e.g. TD3). This can also be used
        in addition to the stochastic policy for SAC.
    :param learning_starts: Number of steps before learning for the warm-up phase.
    :param n_envs:
    :return: action to take in the environment
        and scaled action that will be stored in the replay buffer.
        The two differs when the action space is not normalized (bounds are not [-1, 1]).
    """
    # Select action randomly or according to policy
    choice_config = {
        "curriculum_stages": self.curriculum_stages,
        "curriculum_stage_idx": self.curriculum_stage_idx,
        "time_step": self.ep_timestep,
        "curriculum_val_ep": self.ep_curriculum_values,
        "env": self.get_env(),
        "obs": self._last_obs,
        "variance_fn": self.variance_fn,
        "exp_time_step_coeff": self.exp_time_step_coeff,
    }
    if self.num_timesteps < learning_starts and not (
        self.use_sde and self.use_sde_at_warmup
    ):
        # Warmup phase

        use_learner, self.curriculum_val_t = self.learner_or_guide_action(choice_config)
    else:
        # Note: when using continuous actions,
        # we assume that the policy uses tanh to scale the action
        # We use non-deterministic action in the case of SAC, for TD3, it does not matter
        assert self._last_obs is not None, "self._last_obs was not set"
        if self.curriculum_stage_idx == len(self.curriculum_stages) - 1:
            use_learner = True
        else:
            use_learner, self.curriculum_val_t = self.learner_or_guide_action(
                choice_config
            )
        # use_learner = True

    if np.random.random() < self.guide_randomness:
        guide_act = np.array([self.action_space.sample() for _ in range(n_envs)])
    else:
        guide_act, _ = self.guide_policy.predict(self._last_obs, deterministic=True)

    if use_learner:
        unscaled_action, _ = self.predict(self._last_obs, deterministic=False)
        if self.horizon_fn == "var_nn_adaptive":
            selector = self.action_selector.predict(self._last_obs)
            if selector[0][0] == 0:
                use_learner = False
            # unscaled_action = np.sum(
            #    np.stack((unscaled_action, guide_act), axis=-1) * selector, axis=-1
            # )
    else:
        unscaled_action = guide_act
    # Rescale the action from [low, high] to [-1, 1]
    if isinstance(self.action_space, spaces.Box):
        scaled_action = self.policy.scale_action(unscaled_action)

        # Add noise to the action (improve exploration)
        if action_noise is not None:
            scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

        # We store the scaled action in the buffer
        buffer_action = scaled_action
        action = self.policy.unscale_action(scaled_action)
    else:
        # Discrete case, no need to normalize or clip
        buffer_action = unscaled_action
        action = buffer_action
    self.last_use_learner = use_learner
    if not self.guide_in_buffer:
        return action, buffer_action, use_learner
    else:
        return action, buffer_action


def run_grl_training(config, seed):
    algo = config["algo"]
    env_name = config["env_name"]
    pretrained_path = config["pretrained_path"]
    if config["grl_config"]["n_curriculum_stages"] == 0:
        assert (
            config["grl_config"]["horizon_fn"] == "agent_type"
        ), "If n_curriculum_stages=0 (run guide only), the horizon function must be agent_type"
    config["seed"] = seed
    import curriculum_action_choice_utils as cacu

    cacu.SAMPLE_PERC = config["grl_config"]["sample_perc"]

    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    if "train_freq" in config:
        config["gradient_steps"] = config["train_freq"]
    # Evaluate the guide policy
    guide_policy = MODELS[algo].load(pretrained_path)
    _, _, guide_curric_vals = evaluate_imperfect_policy(
        guide_policy,
        eval_env,
        return_guide_vals=True,
        return_episode_rewards=True,
        n_eval_episodes=config["pretrain_eval_episodes"],
        randomness=(
            config["grl_config"]["guide_randomness"]
            + (1 / config["grl_config"]["n_curriculum_stages"])
        ),
        curriculum_fns=CURRICULUM_FNS[config["grl_config"]["horizon_fn"]],
    )
    guide_return, guide_var, _ = evaluate_imperfect_policy(
        guide_policy,
        eval_env,
        return_guide_vals=True,
        return_episode_rewards=True,
        n_eval_episodes=config["pretrain_eval_episodes"],
        randomness=(
            config["grl_config"]["guide_randomness"]
            + (1 / config["grl_config"]["n_curriculum_stages"])
        ),
        curriculum_fns=CURRICULUM_FNS[config["grl_config"]["horizon_fn"]],
    )
    # print(f"Guide return: {np.mean(guide_return)}+\-{np.mean(guide_std)}")

    # Patch the algo with modified functions
    SAC._sample_action = _sample_action_patch
    sb3_eval.evaluate_policy = evaluate_policy_patch
    # SAC._store_transition = _store_transition_patch

    # Set up the model callbacks
    if config["debug"]:
        project = "sb3-sac-curricula_debug"
    else:
        project = "sb3-sac-curricula_all_envs"
    run = wandb.init(
        entity="lauren-taylor-the-university-of-adelaide",
        project=project,
        sync_tensorboard=True,
        monitor_gym=True,
        config=config,
        save_code=False,
    )
    wandb_cb = WandbCallback(
        gradient_save_freq=10000,
        # model_save_path=f"saved_models/{env_name}_{algo}_{config['seed']}",
        verbose=2,
    )

    curriculum_mgmt_cb = CurriculumMgmtCallback(
        guide_policy, np.mean(guide_return), guide_curric_vals, config["grl_config"]
    )
    curriculum_update_cb = CurriculumStageUpdateCallback(
        config["grl_config"]["horizon_fn"]
    )
    eval_cb = ModifiedEvalCallback(
        eval_env,
        best_model_save_path=f"./saved_models/{env_name}_{algo}_{config['seed']}",
        log_path=f"./saved_models/{env_name}_{algo}_{config['seed']}",
        eval_freq=config["eval_freq"],
        n_eval_episodes=config["n_eval_episodes"],
        deterministic=True,
        render=False,
        callback_after_eval=curriculum_update_cb,
    )

    # Create the model
    if config["grl_config"]["horizon_fn"] == "var_nn_adaptive":
        SAC.policy_aliases["MlpPolicy"] = SACPolicyPatch
        SAC.train = train_patch
    model = SAC(
        "MlpPolicy",
        env,
        stats_window_size=200,
        **config["algo_config"],
        tensorboard_log=f"./saved_models/{env_name}_{algo}_{config['seed']}",
        verbose=1,
        replay_buffer_class=GRLReplayBuffer,
    )
    if config["grl_config"]["horizon_fn"] == "var_nn_adaptive":
        model.action_selector = model.policy.action_selector
    else:
        model.action_selector = None

    # Train
    model.learn(
        total_timesteps=config["training_steps"],
        callback=[wandb_cb, curriculum_mgmt_cb, eval_cb],
        log_interval=5,
    )

    run.finish()


@ray.remote(num_gpus=0)
def ray_grl_training(config, seed):
    run_grl_training(config, seed)


def hyperparam_training(hyperparam_config):
    hyperparam_config["eval_freq"] = tune.choice([5000, 10000, 20000])
    hyperparam_config["n_eval_episodes"] = tune.choice([100, 250, 500])
    hyperparam_config["grl_config"]["n_curriculum_stages"] = tune.choice([15, 20, 25])
    hyperparam_config["grl_config"]["tolerance"] = tune.uniform(0.05, 0.2)

    # hyperparam_config["algo_config"]["buffer_size"] = tune.choice([100000, 1000000])
    # hyperparam_config["learning_rate"] = tune.loguniform(1e-7, 1e-5)
    # hyperparam_config["tau"] = tune.loguniform(1e-4, 1e-2)
    # hyperparam_config["train_freq"] = tune.choice([32, 64, 128])

    tuner = tune.Tuner(
        run_grl_training,
        tune_config=tune.TuneConfig(
            num_samples=1,
            scheduler=ASHAScheduler(
                time_attr="training_iteration",
                grace_period=5,
                metric="eval_return",
                mode="max",
            ),
        ),
        param_space=hyperparam_config,
        run_config=tune.RunConfig(
            storage_path=Path("./hyperparam_results").resolve(), name="tuning"
        ),
    )
    tuner.fit()
