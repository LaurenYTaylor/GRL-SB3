import numpy as np
import torch

import torch as th
from torch.nn import functional as F
from stable_baselines3.common.utils import polyak_update


def train_td3_patch(self, gradient_steps: int, batch_size: int = 100) -> None:
    # Switch to train mode (this affects batch norm / dropout)
    self.policy.set_training_mode(True)

    # Update learning rate according to lr schedule
    self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

    actor_losses, critic_losses = [], []
    guide_q_values, learner_q_values = [], []
    used_learner = []
    guide_q_dict = np.zeros(200, dtype=float)
    learner_q_dict = np.zeros(200, dtype=float)
    for _ in range(gradient_steps):
        self._n_updates += 1
        # Sample replay buffer
        replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

        def make_actions(observations, guide_inds, learner_inds, actions):
            next_actions = torch.empty_like(actions)
            if self.guide_randomness > 0:
                random_max_ind = int(self.guide_randomness * len(guide_inds))
                rand_inds = guide_inds[:random_max_ind]
                next_actions[rand_inds] = torch.tensor(
                    np.array(
                        [self.action_space.sample() for _ in range(len(rand_inds))]
                    ),
                    device=next_actions.device,
                    dtype=next_actions.dtype,
                )
            else:
                random_max_ind = 0

            guide_inds = guide_inds[random_max_ind:]
            next_actions[guide_inds] = torch.tensor(
                self.guide_policy.predict(
                    observations[guide_inds].cpu().detach().numpy(), deterministic=True
                )[0],
                device=next_actions.device,
                dtype=next_actions.dtype,
            )
            next_actions[learner_inds] = self.actor_target(observations[learner_inds])
            return actions

        guide_inds = torch.where(replay_data.used_learner == 0)[0]
        learner_inds = torch.where(replay_data.used_learner == 1)[0]

        with th.no_grad():
            # Select action according to policy and add clipped noise
            noise = replay_data.actions.clone().data.normal_(
                0, self.target_policy_noise
            )
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_actions = make_actions(
                replay_data.next_observations,
                guide_inds,
                learner_inds,
                replay_data.actions,
            )
            next_actions = (next_actions + noise).clamp(-1, 1)

            # actions = make_actions(replay_data.next_observations)
            # next_actions = (actions + noise).clamp(-1, 1)

            # Compute the next Q-values: min over all critics targets
            next_q_values = th.cat(
                self.critic_target(replay_data.next_observations, next_actions), dim=1
            )
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            target_q_values = (
                replay_data.rewards
                + (1 - replay_data.dones) * self.gamma * next_q_values
            )

        # Get current Q-values estimates for each critic network
        current_q_values = self.critic(replay_data.observations, replay_data.actions)

        guide_q_values_mean = current_q_values[0][guide_inds].mean().item()
        learner_q_values_mean = current_q_values[0][learner_inds].mean().item()
        guide_ts = (
            replay_data.time_steps[guide_inds].cpu().numpy().astype("int64").flatten()
        )
        learner_ts = (
            replay_data.time_steps[learner_inds].cpu().numpy().astype("int64").flatten()
        )

        n = len(guide_q_dict)
        sums = np.zeros(n, dtype=float)
        counts = np.zeros(n, dtype=int)
        np.add.at(
            sums, guide_ts, target_q_values[guide_inds].detach().cpu().numpy().flatten()
        )
        np.add.at(counts, guide_ts, 1)
        guide_q_dict = (guide_q_dict + sums) / (1 + counts)

        if len(learner_inds) > 0:
            sums = np.zeros(n, dtype=float)
            counts = np.zeros(n, dtype=int)
            np.add.at(
                sums,
                learner_ts,
                target_q_values[learner_inds].detach().cpu().numpy().flatten(),
            )
            np.add.at(counts, learner_ts, 1)

        slices = 10
        for i in range(slices):
            start_idx = int(200 / slices * i)
            end_idx = int(start_idx + 200 / slices)
            guide_q_values_ts_mean = np.mean(guide_q_dict[start_idx:end_idx])
            learner_q_values_ts_mean = np.mean(learner_q_dict[start_idx:end_idx])

        used_learner.append(replay_data.used_learner.mean().item())
        guide_q_values.append(guide_q_values_mean)
        learner_q_values.append(learner_q_values_mean)
        # Compute critic loss
        critic_loss = sum(
            F.mse_loss(current_q, target_q_values) for current_q in current_q_values
        )
        assert isinstance(critic_loss, th.Tensor)
        critic_losses.append(critic_loss.item())

        # Optimize the critics
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Delayed policy updates
        if self._n_updates % self.policy_delay == 0:
            # Compute actor loss
            # actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
            # actions = make_actions(replay_data.observations)
            actions = replay_data.noiseless_actions
            actor_loss = -self.critic.q1_forward(
                replay_data.observations, actions
            ).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            polyak_update(
                self.critic.parameters(), self.critic_target.parameters(), self.tau
            )
            polyak_update(
                self.actor.parameters(), self.actor_target.parameters(), self.tau
            )
            # Copy running stats, see GH issue #996
            polyak_update(
                self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0
            )
            polyak_update(
                self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0
            )

    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
    if len(actor_losses) > 0:
        self.logger.record("train/actor_loss", np.mean(actor_losses))
    self.logger.record("train/critic_loss", np.mean(critic_losses))
    slices = 10
    for i in range(slices):
        # Record the mean of the guide and learner Q-values for each slice
        start_idx = int(200 / slices * i)
        end_idx = int(start_idx + 200 / slices)
        self.logger.record(
            f"train/g-l_q_values_{i}",
            np.mean(guide_q_dict[start_idx:end_idx])
            - np.mean(learner_q_dict[start_idx:end_idx]),
        )
        # self.logger.record(f"train/learner_q_values_{i}", np.mean(learner_q_dict[start_idx:end_idx]))
    self.logger.record(f"train/guide_q_values_mean", np.mean(guide_q_values))
    self.logger.record(f"train/learner_q_values_mean", np.mean(learner_q_values))
    self.logger.record("train/batch_learner_frac", np.mean(used_learner))


def train_simple_td3_patch(self, gradient_steps: int, batch_size: int = 100) -> None:
    # Switch to train mode (this affects batch norm / dropout)
    self.policy.set_training_mode(True)

    # Update learning rate according to lr schedule
    self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

    actor_losses, critic_losses = [], []
    guide_q_values, learner_q_values = [], []
    used_learner = []
    guide_q_dict = np.zeros(200, dtype=float)
    learner_q_dict = np.zeros(200, dtype=float)
    for _ in range(gradient_steps):

        # Sample replay buffer
        if self.replay_buffer.pos < batch_size // 2 and not self.replay_buffer.full:
            continue
        self._n_updates += 1
        replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

        def make_actions(observations, guide_inds, learner_inds, actions):
            next_actions = torch.empty_like(actions)
            if self.guide_randomness > 0:
                random_max_ind = int(self.guide_randomness * len(guide_inds))
                rand_inds = guide_inds[:random_max_ind]
                next_actions[rand_inds] = torch.tensor(
                    np.array(
                        [self.action_space.sample() for _ in range(len(rand_inds))]
                    ),
                    device=next_actions.device,
                    dtype=next_actions.dtype,
                )
            else:
                random_max_ind = 0

            guide_inds = guide_inds[random_max_ind:]
            next_actions[guide_inds] = torch.tensor(
                self.guide_policy.predict(
                    observations[guide_inds].cpu().detach().numpy(), deterministic=True
                )[0],
                device=next_actions.device,
                dtype=next_actions.dtype,
            )
            next_actions[learner_inds] = self.actor_target(observations[learner_inds])
            return actions

        guide_inds = torch.where(replay_data.used_learner == 0)[0]
        learner_inds = torch.where(replay_data.used_learner == 1)[0]

        with th.no_grad():
            # Select action according to policy and add clipped noise
            noise = replay_data.actions.clone().data.normal_(
                0, self.target_policy_noise
            )
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)

            # next_actions = make_actions(replay_data.next_observations, guide_inds, learner_inds, replay_data.actions)
            next_actions = self.actor_target(replay_data.next_observations)
            next_actions = (next_actions + noise).clamp(-1, 1)

            # Compute the next Q-values: min over all critics targets
            next_q_values = th.cat(
                self.critic_target(replay_data.next_observations, next_actions), dim=1
            )
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            target_q_values = (
                replay_data.rewards
                + (1 - replay_data.dones) * self.gamma * next_q_values
            )
            # target_q_values = replay_data.rewards

        # Get current Q-values estimates for each critic network
        current_q_values = self.critic(replay_data.observations, replay_data.actions)

        # guide_q_values_mean = replay_data.rewards[guide_inds].mean().item()
        # learner_q_values_mean = replay_data.rewards[learner_inds].mean().item()
        guide_q_values_mean = target_q_values[guide_inds].mean().item()
        learner_q_values_mean = target_q_values[learner_inds].mean().item()
        guide_ts = (
            replay_data.time_steps[guide_inds].cpu().numpy().astype("int64").flatten()
        )
        learner_ts = (
            replay_data.time_steps[learner_inds].cpu().numpy().astype("int64").flatten()
        )

        n = len(guide_q_dict)
        sums = np.zeros(n, dtype=float)
        counts = np.zeros(n, dtype=int)
        # np.add.at(sums, guide_ts, replay_data.rewards[guide_inds].detach().cpu().numpy().flatten())
        np.add.at(
            sums, guide_ts, target_q_values[guide_inds].detach().cpu().numpy().flatten()
        )
        np.add.at(counts, guide_ts, 1)
        guide_q_dict = (guide_q_dict + sums) / (1 + counts)

        if len(learner_inds) > 0:
            sums = np.zeros(n, dtype=float)
            counts = np.zeros(n, dtype=int)
            # np.add.at(sums, learner_ts, replay_data.rewards[learner_inds].detach().cpu().numpy().flatten())
            np.add.at(
                sums,
                learner_ts,
                target_q_values[learner_inds].detach().cpu().numpy().flatten(),
            )
            np.add.at(counts, learner_ts, 1)
            learner_q_dict = (learner_q_dict + sums) / (1 + counts)
        obs = torch.tensor(
            [-1.0] * 5,
            device=replay_data.observations.device,
            dtype=replay_data.observations.dtype,
        )
        j = 0
        for i in range(-1, 5):

            if i > -1:
                obs[i] = i
            rel_idxs = torch.argwhere(
                torch.all(replay_data.observations == obs, dim=-1)
            )
            if len(rel_idxs) == 0:
                continue

            learner_acts = torch.mean(
                replay_data.actions[rel_idxs[torch.isin(rel_idxs, learner_inds)[:, 0]]],
                dim=0,
            )
            guide_acts = torch.mean(
                replay_data.actions[rel_idxs[torch.isin(rel_idxs, guide_inds)[:, 0]]],
                dim=0,
            )
            learner_act = torch.argmax(learner_acts)
            self.logger.record(f"train/learner_act_{j}", learner_act.item())
            j += 1
            print(
                f"N LEARNER: {np.sum(self.replay_buffer.used_learner)} Obs: {obs}, Learner acts: {learner_acts}, Guide acts: {guide_acts}"
            )

        used_learner.append(replay_data.used_learner.mean().item())
        guide_q_values.append(guide_q_values_mean)
        learner_q_values.append(learner_q_values_mean)
        # Compute critic loss
        critic_loss = sum(
            F.mse_loss(current_q, target_q_values) for current_q in current_q_values
        )
        assert isinstance(critic_loss, th.Tensor)
        critic_losses.append(critic_loss.item())

        # Optimize the critics
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Delayed policy updates
        if self._n_updates % self.policy_delay == 0:
            # Compute actor loss
            actor_loss = -self.critic.q1_forward(
                replay_data.observations[learner_inds],
                self.actor(replay_data.observations[learner_inds]),
            ).mean()
            # actions = make_actions(replay_data.observations)

            # actions = replay_data.noiseless_actions
            # actor_loss = -self.critic.q1_forward(replay_data.observations, actions).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            polyak_update(
                self.critic.parameters(), self.critic_target.parameters(), self.tau
            )
            polyak_update(
                self.actor.parameters(), self.actor_target.parameters(), self.tau
            )
            # Copy running stats, see GH issue #996
            polyak_update(
                self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0
            )
            polyak_update(
                self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0
            )

    if self._n_updates > 0:
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        slices = len(self.curriculum_stages)
        max_steps = self.get_env().envs[0].spec.max_episode_steps
        for i in range(max_steps):
            # Record the mean of the guide and learner Q-values for each slice
            start_idx = int(max_steps / slices * i)
            end_idx = int(start_idx + max_steps / slices)
            self.logger.record(
                f"train/g-l_q_values_{i}",
                np.mean(guide_q_dict[start_idx:end_idx])
                - np.mean(learner_q_dict[start_idx:end_idx]),
            )
            if (
                self.curriculum_stage_idx == 9
                and (
                    np.mean(guide_q_dict[start_idx:end_idx])
                    - np.mean(learner_q_dict[start_idx:end_idx])
                )
                == 0
            ):
                import pdb

                pdb.set_trace()
            # self.logger.record(f"train/learner_q_values_{i}", np.mean(learner_q_dict[start_idx:end_idx]))

        # self.logger.record(f"train/guide_q_values_mean", np.mean(guide_q_values))
        # self.logger.record(f"train/learner_q_values_mean", np.mean(learner_q_values))
        self.logger.record("train/batch_learner_frac", np.mean(used_learner))


def train_sac_patch(self, gradient_steps: int, batch_size: int = 64) -> None:
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
    actor_losses, critic_losses = [], []
    log_probs_actor = []
    log_probs_all = []
    target_qs = []
    buffer_qs = []
    buffer_adjusted_qs = []
    for gradient_step in range(gradient_steps):
        # Sample replay buffer
        replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
        # We need to sample because `log_std` may have changed between two gradient steps
        if self.use_sde:
            self.actor.reset_noise()

        # Action by the current actor for the sampled state
        try:
            original_actions_pi, original_log_prob = self.actor.action_log_prob(
                replay_data.observations
            )
            log_probs_actor.append(original_log_prob.mean().item())
        except ValueError:
            import pdb

            pdb.set_trace()

        """
        actions_taken = self.replay_buffer.to_torch(replay_data.actions)
        actions_taken_prob = torch.clip(
            self.actor.action_dist.log_prob(actions_taken), -1e4, 1e4
        )

        selected_guide = 1 - replay_data.used_learner
        selected_learner = replay_data.used_learner
        policy_means.append(np.mean(replay_data.used_learner.cpu().numpy()))

        actions_pi = (
            original_actions_pi * selected_learner + actions_taken * selected_guide
        )
        log_prob = original_log_prob * torch.squeeze(
            selected_learner, 1
        ) + actions_taken_prob * torch.squeeze(selected_guide, 1)
        """
        actions_pi = original_actions_pi
        log_prob = original_log_prob
        log_probs_all.append(log_prob.mean().item())
        # print(actions_taken_prob)
        log_prob = log_prob.reshape(-1, 1)

        ent_coef_loss = None
        if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            ent_coef = th.exp(self.log_ent_coef.detach())
            assert isinstance(self.target_entropy, float)
            ent_coef_loss = -(
                self.log_ent_coef * (original_log_prob + self.target_entropy).detach()
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
            # next_guide_actions,_ = self.guide_policy.predict(replay_data.next_observations.cpu(), deterministic=True)
            # next_guide_actions = th.tensor(next_guide_actions, device=next_actions.device)
            # next_guide_log_prob = torch.clip(self.actor.action_dist.log_prob(next_guide_actions), -1e4, 1e4)
            # next_actions = (next_actions * selected_learner + next_guide_actions * selected_guide)
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
            target_qs.append(target_q_values.mean().item())

        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        current_q_values = self.critic(replay_data.observations, replay_data.actions)
        # Compute critic loss
        critic_loss = 0.5 * sum(
            F.mse_loss(current_q, target_q_values) for current_q in current_q_values
        )
        assert isinstance(critic_loss, th.Tensor)  # for type checker
        critic_losses.append(critic_loss.item())  # type: ignore[union-attr]
        buffer_qs.append(
            np.max(
                [current_q_values[0].mean().item(), current_q_values[1].mean().item()]
            )
        )

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
        buffer_adjusted_qs.append(min_qf_pi.mean().item())

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
    # self.logger.record("train/mean_policy", np.mean(policy_means))
    if len(ent_coef_losses) > 0:
        self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
    self.logger.record("train/log_prob_actor", np.mean(log_probs_actor))
    self.logger.record("train/log_prob_all", np.mean(log_probs_all))
    self.logger.record("train/target_qs", np.mean(buffer_qs))
    self.logger.record("train/buffer_qs", np.mean(buffer_qs))
    self.logger.record("train/buffer_adjusted_qs", np.mean(buffer_adjusted_qs))
