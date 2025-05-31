from curriculum_utils import CURRICULUM_FNS
from collections import deque
from typing import Union, Optional
from ray import tune
import os
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization
import numpy as np
import grl_utils


class ModifiedEvalCallback(EvalCallback):
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(
            eval_env,
            callback_on_new_best,
            callback_after_eval,
            n_eval_episodes,
            eval_freq,
            log_path,
            best_model_save_path,
            deterministic,
            render,
            verbose,
            warn,
        )

    def _on_step(self) -> bool:
        if not hasattr(self, "rolling_n_returns"):
            self.rolling_n_returns = deque(maxlen=self.model.rolling_mean_n)
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []
            episode_rewards, episode_lengths, learner_usage, episode_successes = (
                grl_utils.evaluate_policy_patch(
                    self.model,
                    self.eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    render=self.render,
                    deterministic=self.deterministic,
                    return_episode_rewards=True,
                    warn=self.warn,
                    callback=self._log_success_callback,
                )
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,  # type: ignore[arg-type]
                )

            mean_learner_usage = np.mean(learner_usage)
            np.std(learner_usage)
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(
                episode_lengths
            )
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_learner_usage", float(mean_learner_usage))
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            self.logger.record("eval/success_rate", np.mean(episode_successes))

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(
                "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
            )
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(
                        os.path.join(self.best_model_save_path, "best_model")
                    )
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            self.rolling_n_returns.append(mean_reward)
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


class CurriculumMgmtCallback(BaseCallback):
    """
    Custom callback that sets the initial curriculum stage index and rolling mean.
    """

    def __init__(
        self,
        guide_policy,
        guide_return,
        guide_curriculum_val,
        curriculum_config,
        verbose=0,
    ):
        super().__init__(verbose)
        self.guide_policy = guide_policy
        self.guide_return = guide_return
        self.guide_curriculum_val = guide_curriculum_val
        self.curriculum_config = curriculum_config

    def _on_training_start(self) -> None:
        self.model.guide_policy = self.guide_policy
        self.model.guide_return = self.guide_return
        self.model.guide_randomness = self.curriculum_config["guide_randomness"]
        self.model.rolling_mean_n = self.curriculum_config["rolling_mean_n"]
        self.model.tolerance = self.curriculum_config["tolerance"]
        self.model.guide_curriculum_val = self.guide_curriculum_val
        self.model.guide_in_buffer = self.curriculum_config["guide_in_buffer"]
        self.model.horizon_fn = self.curriculum_config["horizon_fn"]
        self.model.learner_or_guide_action = CURRICULUM_FNS[
            self.curriculum_config["horizon_fn"]
        ]["action_choice_fn"]
        if self.curriculum_config["horizon_fn"] == "exp_time_step":
            self.model.exp_time_step_coeff = self.curriculum_config[
                "exp_time_step_coeff"
            ]
        else:
            self.model.exp_time_step_coeff = None
        if self.curriculum_config["n_curriculum_stages"] > 0:
            self.model.curriculum_stages = CURRICULUM_FNS[
                self.curriculum_config["horizon_fn"]
            ]["generate_curriculum_fn"](
                self.guide_curriculum_val, self.curriculum_config["n_curriculum_stages"]
            )
        else:
            self.model.curriculum_stages = []
        self.model.variance_fn = self.curriculum_config["variance_fn"]
        self.model.curriculum_val_t = 0.0
        self.model.curriculum_stage_idx = 0
        self.model.ep_curriculum_values = [self.model.curriculum_val_t]
        self.model.ep_timestep = 0

    def _on_step(self) -> bool:
        done = self.locals["dones"][-1]
        if done:
            self.logger.record(
                "train/ep_curriculum_val", np.mean(self.model.ep_curriculum_values)
            )
            self.model.ep_timestep = 0
            self.model.ep_curriculum_values = [self.model.curriculum_val_t]
        elif self.model.ep_timestep == 0:
            # replace dummy 0 with actual
            self.model.ep_timestep += 1
            self.model.ep_curriculum_values = [self.model.curriculum_val_t]
        else:
            self.model.ep_timestep += 1
            self.model.ep_curriculum_values.append(self.model.curriculum_val_t)
        self.locals["infos"][-1]["last_use_learner"] = self.model.last_use_learner
        return True


class CurriculumStageUpdateCallback(BaseCallback):
    """
    Custom callback that decides whether to use the guide or learner agent at each time step.
    """

    parent: EvalCallback

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if not hasattr(self, "best_eval_return"):
            self.best_eval_return = self.parent.model.guide_return

        prev_best = (
            self.best_eval_return - self.parent.model.tolerance * self.best_eval_return
        )
        if len(self.parent.rolling_n_returns) == self.parent.model.rolling_mean_n:
            tune.report({"eval_return": self.parent.last_mean_reward})
            if np.mean(
                self.parent.rolling_n_returns
            ) > prev_best and self.parent.model.curriculum_stage_idx < (
                len(self.parent.model.curriculum_stages) - 1
            ):
                self.parent.model.curriculum_stage_idx += 1
                if self.parent.rolling_n_returns[-1] > self.best_eval_return:
                    self.best_eval_return = self.parent.rolling_n_returns[-1]
            self.parent.logger.record(
                "eval/eval_rolling_mean", np.mean(self.parent.rolling_n_returns)
            )
        try:
            current_stage = self.parent.model.curriculum_stages[
                self.parent.model.curriculum_stage_idx
            ]
        except IndexError:
            current_stage = 0
        print(
            f"Best Return: {self.best_eval_return}, Latest Return: {self.parent.last_mean_reward}, Current Stage Idx: {self.parent.model.curriculum_stage_idx}/{len(self.parent.model.curriculum_stages)}, Current Stage: {current_stage}"
        )

        self.parent.logger.record(
            "eval/curriculum_stage_idx", self.model.curriculum_stage_idx
        )
        self.parent.logger.record(
            "eval/curriculum_stage",
            self.parent.model.curriculum_stages[self.model.curriculum_stage_idx],
        )
        self.parent.logger.record("eval/best_eval_w_tolerance", prev_best)
        return True
