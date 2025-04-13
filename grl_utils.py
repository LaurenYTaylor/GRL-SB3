import numpy as np
import wandb
import warnings
from stable_baselines3 import SAC
from stable_baselines3.common.noise import ActionNoise
from wandb.integration.sb3 import WandbCallback

from typing import Optional, Union, Callable, Any
import gymnasium as gym
import gymnasium_robotics
import warnings
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv, VecMonitor, is_vecenv_wrapped
from stable_baselines3.common import evaluation as sb3_eval
from grl_callbacks import CurriculumMgmtCallback, CurriculumStageUpdateCallback, ModifiedEvalCallback

MODELS = {"SAC": SAC}
    
def evaluate_policy_patch(
    model: "type_aliases.PolicyPredictor",
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

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

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

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    ep_curriculum_values = []
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        if model.curriculum_stage_idx == len(model.curriculum_stages)-1:
            use_learner = True
        elif (current_lengths[-1]==0):
            use_learner = False
        else:
            if model.variance_fn is not None:
                variance = model.variance_fn(observations, env)
            else:
                variance = None
            choice_config = {
                "curriculum_stage": model.curriculum_stages[model.curriculum_stage_idx],
                "time_step": current_lengths[-1],
                "curriculum_val_ep": ep_curriculum_values,
                "env": env,
                "obs": observations,
                "variance": variance}
            use_learner, curriculum_val = model.learner_or_guide_action(choice_config)
            ep_curriculum_values.append(curriculum_val)
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
                actions = env.action_space.sample()
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
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
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward

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
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            assert self._last_obs is not None, "self._last_obs was not set"
            if self.curriculum_stage_idx == len(self.curriculum_stages)-1:
                use_learner = True
            elif (self.ep_timestep==0):
                use_learner = False
            else:
                if self.variance_fn is not None:
                    variance = self.variance_fn(self._last_obs, self.get_env())
                else:
                    variance = None
                choice_config = {
                    "curriculum_stage": self.curriculum_stages[self.curriculum_stage_idx],
                    "time_step": self.ep_timestep,
                    "curriculum_val_ep": self.ep_curriculum_values,
                    "env": self.get_env(),
                    "obs": self._last_obs,
                    "variance": variance}
                use_learner, self.curriculum_val_t = self.learner_or_guide_action(choice_config)
            if use_learner:
                unscaled_action, _ = self.predict(self._last_obs, deterministic=False)
                
            else:
                if np.random.random() < self.guide_randomness:
                    unscaled_action = self.action_space.sample() 
                else:
                    unscaled_action, _ = self.guide_policy.predict(self._last_obs, deterministic=True)
        
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
        return action, buffer_action

def run_grl_training(config):
    algo = config["algo"]
    env_name = config["env_name"]
    pretrained_path = config["pretrained_path"]
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    if "train_freq" in config:
        config["gradient_steps"] = config["train_freq"]
    # Evaluate the guide policy
    guide_policy = MODELS[algo].load(pretrained_path)
    guide_return = sb3_eval.evaluate_policy(guide_policy,
    eval_env,
    n_eval_episodes=config["n_eval_episodes"])[0]
    print(f"Guide return: {guide_return}")
    
    # Patch the algo with modified functions
    SAC._sample_action = _sample_action_patch
    sb3_eval.evaluate_policy = evaluate_policy_patch

    # Set up the model callbacks
    run = wandb.init(
            project="sb3-sac-curriculum_hyperparam",
            sync_tensorboard=True,
            monitor_gym=True,
            config={"env_name": env_name, "algorithm": algo, "seed": config["seed"], "training_steps": config["training_steps"]},
            save_code=False,
        )    
    wandb_cb = WandbCallback(gradient_save_freq=10000,model_save_path=f"saved_models/{env_name}_{algo}_{config['seed']}", verbose=2)
    curriculum_mgmt_cb = CurriculumMgmtCallback(guide_policy, guide_return, 1, config["grl_config"])
    curriculum_update_cb = CurriculumStageUpdateCallback()
    eval_cb = ModifiedEvalCallback(
        eval_env,
        best_model_save_path=f"./saved_models/{env_name}_{algo}_{config['seed']}",
        log_path=f"./saved_models/{env_name}_{algo}_{config['seed']}",
        eval_freq=config["eval_freq"],
        n_eval_episodes=config["n_eval_episodes"],
        deterministic=True,
        render=False,
        callback_after_eval=curriculum_update_cb
        )
    
    # Create the model
    model = SAC("MlpPolicy", env,
            **config["algo_config"], tensorboard_log=f"./saved_models/{env_name}_{algo}_{config['seed']}", verbose=1)
    
    # Train
    model.learn(
            total_timesteps=config["training_steps"],
            callback=[wandb_cb, curriculum_mgmt_cb, eval_cb]
        )
    run.finish()
  
    

