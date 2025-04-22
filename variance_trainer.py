from typing import Callable
import torch
from torch import nn
import numpy as np
import gymnasium as gym
from typing import Tuple
from matplotlib import pyplot as plt
import os

GAMMA = 0.99


def to_tensor(data: np.ndarray) -> torch.Tensor:
    return torch.tensor(data, dtype=torch.float32)


@torch.no_grad()
def run_episodes(
    env: gym.Env, seeds, actor=None, batch_size=None, random_frac=0
) -> Tuple[np.ndarray, np.ndarray]:
    states = []
    next_states = []
    next_dones = []
    actions = []
    rewards = []
    dones = []

    if isinstance(seeds, int):
        cond = lambda x: True
        ep = seeds
    else:
        cond = lambda x: x < len(seeds)
        ep = seeds[0]
    while cond(ep):
        if isinstance(seeds, list) or ep == seeds:
            try:
                env.seed(ep)
                next_state = env.reset()
            except AttributeError:
                next_state, _ = env.reset(seed=ep)
                env.action_space.seed(ep)
        else:
            next_state = env.reset()
            if isinstance(next_state, tuple):
                next_state, _ = next_state
        ts = 0
        next_done = False
        while not next_done:
            states.append(to_tensor(next_state))
            dones.append(next_done)
            if actor is None or np.random.random() <= random_frac:
                action = env.action_space.sample()
            else:
                try:
                    action = actor(env, next_state)
                except TypeError:
                    try:
                        action = actor.predict(next_state, "cpu")
                    except RuntimeError:
                        action = actor.predict(next_state, "cuda")
                    action = action[0]
            next_state, reward, term, trunc, _ = env.step(action)
            if "antmaze" in env.unwrapped.spec.id:
                reward -= 1
            next_done = term or trunc
            actions.append(action)
            rewards.append(reward)
            next_states.append(to_tensor(next_state))
            next_dones.append(next_done)
            if batch_size is not None and len(states) == batch_size:
                return states, actions, rewards, dones, next_states, next_dones
            ts += 1
        ep += 1
    return states, actions, rewards, dones, next_states, next_dones


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def nll_loss(
    pred: torch.Tensor, target: torch.Tensor, var: torch.Tensor
) -> torch.Tensor:
    loss = torch.nn.functional.gaussian_nll_loss(pred, target, var)
    return loss


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mean_sq_err = 0.5 * ((pred - target) ** 2).mean()
    return mean_sq_err


class StateActionVarianceLearner:
    def __init__(self, state_dim, action_dim, random_action, actor, batch_size=256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.random_action = random_action
        self.batch_size = batch_size
        self.actor = actor

        self.mf = StateDepFunction(state_dim)
        self.vf = StateDepFunction(state_dim)

        self.mqf = StateDepQFunction(state_dim)
        self.vqf = StateDepQFunction(state_dim)

        self.m_optimizer = torch.optim.Adam(self.mf.parameters(), lr=0.0001)
        self.mv_optimizer = torch.optim.Adam(self.vf.parameters(), lr=0.0001)

        self.mq_optimizer = torch.optim.Adam(self.mqf.parameters(), lr=0.0001)
        self.mvq_optimizer = torch.optim.Adam(self.vqf.parameters(), lr=0.0001)

    def get_values(self, obs, actions, rewards, next_obs, dones, next_dones):
        batch_size = len(obs)
        values_samp = torch.zeros(batch_size)
        values_pred = torch.zeros(batch_size)
        variance_pred = torch.zeros(batch_size)
        q_values_pred = torch.zeros(batch_size)
        q_variance_pred = torch.zeros(batch_size)
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                nextnonterminal = 1.0 - next_dones[t]
                next_val = self.mf(next_obs[t])
            else:
                nextnonterminal = 1.0 - next_dones[t]
                next_val = values_samp[t + 1]
            values_pred[t], variance_pred[t] = self.mf(obs[t]), self.vf(obs[t])
            q_values_pred[t], q_variance_pred[t] = self.mqf(
                np.concatenate(obs[t], actions[t])
            ), self.vqf(np.concatenate(obs[t], actions[t]))
            variance_pred[t] = torch.clip(torch.exp(variance_pred[t]), 1e-4, 100000000)
            values_samp[t] = rewards[t - 1] + GAMMA * next_val * nextnonterminal
        return values_samp, values_pred, variance_pred, q_values_pred, q_variance_pred

    def _update_v(self, batch, update_vf) -> torch.Tensor:
        # Update value function
        (observations, actions, rewards, dones, next_observations, next_dones) = batch

        log_dict = {}
        values_samp, values_pred, variance_pred, q_values_pred, q_variance_pred = (
            self.get_values(
                observations, actions, rewards, next_observations, dones, next_dones
            )
        )
        v_loss = nll_loss(values_pred, values_samp, variance_pred)
        q_loss = nll_loss(q_values_pred, values_samp, q_variance_pred)

        if update_vf:
            self.mv_optimizer.zero_grad()
            v_loss.backward()
            self.mv_optimizer.step()

            self.mvq_optimizer.zero_grad()
            q_loss.backward()
            self.mvq_optimizer.step()

        else:
            self.m_optimizer.zero_grad()
            v_loss.backward()
            self.m_optimizer.step()

            self.mq_optimizer.zero_grad()
            q_loss.backward()
            self.mq_optimizer.step()

        log_dict["variance_pred_vec"] = variance_pred[:5]
        log_dict["value_loss"] = v_loss.item()
        log_dict["values_pred_vec"] = values_pred[:5]
        log_dict["values_samp_vec"] = values_samp[:5]

        return log_dict

    def run_training(self, env, n_updates=10000, evaluate=False):
        vf_losses = []
        log_freq = 100
        for n in range(n_updates):
            batch = run_episodes(
                env,
                n,
                self.actor,
                batch_size=self.batch_size,
                random_frac=self.random_action,
            )
            log_dict = self._update_v(batch, update_vf=(n > n_updates / 2))
            if n % log_freq == 0:
                print(f"Iteration {n}/{n_updates}:")
                for k, v in log_dict.items():
                    if "loss" in k:
                        vf_losses.append(v)
                    print(f"{k}: {v}")
        log_dict["var_learner/loss"] = vf_losses
        self.mf.eval()
        self.vf.eval()

        save_path = f"{os.getcwd()}/variance_fns"
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if evaluate:
            plt.plot(vf_losses)
            plt.savefig(f"{save_path}/losses_vf_{env.unwrapped.spec.id}.png")
            self.test_model(env, self.actor)
        torch.save(
            self.vf.state_dict(),
            f"{save_path}/{env.unwrapped.spec.id}_{n_updates}_vf.pt",
        )
        torch.save(
            self.mf.state_dict(),
            f"{save_path}/{env.unwrapped.spec.id}_{n_updates}_mf.pt",
        )
        del self.mf
        return self.vf

    def test_model(self, env):
        states, rewards, dones, next_states, next_dones = run_episodes(
            env, actor=self.actor, seeds=[0], eval=True, random_frac=0
        )
        values_samp, values_pred, variance_pred = self.get_values(
            states, rewards, next_states, dones, next_dones
        )
        stds_pred = torch.sqrt(variance_pred)

        state_keys, val_y_samp, val_y_pred, std_y_pred = {}, {}, {}, {}
        state_key = 0
        for i, state in enumerate(states):
            if state in state_keys:
                state_key = state_keys[state]
                val_y_samp[state_key].append(values_samp[i].detach().numpy().item())
                val_y_pred[state_key].append(values_pred[i].detach().numpy().item())
                std_y_pred[state_key].append(stds_pred[i].detach().numpy().item())
            else:
                state_keys[state] = state_key
                val_y_samp[state_key] = [values_samp[i].detach().numpy().item()]
                val_y_pred[state_key] = [values_pred[i].detach().numpy().item()]
                std_y_pred[state_key] = [stds_pred[i].detach().numpy().item()]
                state_key += 1

        true_y = []
        pred_y = []
        for state_key in state_keys.values():
            for i in range(len(val_y_samp[state_key])):
                true_y.append((state_key, val_y_samp[state_key][i]))
                pred_y.append(
                    (state_key, val_y_pred[state_key][i], std_y_pred[state_key][i])
                )

        true_y = np.array(true_y)
        pred_y = np.array(pred_y)

        np.save(
            f"{os.getcwd()}/variance_fns/true_y_{env.unwrapped.spec.id}.npy",
            true_y,
        )
        np.save(
            f"{os.getcwd()}/variance_fns/pred_y_{env.unwrapped.spec.id}.npy",
            pred_y,
        )


class VarianceLearner:
    def __init__(self, state_dim, action_dim, random_action, actor, batch_size=256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.random_action = random_action
        self.batch_size = batch_size
        self.actor = actor

        self.mf = StateDepFunction(state_dim)
        self.vf = StateDepFunction(state_dim)

        self.m_optimizer = torch.optim.Adam(self.mf.parameters(), lr=0.0005)
        self.mv_optimizer = torch.optim.Adam(self.vf.parameters(), lr=0.0005)

    def get_values(self, obs, rewards, next_obs, dones, next_dones):
        batch_size = len(obs)
        values_samp = torch.zeros(batch_size)
        values_pred = torch.zeros(batch_size)
        variance_pred = torch.zeros(batch_size)
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                nextnonterminal = 1.0 - next_dones[t]
                next_val = self.mf(next_obs[t])
            else:
                nextnonterminal = 1.0 - next_dones[t]
                next_val = values_samp[t + 1]
            values_pred[t], variance_pred[t] = self.mf(obs[t]), self.vf(obs[t])
            variance_pred[t] = torch.clip(torch.exp(variance_pred[t]), 1e-4, 100000000)
            values_samp[t] = rewards[t - 1] + GAMMA * next_val * nextnonterminal
        return values_samp, values_pred, variance_pred

    def _update_v(self, batch, update_vf) -> torch.Tensor:
        # Update value function
        (observations, actions, rewards, dones, next_observations, next_dones) = batch

        log_dict = {}
        values_samp, values_pred, variance_pred = self.get_values(
            observations, rewards, next_observations, dones, next_dones
        )
        v_loss = nll_loss(values_pred, values_samp, variance_pred)

        if update_vf:
            self.mv_optimizer.zero_grad()
            v_loss.backward()
            self.mv_optimizer.step()
        else:
            self.m_optimizer.zero_grad()
            v_loss.backward()
            self.m_optimizer.step()

        log_dict["variance_pred_vec"] = variance_pred[:5]
        log_dict["value_loss"] = v_loss.item()
        log_dict["values_pred_vec"] = values_pred[:5]
        log_dict["values_samp_vec"] = values_samp[:5]

        return log_dict

    def run_training(self, env, n_updates=10000, evaluate=False):
        vf_losses = []
        log_freq = 100
        for n in range(n_updates):
            batch = run_episodes(
                env,
                n,
                self.actor,
                batch_size=self.batch_size,
                random_frac=self.random_action,
            )
            log_dict = self._update_v(batch, update_vf=(n > n_updates / 5))
            if n % log_freq == 0:
                print(f"Iteration {n}/{n_updates}:")
                for k, v in log_dict.items():
                    if "loss" in k:
                        vf_losses.append(v)
                    print(f"{k}: {v}")
        log_dict["var_learner/loss"] = vf_losses
        self.mf.eval()
        self.vf.eval()

        save_path = f"{os.getcwd()}/variance_fns/var_functions"
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if evaluate:
            plt.plot(vf_losses)
            plt.savefig(f"{save_path}/losses_vf_{env.unwrapped.spec.id}.png")
            self.test_model(env)
        torch.save(
            self.vf.state_dict(),
            f"{save_path}/{env.unwrapped.spec.id}_{n_updates}_vf.pt",
        )
        torch.save(
            self.mf.state_dict(),
            f"{save_path}/{env.unwrapped.spec.id}_{n_updates}_mf.pt",
        )
        del self.mf
        return self.vf

    def test_model(self, env):
        states, actions, rewards, dones, next_states, next_dones = run_episodes(
            env, actor=self.actor, seeds=[0], random_frac=0
        )
        values_samp, values_pred, variance_pred = self.get_values(
            states, rewards, next_states, dones, next_dones
        )
        stds_pred = torch.sqrt(variance_pred)

        state_keys, val_y_samp, val_y_pred, std_y_pred = {}, {}, {}, {}
        state_key = 0
        for i, state in enumerate(states):
            if state in state_keys:
                state_key = state_keys[state]
                val_y_samp[state_key].append(values_samp[i].detach().numpy().item())
                val_y_pred[state_key].append(values_pred[i].detach().numpy().item())
                std_y_pred[state_key].append(stds_pred[i].detach().numpy().item())
            else:
                state_keys[state] = state_key
                val_y_samp[state_key] = [values_samp[i].detach().numpy().item()]
                val_y_pred[state_key] = [values_pred[i].detach().numpy().item()]
                std_y_pred[state_key] = [stds_pred[i].detach().numpy().item()]
                state_key += 1

        true_y = []
        pred_y = []
        for state_key in state_keys.values():
            for i in range(len(val_y_samp[state_key])):
                true_y.append((state_key, val_y_samp[state_key][i]))
                pred_y.append(
                    (state_key, val_y_pred[state_key][i], std_y_pred[state_key][i])
                )

        true_y = np.array(true_y)
        pred_y = np.array(pred_y)

        np.save(
            f"{os.getcwd()}/variance_fns/true_y_{env.unwrapped.spec.id}_{str(self.random_action).replace('.','-')}.npy",
            true_y,
        )
        np.save(
            f"{os.getcwd()}/variance_fns/pred_y_{env.unwrapped.spec.id}_{str(self.random_action).replace('.','-')}.npy",
            pred_y,
        )


class StateDepFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class StateDepQFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim + 1, *([hidden_dim] * n_hidden), 1]
        self.q = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.q(state)


if __name__ == "__main__":
    path = f"{os.getcwd()}/variance_fns/"
    env = gym.make("antmaze-umaze-v2")
    randomness = 0.1
    n_updates = 1000
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    try:
        vf = StateDepFunction(state_dim)
        mf = StateDepFunction(state_dim)
        vf.load_state_dict(torch.load(path + "_vf.pt"))
        mf.load_state_dict(torch.load(path + "_mf.pt"))
        variance_learner = VarianceLearner(state_dim, action_dim, None, None)
        variance_learner.vf = vf
        variance_learner.mf = mf

        variance_learner.test_model(env)
    except FileNotFoundError:
        vf = VarianceLearner(state_dim, action_dim, randomness, None).run_training(
            env, n_updates=n_updates, evaluate=True
        )
        # config.vf = vf.eval()
