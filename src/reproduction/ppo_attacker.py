"""PPO actor-critic for the stealthy-attack generator (Sec. IV-B, Table I, Fig. 2).

Fig. 2 depicts an LSTM-based policy network taking the per-session state and
producing (mu, sigma) for a Gaussian policy over the scalar attack coefficient
C_attack, plus a value head. We implement exactly that: an LSTM processes the
sequence of sessions in a compromised-EVCP rollout (temporal order = order of
charging requests), consistent with the paper's single input diagram box
labeled "LSTM" feeding both the policy and (implicitly) the value estimate.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from .config import AttackerConfig


class ActorCritic(nn.Module):
    def __init__(self, cfg: AttackerConfig):
        super().__init__()
        self.lstm = nn.LSTM(cfg.state_dim, cfg.lstm_hidden, batch_first=True)
        self.pi_mu = nn.Sequential(nn.Linear(cfg.lstm_hidden, cfg.mlp_hidden), nn.ReLU(), nn.Linear(cfg.mlp_hidden, cfg.action_dim))
        self.pi_log_std = nn.Parameter(torch.zeros(cfg.action_dim))
        self.value = nn.Sequential(nn.Linear(cfg.lstm_hidden, cfg.mlp_hidden), nn.ReLU(), nn.Linear(cfg.mlp_hidden, 1))

    def forward(self, state_seq: torch.Tensor, hidden=None):
        # state_seq: (B, T, state_dim)
        feats, hidden = self.lstm(state_seq, hidden)
        mu = self.pi_mu(feats)
        std = torch.exp(self.pi_log_std).expand_as(mu)
        v = self.value(feats).squeeze(-1)
        return mu, std, v, hidden

    def act(self, state_seq: torch.Tensor, hidden=None):
        mu, std, v, hidden = self.forward(state_seq, hidden)
        dist = Normal(mu, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp, v, hidden


@dataclass
class Rollout:
    states: list
    actions: list
    logps: list
    rewards: list
    values: list
    dones: list


def compute_gae(rewards, values, dones, gamma, lam):
    """Standard GAE-lambda advantage estimate (PPO training-loop mechanic not
    numerically specified by the paper beyond gamma=0.99 -- see config.py)."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    next_value = 0.0
    for t in reversed(range(T)):
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        adv[t] = last_gae
        next_value = values[t]
    returns = adv + np.array(values, dtype=np.float32)
    return adv, returns


def ppo_update(model: ActorCritic, optimizer, rollout: Rollout, cfg: AttackerConfig, device):
    states = torch.tensor(np.array(rollout.states), dtype=torch.float32, device=device).unsqueeze(0)  # (1,T,3)
    actions = torch.tensor(np.array(rollout.actions), dtype=torch.float32, device=device).unsqueeze(0)
    old_logps = torch.tensor(np.array(rollout.logps), dtype=torch.float32, device=device)
    values = np.array(rollout.values, dtype=np.float32)
    adv, returns = compute_gae(rollout.rewards, values, rollout.dones, cfg.gamma, cfg.gae_lambda)
    adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

    T = states.shape[1]
    idx_all = np.arange(T)
    total_loss_val = 0.0
    for _ in range(cfg.ppo_epochs_per_update):
        np.random.shuffle(idx_all)
        for start in range(0, T, cfg.minibatch_size):
            batch_idx = idx_all[start : start + cfg.minibatch_size]
            mu, std, v, _ = model(states)
            mu_b, std_b, v_b = mu[0, batch_idx], std[0, batch_idx], v[0, batch_idx]
            dist = Normal(mu_b, std_b)
            new_logp = dist.log_prob(actions[0, batch_idx]).sum(-1)
            entropy = dist.entropy().sum(-1).mean()

            ratio = torch.exp(new_logp - old_logps[batch_idx])
            surr1 = ratio * adv_t[batch_idx]
            surr2 = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * adv_t[batch_idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(v_b, returns_t[batch_idx])
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            total_loss_val += loss.item()
    return total_loss_val
