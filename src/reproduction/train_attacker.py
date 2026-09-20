"""Trains ONE PPO attacker (Sec. IV-B, Table I, Fig. 2), shared across all 6
SCMA types, on TRAIN split sessions only, seed=0.

Per SOURCE_PAPER_AUDIT.md and the task brief: C_attack is predicted once per
session (Algorithm 1 line 15) independent of which manipulation-type branch
(lines 16-36) is then applied -- so the attacker's MDP is trained under SCMA-1
(demand-only) dynamics, which is what C_attack fundamentally parameterizes via
the shared apply_demand_attack() function reused by all 6 SCMA types
(scma_env.py). This is a documented structural choice, not an approximation:
training under SCMA-1 dynamics and reusing the frozen policy for all 6 types
at dataset-generation time is exactly what Algorithm 1's control flow implies.

"Iterations" (Table I: 200) = PPO policy-update iterations, each consuming one
rollout of `rollout_len` (256, ASSUMPTION) sessions drawn from the TRAIN pool
in chronological order, cycling through the pool (sessions sorted by
connection_time; iteration i uses pool[(i*rollout_len) % N : ...]).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import ReproductionConfig
from .dataset import PROJECT_ROOT, load_manifest
from .ppo_attacker import ActorCritic, Rollout, ppo_update
from .scma_env import SCMAAttackEnv, Session


def build_train_sessions(manifest) -> list[Session]:
    sub = manifest[manifest.split == "train"].sort_values("connection_time")
    sessions = []
    for r in sub.itertuples(index=False):
        duration_h = max((r.disconnection_time - r.connection_time).total_seconds() / 3600.0, 1e-3)
        t_avail = min(duration_h, 24.0)
        sessions.append(Session(t_start_h=0.0, t_end_h=t_avail, e_demand_kwh=float(r.kwh_delivered), t_avail_h=t_avail, dt_min=5.0))
    return sessions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=None, help="override AttackerConfig.train_iters")
    parser.add_argument("--rollout-len", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="tiny run to sanity-check plumbing")
    args = parser.parse_args()

    cfg = ReproductionConfig()
    acfg = cfg.attacker
    if args.iters is not None:
        acfg.train_iters = args.iters
    if args.rollout_len is not None:
        acfg.rollout_len = args.rollout_len
    if args.smoke:
        acfg.train_iters = 5
        acfg.rollout_len = 32

    torch.manual_seed(acfg.seed)
    np.random.seed(acfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_attacker] device={device}")

    manifest = load_manifest()
    sessions = build_train_sessions(manifest)
    pool_n = len(sessions)
    print(f"[train_attacker] {pool_n} TRAIN sessions available for PPO rollouts (SCMA-1 dynamics)")

    model = ActorCritic(acfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=acfg.lr)

    reward_history = []
    for it in range(acfg.train_iters):
        start = (it * acfg.rollout_len) % pool_n
        idxs = [(start + k) % pool_n for k in range(acfg.rollout_len)]
        batch_sessions = [sessions[i] for i in idxs]
        env = SCMAAttackEnv(batch_sessions, scma_type=1, theta_thr=acfg.theta_thr, c_adjust=acfg.c_adjust)
        state = env.reset()
        rollout = Rollout([], [], [], [], [], [])
        hidden = None
        done = False
        while not done:
            state_t = torch.tensor(state, dtype=torch.float32, device=device).view(1, 1, -1)
            with torch.no_grad():
                action, logp, v, hidden = model.act(state_t, hidden)
            a = float(action.item())
            next_state, reward, done, info = env.step(a)
            rollout.states.append(state)
            rollout.actions.append([a])
            rollout.logps.append(logp.item())
            rollout.rewards.append(reward)
            rollout.values.append(v.item())
            rollout.dones.append(float(done))
            state = next_state
        loss = ppo_update(model, optimizer, rollout, acfg, device)
        avg_reward = float(np.mean(rollout.rewards))
        reward_history.append(avg_reward)
        if it % 10 == 0 or it == acfg.train_iters - 1:
            print(f"[train_attacker] iter {it}/{acfg.train_iters}  avg_reward={avg_reward:.4f}  ppo_loss_sum={loss:.4f}")

    ckpt_dir = PROJECT_ROOT / "results" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "ppo_attacker.pt"
    torch.save({"model_state": model.state_dict(), "config": acfg.__dict__}, ckpt_path)
    print(f"[train_attacker] saved checkpoint -> {ckpt_path}")

    fig_dir = PROJECT_ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(reward_history, alpha=0.5, label="per-iteration avg reward")
    window = min(10, max(1, len(reward_history) // 5))
    if len(reward_history) >= window and window > 1:
        ma = np.convolve(reward_history, np.ones(window) / window, mode="valid")
        plt.plot(range(window - 1, len(reward_history)), ma, label=f"{window}-iter moving avg", linewidth=2)
    plt.xlabel("PPO update iteration")
    plt.ylabel("avg episode reward")
    plt.title("SCMA attacker PPO convergence (SCMA-1 dynamics, reshaped reward)")
    plt.legend()
    plt.tight_layout()
    fig_path = fig_dir / "attacker_convergence.png"
    plt.savefig(fig_path, dpi=150)
    print(f"[train_attacker] saved convergence plot -> {fig_path}")

    tail = reward_history[-20:] if len(reward_history) >= 20 else reward_history
    print(f"[train_attacker] final avg reward (last {len(tail)} iters): mean={np.mean(tail):.4f} std={np.std(tail):.4f}")

    log_path = ckpt_dir / "attacker_training_log.json"
    with open(log_path, "w") as f:
        json.dump(
            {
                "reward_history": reward_history,
                "final_mean_reward_last20": float(np.mean(tail)),
                "final_std_reward_last20": float(np.std(tail)),
                "train_iters": acfg.train_iters,
                "rollout_len": acfg.rollout_len,
                "pool_n_sessions": pool_n,
            },
            f,
            indent=2,
        )
    print(f"[train_attacker] saved training log -> {log_path}")


if __name__ == "__main__":
    main()
