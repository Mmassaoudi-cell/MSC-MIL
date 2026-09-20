"""Trains K=4 small, differently-configured PPO attacker generators for
Candidate 1 (MGAT-MIL) and Candidate 4 (GRMA-MIL)'s multi-generator training
pool (MODEL_CANDIDATES.md Candidate 1: "attacks are drawn from K=4
independently-configured PPO generators (varying theta_thr, c_adjust, and
random seed)").

Reuses src/reproduction/train_attacker.py's exact PPO loop logic (ActorCritic,
Rollout, ppo_update, SCMAAttackEnv under SCMA-1 dynamics, per that module's
own documented rationale for training the attacker under SCMA-1 dynamics
regardless of which SCMA type its C_attack is later applied to) -- just
called K times with varied AttackerConfig fields and a much smaller
train_iters/rollout_len budget, since this is generator-diversity plumbing
for a Stage-1 SMOKE TEST, not the main 200-iteration reproduction attacker.

Generator configs (documented choice -- MODEL_CANDIDATES.md specifies WHICH
fields to vary, not the exact values):
  gen0: theta_thr=0.10, seed=0   (tighter stealth budget)
  gen1: theta_thr=0.15, seed=1   (matches the main reproduction's theta_thr)
  gen2: theta_thr=0.20, seed=2   (looser stealth budget)
  gen3: theta_thr=0.25, seed=3, c_adjust=1.2  (loosest budget + adjusted gain)
c_adjust is held at 1.0 for gen0-2 and varied only for gen3 -- enough
generator diversity for a smoke test without a full grid search over both
knobs (which is out of scope at Stage 1).

Cheap training budget: train_iters=50 (vs. the main attacker's 200),
rollout_len=128 (vs. 256) -- a fraction of the main attacker's compute,
consistent with the task brief's "keep these cheap" instruction.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from src.reproduction.config import AttackerConfig
from src.reproduction.dataset import PROJECT_ROOT, load_manifest
from src.reproduction.ppo_attacker import ActorCritic, Rollout, ppo_update
from src.reproduction.scma_env import SCMAAttackEnv, Session
from src.reproduction.train_attacker import build_train_sessions

CKPT_DIR = PROJECT_ROOT / "results" / "checkpoints"

GEN_CONFIGS = [
    {"theta_thr": 0.10, "c_adjust": 1.0, "seed": 0},
    {"theta_thr": 0.15, "c_adjust": 1.0, "seed": 1},
    {"theta_thr": 0.20, "c_adjust": 1.0, "seed": 2},
    {"theta_thr": 0.25, "c_adjust": 1.2, "seed": 3},
]

SMOKE_TRAIN_ITERS = 50
SMOKE_ROLLOUT_LEN = 128


def train_one_generator(gen_id: int, overrides: dict, sessions: list[Session], device) -> dict:
    acfg = AttackerConfig()
    acfg.theta_thr = overrides["theta_thr"]
    acfg.c_adjust = overrides["c_adjust"]
    acfg.seed = overrides["seed"]
    acfg.train_iters = SMOKE_TRAIN_ITERS
    acfg.rollout_len = SMOKE_ROLLOUT_LEN

    torch.manual_seed(acfg.seed)
    np.random.seed(acfg.seed)

    model = ActorCritic(acfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=acfg.lr)

    pool_n = len(sessions)
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
        ppo_update(model, optimizer, rollout, acfg, device)
        reward_history.append(float(np.mean(rollout.rewards)))

    ckpt_path = CKPT_DIR / f"ppo_attacker_gen{gen_id}.pt"
    torch.save({"model_state": model.state_dict(), "config": acfg.__dict__}, ckpt_path)
    tail = reward_history[-10:]
    result = {
        "gen_id": gen_id,
        "theta_thr": acfg.theta_thr,
        "c_adjust": acfg.c_adjust,
        "seed": acfg.seed,
        "train_iters": acfg.train_iters,
        "rollout_len": acfg.rollout_len,
        "final_mean_reward_last10": float(np.mean(tail)),
        "ckpt_path": str(ckpt_path),
    }
    print(
        f"[train_multi_attackers] gen{gen_id} theta_thr={acfg.theta_thr} c_adjust={acfg.c_adjust} "
        f"seed={acfg.seed} final_reward(last10)={result['final_mean_reward_last10']:.4f} -> {ckpt_path}"
    )
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_multi_attackers] device={device}  K={len(GEN_CONFIGS)} generators")

    manifest = load_manifest()
    sessions = build_train_sessions(manifest)
    print(f"[train_multi_attackers] {len(sessions)} TRAIN sessions available for PPO rollouts")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for gen_id, overrides in enumerate(GEN_CONFIGS):
        results.append(train_one_generator(gen_id, overrides, sessions, device))

    log_path = CKPT_DIR / "multi_attacker_training_log.json"
    with open(log_path, "w") as f:
        json.dump({"generators": results}, f, indent=2)
    print(f"[train_multi_attackers] saved log -> {log_path}")


if __name__ == "__main__":
    main()
