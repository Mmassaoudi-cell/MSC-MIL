"""Trains ONE held-out 5th PPO attacker generator for Stage 2 Part B's
cross-generator robustness diagnostic (SOURCE_WEAKNESS_ANALYSIS.md sec. A1).

theta_thr=0.30 is deliberately OUTSIDE the {0.10,0.15,0.20,0.25} range used by
the K=4 training generators (train_multi_attackers.py's GEN_CONFIGS) -- this
simulates "an attacker style never seen during training" (looser stealth
budget = larger induced deviations than any training-time generator saw).
seed=99 is also disjoint from the K=4 generators' seeds (0-3) and the primary
reproduction attacker's seed (0).

Reuses train_multi_attackers.py's exact per-generator training logic
(train_one_generator) -- same cheap budget (train_iters=50, rollout_len=128)
as the K=4 generators, for direct comparability (this is a 5th generator of
the SAME cheap "generator diversity" family, not a differently-resourced
attacker). Only the save path differs: results/checkpoints/ppo_attacker_heldout.pt
instead of ppo_attacker_gen{k}.pt.
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

HELDOUT_CONFIG = {"theta_thr": 0.30, "c_adjust": 1.0, "seed": 99}
TRAIN_ITERS = 50
ROLLOUT_LEN = 128


def train_heldout(sessions: list[Session], device) -> dict:
    acfg = AttackerConfig()
    acfg.theta_thr = HELDOUT_CONFIG["theta_thr"]
    acfg.c_adjust = HELDOUT_CONFIG["c_adjust"]
    acfg.seed = HELDOUT_CONFIG["seed"]
    acfg.train_iters = TRAIN_ITERS
    acfg.rollout_len = ROLLOUT_LEN

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
        if it % 10 == 0 or it == acfg.train_iters - 1:
            print(f"[train_heldout_attacker] iter {it}/{acfg.train_iters} avg_reward={reward_history[-1]:.4f}")

    ckpt_path = CKPT_DIR / "ppo_attacker_heldout.pt"
    torch.save({"model_state": model.state_dict(), "config": acfg.__dict__}, ckpt_path)
    tail = reward_history[-10:]
    result = {
        "theta_thr": acfg.theta_thr, "c_adjust": acfg.c_adjust, "seed": acfg.seed,
        "train_iters": acfg.train_iters, "rollout_len": acfg.rollout_len,
        "final_mean_reward_last10": float(np.mean(tail)),
        "ckpt_path": str(ckpt_path),
    }
    print(f"[train_heldout_attacker] done. final_reward(last10)={result['final_mean_reward_last10']:.4f} -> {ckpt_path}")
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_heldout_attacker] device={device} config={HELDOUT_CONFIG}")
    manifest = load_manifest()
    sessions = build_train_sessions(manifest)
    print(f"[train_heldout_attacker] {len(sessions)} TRAIN sessions available for PPO rollouts")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    result = train_heldout(sessions, device)

    log_path = CKPT_DIR / "heldout_attacker_training_log.json"
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[train_heldout_attacker] saved log -> {log_path}")


if __name__ == "__main__":
    main()
