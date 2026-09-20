"""Stage 3 Part B, step 1 -- trains a HARDER held-out 5th PPO generator than
Stage 2's (MODEL_SELECTION_REPORT.md Stage 2 Part B flagged this as a real
caveat: that diagnostic used a noisy 50-iteration/128-rollout budget and
theta_thr=0.30, and found essentially zero degradation for ANY model,
inconclusive on whether generator-invariance helps).

This version:
  - REAL training budget matching the main reproduction attacker:
    train_iters=200, rollout_len=256 (AttackerConfig defaults -- see
    src/reproduction/config.py), not the cheap 50/128 budget the K=4 pool and
    Stage 2's held-out generator used.
  - theta_thr=0.35 -- further outside the K=4 training generators' {0.10-0.25}
    range than Stage 2's 0.30 was.
  - seed=123 -- disjoint from the K=4 generators (0-3), the primary attacker
    (0), and Stage 2's held-out generator (99).

Saved to results/checkpoints/ppo_attacker_heldout_hard.pt, log to
results/checkpoints/heldout_attacker_hard_training_log.json (compared against
Stage 2's heldout_attacker_training_log.json final_mean_reward_last10=0.0168
to check whether this budget converges more cleanly, per the task brief).

Reuses train_heldout_attacker.py's exact per-iteration PPO loop logic
(train_heldout), just with different AttackerConfig overrides / save path.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from src.reproduction.config import AttackerConfig
from src.reproduction.dataset import PROJECT_ROOT, load_manifest
from src.reproduction.ppo_attacker import ActorCritic, Rollout, ppo_update
from src.reproduction.scma_env import SCMAAttackEnv
from src.reproduction.train_attacker import build_train_sessions

CKPT_DIR = PROJECT_ROOT / "results" / "checkpoints"

HARD_HELDOUT_CONFIG = {"theta_thr": 0.35, "c_adjust": 1.0, "seed": 123}
TRAIN_ITERS = 200  # matches AttackerConfig default / main reproduction attacker
ROLLOUT_LEN = 256  # matches AttackerConfig default / main reproduction attacker


def train_heldout_hard(sessions, device) -> dict:
    acfg = AttackerConfig()
    acfg.theta_thr = HARD_HELDOUT_CONFIG["theta_thr"]
    acfg.c_adjust = HARD_HELDOUT_CONFIG["c_adjust"]
    acfg.seed = HARD_HELDOUT_CONFIG["seed"]
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
        if it % 20 == 0 or it == acfg.train_iters - 1:
            print(f"[train_heldout_attacker_hard] iter {it}/{acfg.train_iters} avg_reward={reward_history[-1]:.4f}")

    ckpt_path = CKPT_DIR / "ppo_attacker_heldout_hard.pt"
    torch.save({"model_state": model.state_dict(), "config": acfg.__dict__}, ckpt_path)
    tail = reward_history[-10:]
    result = {
        "theta_thr": acfg.theta_thr, "c_adjust": acfg.c_adjust, "seed": acfg.seed,
        "train_iters": acfg.train_iters, "rollout_len": acfg.rollout_len,
        "final_mean_reward_last10": float(np.mean(tail)),
        "final_mean_reward_std_last10": float(np.std(tail)),
        "reward_history": reward_history,
        "ckpt_path": str(ckpt_path),
    }
    print(f"[train_heldout_attacker_hard] done. final_reward(last10)={result['final_mean_reward_last10']:.4f} "
          f"+/-{result['final_mean_reward_std_last10']:.4f} -> {ckpt_path}")
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_heldout_attacker_hard] device={device} config={HARD_HELDOUT_CONFIG} "
          f"train_iters={TRAIN_ITERS} rollout_len={ROLLOUT_LEN}")
    manifest = load_manifest()
    sessions = build_train_sessions(manifest)
    print(f"[train_heldout_attacker_hard] {len(sessions)} TRAIN sessions available for PPO rollouts")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    result = train_heldout_hard(sessions, device)

    log_path = CKPT_DIR / "heldout_attacker_hard_training_log.json"
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[train_heldout_attacker_hard] saved log -> {log_path}")


if __name__ == "__main__":
    main()
