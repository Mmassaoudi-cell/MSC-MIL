"""Stage 3 Part A -- Optuna hyperparameter tuning (validation-only) for the 2
Stage-2 finalists (MSC-MIL, GRMA-MIL; MGAT-MIL was cut in Stage 2, see
MODEL_SELECTION_REPORT.md). optuna.create_study(direction="maximize") with
the TPE sampler (Optuna's default -- NOT a grid search, per Master Prompt
Sec.9's explicit "avoid giant grid searches" instruction).

SCMA-1 ONLY (the one type with real screening signal -- Stage 2 found SCMA-3/6
trivially separable for every candidate at every seed). 1 seed (0) per trial
(Optuna trials are inherently noisy/cheap by design; TPE handles a noisy
objective). max_epochs=30, patience=8 -- SAME epoch budget as Stage 1/2, not
itself searched (task brief: keep the search space focused).

Search space (task brief, Part A):
  MSC-MIL:  beta [0.05,2.0] log-uniform, temperature [0.05,0.5] uniform,
            dropout [0.1,0.4] uniform, lr [3e-4,3e-3] log-uniform.
  GRMA-MIL: same 4 PLUS lambda_max [0.1,2.0] uniform, gamma [1,20] uniform.
  Pooling scales {1,3,12} and K=4 generators are fixed ARCHITECTURE, not
  searched here (too expensive / not what Part A is asking).

Usage: python -m src.candidates.stage3_optuna --candidate msc_mil --n-trials 20
       python -m src.candidates.stage3_optuna --candidate grma_mil --n-trials 20
(run as 2 separate background processes -- allowed "at most 2-at-a-time" per
the task's GPU-contention rule, since these are 2 independent studies.)

Output: results/aggregate/stage3_optuna_<candidate>.csv (study.trials_dataframe()),
        results/aggregate/stage3_optuna_<candidate>_best.json (best trial's
        params + value, consumed directly by stage3_tuned_eval.py).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.candidates.common import dann_lambda, supcon_loss
from src.candidates.grma_mil import GRMAMIL
from src.candidates.msc_mil import MSCMIL
from src.candidates.smoke_test import _common_loop, n_params, train_msc
from src.candidates import multigen_dataset as mgd
from src.reproduction import dataset as dsmod
from src.reproduction.config import ReproductionConfig
from src.reproduction.metrics import macro_f1_score
from src.reproduction.mil_ids import bce_l2_loss
from src.reproduction.ppo_attacker import ActorCritic

PROJECT_ROOT = dsmod.PROJECT_ROOT
AGG_DIR = PROJECT_ROOT / "results" / "aggregate"
CKPT_DIR = PROJECT_ROOT / "results" / "checkpoints"

MAX_EPOCHS = 30
PATIENCE = 8
K_GENERATORS = 4
SCMA_TYPE = 1
SEED = 0


def train_grma_tuned(model, ids_cfg, multigen_train, val_data, device, seed, lambda_max, gamma, beta, tau, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    """Same as smoke_test.train_grma but with gamma also exposed as a tuned
    hyperparameter (smoke_test.train_grma hardcodes DANN gamma=10 via
    common.dann_lambda's default, since it wasn't tuned in Stage 1/2)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = TensorDataset(
        torch.from_numpy(multigen_train.slots), torch.from_numpy(multigen_train.mask),
        torch.from_numpy(multigen_train.static), torch.from_numpy(multigen_train.labels).float(),
        torch.from_numpy(multigen_train.gen_id),
    )
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=ids_cfg.batch_size, shuffle=True, generator=gen)

    def compute_loss(model, batch, device, progress):
        slots, mask, static, labels, gen_id = [t.to(device) for t in batch]
        lam = dann_lambda(progress, lambda_max=lambda_max, gamma=gamma)
        logits, _, domain_logits, H = model.logits_with_domain(slots, mask, static, lam)
        det = bce_l2_loss(logits, labels, model, ids_cfg.l2_reg)
        valid = gen_id >= 0
        dom = F.cross_entropy(domain_logits[valid], gen_id[valid]) if valid.any() else logits.new_zeros(())
        con = supcon_loss(H, labels, temperature=tau)
        return det + lam * dom + beta * con

    return _common_loop(
        model, device, ids_cfg, loader,
        val_data.slots, val_data.mask, val_data.static, val_data.labels,
        compute_loss, max_epochs=max_epochs, patience=patience,
    )


def make_objective_msc(base_data, ids_cfg, device):
    def objective(trial: optuna.Trial) -> float:
        beta = trial.suggest_float("beta", 0.05, 2.0, log=True)
        temperature = trial.suggest_float("temperature", 0.05, 0.5)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model = MSCMIL(
            input_dim=ids_cfg.input_dim, encoder_hidden=ids_cfg.encoder_hidden, encoder_out=ids_cfg.encoder_out,
            lstm_hidden_per_dir=ids_cfg.lstm_hidden_per_dir, static_dim=ids_cfg.static_dim,
            static_hidden=ids_cfg.static_hidden, dropout=dropout,
        ).to(device)
        trial_cfg = dataclasses.replace(ids_cfg, lr=lr)

        t0 = time.time()
        history, best_state, best_epoch, best_val_f1, converged, fail_reason = train_msc(
            model, trial_cfg, base_data, device, SEED, beta=beta, tau=temperature,
            max_epochs=MAX_EPOCHS, patience=PATIENCE,
        )
        dt = time.time() - t0

        trial.set_user_attr("converged", converged)
        trial.set_user_attr("fail_reason", fail_reason)
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("epochs_trained", len(history))
        trial.set_user_attr("n_params", n_params(model))
        trial.set_user_attr("train_time_s", dt)

        if not converged or best_val_f1 is None or best_val_f1 < 0:
            return 0.0
        return float(best_val_f1)

    return objective


def make_objective_grma(base_data, multigen_train, ids_cfg, device):
    def objective(trial: optuna.Trial) -> float:
        beta = trial.suggest_float("beta", 0.05, 2.0, log=True)
        temperature = trial.suggest_float("temperature", 0.05, 0.5)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
        lambda_max = trial.suggest_float("lambda_max", 0.1, 2.0)
        gamma = trial.suggest_float("gamma", 1.0, 20.0)

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model = GRMAMIL(
            input_dim=ids_cfg.input_dim, encoder_hidden=ids_cfg.encoder_hidden, encoder_out=ids_cfg.encoder_out,
            lstm_hidden_per_dir=ids_cfg.lstm_hidden_per_dir, static_dim=ids_cfg.static_dim,
            static_hidden=ids_cfg.static_hidden, n_generators=K_GENERATORS, dropout=dropout,
        ).to(device)
        trial_cfg = dataclasses.replace(ids_cfg, lr=lr)

        t0 = time.time()
        history, best_state, best_epoch, best_val_f1, converged, fail_reason = train_grma_tuned(
            model, trial_cfg, multigen_train, base_data["val"], device, SEED,
            lambda_max=lambda_max, gamma=gamma, beta=beta, tau=temperature,
            max_epochs=MAX_EPOCHS, patience=PATIENCE,
        )
        dt = time.time() - t0

        trial.set_user_attr("converged", converged)
        trial.set_user_attr("fail_reason", fail_reason)
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("epochs_trained", len(history))
        trial.set_user_attr("n_params", n_params(model))
        trial.set_user_attr("train_time_s", dt)

        if not converged or best_val_f1 is None or best_val_f1 < 0:
            return 0.0
        return float(best_val_f1)

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=["msc_mil", "grma_mil"], required=True)
    parser.add_argument("--n-trials", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stage3_optuna] candidate={args.candidate} n_trials={args.n_trials} device={device} SCMA-{SCMA_TYPE} seed={SEED}")

    cfg = ReproductionConfig()
    ids_cfg = cfg.ids
    manifest = dsmod.load_manifest()
    store = dsmod.SlotStore.get()

    primary_ckpt = torch.load(CKPT_DIR / "ppo_attacker.pt", map_location=device)
    primary_attacker = ActorCritic(cfg.attacker).to(device)
    primary_attacker.load_state_dict(primary_ckpt["model_state"])
    primary_attacker.eval()

    pools = dsmod.build_split_pools(manifest, seed=0)
    test_variants = dsmod.build_test_variants(manifest, seed=0)

    t0 = time.time()
    base_data = dsmod.build_type_dataset(manifest, store, pools, test_variants, SCMA_TYPE, primary_attacker, device, cfg, dataset_seed=0)
    print(f"[stage3_optuna] SCMA-{SCMA_TYPE} base dataset built in {time.time()-t0:.1f}s "
          f"(train={len(base_data['train'].labels)}, val={len(base_data['val'].labels)})")

    AGG_DIR.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=f"stage3_{args.candidate}")

    def log_trial(study, trial):
        print(
            f"[stage3_optuna] {args.candidate} trial {trial.number}: value={trial.value:.4f} "
            f"params={trial.params} converged={trial.user_attrs.get('converged')} "
            f"best_epoch={trial.user_attrs.get('best_epoch')} time={trial.user_attrs.get('train_time_s', 0):.1f}s"
        )

    if args.candidate == "msc_mil":
        objective = make_objective_msc(base_data, ids_cfg, device)
    else:
        t1 = time.time()
        scalers = mgd.fit_scalers_for_type(manifest, store, pools, SCMA_TYPE, primary_attacker, device, cfg)
        gen_attackers = []
        for k in range(K_GENERATORS):
            ck = torch.load(CKPT_DIR / f"ppo_attacker_gen{k}.pt", map_location=device)
            m = ActorCritic(cfg.attacker).to(device)
            m.load_state_dict(ck["model_state"])
            m.eval()
            gen_attackers.append(m)
        multigen_train = mgd.build_multigen_train_split(
            manifest, store, pools, SCMA_TYPE, gen_attackers, primary_attacker, device, cfg, scalers, dataset_seed=0
        )
        print(f"[stage3_optuna] SCMA-{SCMA_TYPE} multigen train dataset built in {time.time()-t1:.1f}s (n={len(multigen_train.labels)})")
        objective = make_objective_grma(base_data, multigen_train, ids_cfg, device)

    study.optimize(objective, n_trials=args.n_trials, callbacks=[log_trial])

    df = study.trials_dataframe()
    csv_path = AGG_DIR / f"stage3_optuna_{args.candidate}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[stage3_optuna] wrote trial history -> {csv_path} ({len(df)} trials)")

    best = study.best_trial
    best_out = {
        "candidate": args.candidate,
        "scma_type": SCMA_TYPE,
        "seed": SEED,
        "n_trials": args.n_trials,
        "best_trial_number": best.number,
        "best_val_macro_f1": best.value,
        "best_params": best.params,
        "best_user_attrs": best.user_attrs,
    }
    best_path = AGG_DIR / f"stage3_optuna_{args.candidate}_best.json"
    with open(best_path, "w") as f:
        json.dump(best_out, f, indent=2, default=str)
    print(f"[stage3_optuna] BEST trial for {args.candidate}: #{best.number} val_macro_f1={best.value:.4f} params={best.params} -> {best_path}")
    print("[stage3_optuna] done.")


if __name__ == "__main__":
    main()
