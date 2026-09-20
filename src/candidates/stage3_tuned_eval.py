"""Stage 3 Part A (second half) -- take each candidate's Optuna BEST config
(results/aggregate/stage3_optuna_<candidate>_best.json, produced by
stage3_optuna.py) and re-run it with 5 seeds (0-4) on SCMA-1, SCMA-3, SCMA-6
to get a robust final validation number per candidate (task brief: "confirm
[SCMA-3/6 triviality] still holds with tuned hyperparameters rather than
assuming it").

Raw per-run JSON: results/raw/candidates_stage3/<candidate>_tuned_scma<type>_seed<seed>.json
Aggregate: results/aggregate/candidates_stage3_tuned.csv
Seed=0 checkpoints for SCMA-1/SCMA-6 saved to
results/checkpoints/stage3_<candidate>_scma<type>_seed0_tuned.pt (reused by
Part B's harder cross-generator diagnostic, not retrained there).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.candidates import multigen_dataset as mgd
from src.candidates.grma_mil import GRMAMIL
from src.candidates.msc_mil import MSCMIL
from src.candidates.smoke_test import infer_forward, n_params, train_msc
from src.candidates.stage2_screening import full_eval, measure_latency_ms
from src.candidates.stage3_optuna import train_grma_tuned
from src.reproduction import dataset as dsmod
from src.reproduction.config import ReproductionConfig
from src.reproduction.metrics import aggregate_seeds
from src.reproduction.ppo_attacker import ActorCritic

PROJECT_ROOT = dsmod.PROJECT_ROOT
RAW_DIR = PROJECT_ROOT / "results" / "raw" / "candidates_stage3"
AGG_DIR = PROJECT_ROOT / "results" / "aggregate"
CKPT_DIR = PROJECT_ROOT / "results" / "checkpoints"

MAX_EPOCHS = 30
PATIENCE = 8
K_GENERATORS = 4
SCMA_TYPES = (1, 3, 6)
SEEDS = (0, 1, 2, 3, 4)
CANDIDATES = ("msc_mil", "grma_mil")


def load_best_params(candidate: str) -> dict:
    path = AGG_DIR / f"stage3_optuna_{candidate}_best.json"
    with open(path) as f:
        return json.load(f)["best_params"]


def run_msc(base_data, ids_cfg, device, seed, params):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MSCMIL(
        input_dim=ids_cfg.input_dim, encoder_hidden=ids_cfg.encoder_hidden, encoder_out=ids_cfg.encoder_out,
        lstm_hidden_per_dir=ids_cfg.lstm_hidden_per_dir, static_dim=ids_cfg.static_dim,
        static_hidden=ids_cfg.static_hidden, dropout=params["dropout"],
    ).to(device)
    trial_cfg = dataclasses.replace(ids_cfg, lr=params["lr"])
    history, best_state, best_epoch, best_val_f1, converged, fail_reason = train_msc(
        model, trial_cfg, base_data, device, seed, beta=params["beta"], tau=params["temperature"],
        max_epochs=MAX_EPOCHS, patience=PATIENCE,
    )
    return model, history, best_state, best_epoch, converged, fail_reason


def run_grma(base_data, multigen_train, ids_cfg, device, seed, params):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GRMAMIL(
        input_dim=ids_cfg.input_dim, encoder_hidden=ids_cfg.encoder_hidden, encoder_out=ids_cfg.encoder_out,
        lstm_hidden_per_dir=ids_cfg.lstm_hidden_per_dir, static_dim=ids_cfg.static_dim,
        static_hidden=ids_cfg.static_hidden, n_generators=K_GENERATORS, dropout=params["dropout"],
    ).to(device)
    trial_cfg = dataclasses.replace(ids_cfg, lr=params["lr"])
    history, best_state, best_epoch, best_val_f1, converged, fail_reason = train_grma_tuned(
        model, trial_cfg, multigen_train, base_data["val"], device, seed,
        lambda_max=params["lambda_max"], gamma=params["gamma"], beta=params["beta"], tau=params["temperature"],
        max_epochs=MAX_EPOCHS, patience=PATIENCE,
    )
    return model, history, best_state, best_epoch, converged, fail_reason


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=list(CANDIDATES), default=None,
                         help="if omitted, runs both candidates sequentially in one process")
    args = parser.parse_args()
    candidates = [args.candidate] if args.candidate else list(CANDIDATES)
    need_multigen = "grma_mil" in candidates

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stage3_tuned_eval] device={device} candidates={candidates} types={SCMA_TYPES} seeds={SEEDS}")

    cfg = ReproductionConfig()
    ids_cfg = cfg.ids
    manifest = dsmod.load_manifest()
    store = dsmod.SlotStore.get()

    primary_ckpt = torch.load(CKPT_DIR / "ppo_attacker.pt", map_location=device)
    primary_attacker = ActorCritic(cfg.attacker).to(device)
    primary_attacker.load_state_dict(primary_ckpt["model_state"])
    primary_attacker.eval()

    gen_attackers = []
    if need_multigen:
        for k in range(K_GENERATORS):
            ck = torch.load(CKPT_DIR / f"ppo_attacker_gen{k}.pt", map_location=device)
            m = ActorCritic(cfg.attacker).to(device)
            m.load_state_dict(ck["model_state"])
            m.eval()
            gen_attackers.append(m)

    pools = dsmod.build_split_pools(manifest, seed=0)
    test_variants = dsmod.build_test_variants(manifest, seed=0)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    best_params = {c: load_best_params(c) for c in candidates}
    print(f"[stage3_tuned_eval] best params: {json.dumps(best_params, indent=2)}")

    all_results = []
    for scma_type in SCMA_TYPES:
        t0 = time.time()
        base_data = dsmod.build_type_dataset(manifest, store, pools, test_variants, scma_type, primary_attacker, device, cfg, dataset_seed=0)
        print(f"[stage3_tuned_eval] SCMA-{scma_type} base dataset built in {time.time()-t0:.1f}s "
              f"(train={len(base_data['train'].labels)}, val={len(base_data['val'].labels)})")

        multigen_train = None
        if need_multigen:
            t1 = time.time()
            scalers = mgd.fit_scalers_for_type(manifest, store, pools, scma_type, primary_attacker, device, cfg)
            multigen_train = mgd.build_multigen_train_split(
                manifest, store, pools, scma_type, gen_attackers, primary_attacker, device, cfg, scalers, dataset_seed=0
            )
            print(f"[stage3_tuned_eval] SCMA-{scma_type} multigen train dataset built in {time.time()-t1:.1f}s (n={len(multigen_train.labels)})")

        for candidate in candidates:
            params = best_params[candidate]
            for seed in SEEDS:
                out_path = RAW_DIR / f"{candidate}_tuned_scma{scma_type}_seed{seed}.json"
                if out_path.exists():
                    print(f"[stage3_tuned_eval] SKIP (exists) {out_path}")
                    with open(out_path) as f:
                        all_results.append(json.load(f))
                    continue
                print(f"[stage3_tuned_eval] === {candidate} / SCMA-{scma_type} / seed={seed} (tuned) ===")
                t2 = time.time()
                try:
                    if candidate == "msc_mil":
                        model, history, best_state, best_epoch, converged, fail_reason = run_msc(base_data, ids_cfg, device, seed, params)
                    else:
                        model, history, best_state, best_epoch, converged, fail_reason = run_grma(base_data, multigen_train, ids_cfg, device, seed, params)
                    train_time_s = time.time() - t2
                    result = {
                        "candidate": candidate, "scma_type": scma_type, "seed": seed, "tuned_params": params,
                        "converged": converged, "fail_reason": fail_reason, "n_params": n_params(model),
                        "train_time_s": train_time_s, "epochs_trained": len(history), "best_epoch": best_epoch,
                    }
                    if converged and best_state is not None:
                        model.load_state_dict(best_state)
                        val_data = base_data["val"]
                        val_score = infer_forward(model, val_data.slots, val_data.mask, val_data.static, device)
                        result["val_metrics"] = full_eval(val_data.labels, val_score)
                        result["inference_latency_ms"] = measure_latency_ms(model, val_data, device)
                    else:
                        result["val_metrics"] = None
                        result["inference_latency_ms"] = None
                        model = None
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    result = {
                        "candidate": candidate, "scma_type": scma_type, "seed": seed, "tuned_params": params,
                        "converged": False, "fail_reason": f"exception: {e}", "n_params": None,
                        "train_time_s": None, "epochs_trained": 0, "best_epoch": -1,
                        "val_metrics": None, "inference_latency_ms": None,
                    }
                    model = None

                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2, default=str)
                vm = result.get("val_metrics")
                if vm:
                    print(f"[stage3_tuned_eval] {candidate} SCMA-{scma_type} seed={seed}: OK macro_f1={vm['macro_f1']:.4f} "
                          f"recall={vm['recall']:.4f} far={vm['false_alarm_rate']:.4f} train_time={result['train_time_s']:.1f}s -> {out_path}")
                else:
                    print(f"[stage3_tuned_eval] {candidate} SCMA-{scma_type} seed={seed}: FAIL ({result['fail_reason']}) -> {out_path}")
                all_results.append(result)

                if seed == 0 and model is not None and scma_type in (1, 6):
                    ckpt_path = CKPT_DIR / f"stage3_{candidate}_scma{scma_type}_seed0_tuned.pt"
                    torch.save({"model_state": model.state_dict(), "candidate": candidate, "scma_type": scma_type, "params": params}, ckpt_path)
                    print(f"[stage3_tuned_eval] saved tuned seed0 checkpoint -> {ckpt_path}")

    # Re-read ALL raw JSONs on disk (not just this process's in-memory
    # all_results) so the aggregate is correct/complete whether this script
    # is run once for both candidates or twice (--candidate msc_mil /
    # --candidate grma_mil) as 2 separate background processes.
    disk_results = []
    for candidate in CANDIDATES:
        for scma_type in SCMA_TYPES:
            for seed in SEEDS:
                p = RAW_DIR / f"{candidate}_tuned_scma{scma_type}_seed{seed}.json"
                if p.exists():
                    with open(p) as f:
                        disk_results.append(json.load(f))

    rows = []
    for candidate in CANDIDATES:
        for scma_type in SCMA_TYPES:
            metric_dicts = [
                r["val_metrics"] for r in disk_results
                if r["candidate"] == candidate and r["scma_type"] == scma_type and r.get("val_metrics")
            ]
            if not metric_dicts:
                continue
            agg = aggregate_seeds(metric_dicts)
            for metric, stats in agg.items():
                rows.append({"candidate": candidate, "scma_type": scma_type, "metric": metric, **stats})
            extra_dicts = [
                {"train_time_s": r["train_time_s"], "inference_latency_ms": r["inference_latency_ms"], "n_params": r["n_params"]}
                for r in disk_results
                if r["candidate"] == candidate and r["scma_type"] == scma_type and r.get("val_metrics")
            ]
            if extra_dicts:
                agg2 = aggregate_seeds(extra_dicts)
                for metric, stats in agg2.items():
                    rows.append({"candidate": candidate, "scma_type": scma_type, "metric": metric, **stats})

    agg_df = pd.DataFrame(rows)
    agg_path = AGG_DIR / "candidates_stage3_tuned.csv"
    if not agg_df.empty:
        agg_df = agg_df[["candidate", "scma_type", "metric", "mean", "std", "median", "ci95_lo", "ci95_hi", "n"]]
        agg_df.to_csv(agg_path, index=False)
        print(f"[stage3_tuned_eval] wrote aggregate -> {agg_path} ({len(agg_df)} rows, {len(disk_results)} raw JSONs found on disk)")
    print("[stage3_tuned_eval] Part A (tuned 5-seed) done.")


if __name__ == "__main__":
    main()
