"""Stage 2 validation screening (Master Prompt Sec.9 Stage 2) for the 3
Stage-1-surviving candidates (MGAT-MIL, MSC-MIL, GRMA-MIL; SSM-MIL was
dropped in Stage 1 for a resource-budget failure -- see MODEL_SELECTION_REPORT.md
Stage 1 section). VALIDATION-ONLY: models train on `train`, are evaluated on
`val`. `build_type_dataset` always constructs test_n200/test_full internally
(existing, unmodified dataset code, src/reproduction/dataset.py) -- this
script simply never reads those two keys of its returned dict, satisfying the
"never touch TEST split in this task" rule without needing to modify that
shared dataset-construction function.

Part A: 3 candidates x 3 SCMA types (1, 3, 6 -- documented scope reduction
from the full 6 types, see MODEL_SELECTION_REPORT.md) x 3 seeds (0, 1, 2) =
27 runs.

Reuses smoke_test.py's per-candidate train_* loops (train_mgat/train_msc/
train_grma) and infer_forward/n_params verbatim via import -- not
reimplemented. Loss hyperparameters are unchanged from Stage 1 (documented
ASSUMPTIONs there, not retuned here): GRL lambda_max=1.0/gamma=10, SupCon
beta=0.5/tau=0.1, K=4 generators (multigen_dataset.py + the 4 saved
ppo_attacker_gen{0..3}.pt checkpoints, reused not retrained).

Epoch budget: SAME as Stage 1 (max_epochs=30, patience=8 on val macro-F1@0.5)
-- a documented, deliberate reuse. Stage 2 is a *screening* pass across more
seeds/types, not final hyperparameter/epoch-budget tuning (that's Stage 3,
for the finalists only). Stage 1 showed all 3 candidates converging well
within this budget (10-29 epochs) on both SCMA-1 and SCMA-6.

Additional metrics beyond Stage 1's smoke test (per task spec): macro-F1
(metrics.macro_f1_score, new), balanced_accuracy, PR-AUC, recall, false_alarm
_rate (all from binary_metrics), training time, inference latency (ms/session,
batch=1, GPU, averaged over >=50 val sessions), parameter count.

For seed=0 runs on SCMA-1 and SCMA-6 (the two types used in Part B's
cross-generator diagnostic), the best model state_dict is ALSO saved to
results/checkpoints/stage2_<candidate>_scma<type>_seed0.pt so Part B
(stage2_crossgen.py) can reuse them without retraining.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.candidates import multigen_dataset as mgd
from src.candidates.grma_mil import GRMAMIL
from src.candidates.mgat_mil import MGATMIL
from src.candidates.msc_mil import MSCMIL
from src.candidates.smoke_test import (
    infer_forward,
    n_params,
    train_grma,
    train_mgat,
    train_msc,
)
from src.reproduction import dataset as dsmod
from src.reproduction.config import ReproductionConfig
from src.reproduction.metrics import aggregate_seeds, binary_metrics, macro_f1_score

PROJECT_ROOT = dsmod.PROJECT_ROOT
RAW_DIR = PROJECT_ROOT / "results" / "raw" / "candidates_stage2"
AGG_DIR = PROJECT_ROOT / "results" / "aggregate"
CKPT_DIR = PROJECT_ROOT / "results" / "checkpoints"

MAX_EPOCHS = 30
PATIENCE = 8
K_GENERATORS = 4
SCMA_TYPES = (1, 3, 6)
SEEDS = (0, 1, 2)
CANDIDATES = ("mgat_mil", "msc_mil", "grma_mil")
LATENCY_N_SESSIONS = 50
LATENCY_WARMUP = 5


@torch.no_grad()
def measure_latency_ms(model, val_data, device, n_sessions=LATENCY_N_SESSIONS, warmup=LATENCY_WARMUP) -> float:
    """Batch=1 GPU inference latency (ms/session), averaged over >=n_sessions
    val sessions. Warms up `warmup` iterations first (not counted) to exclude
    CUDA/cuDNN lazy-init cost from the measurement."""
    model.eval()
    n = min(n_sessions, len(val_data.labels))
    idx = np.arange(n)
    times = []
    for i in range(warmup + n):
        j = idx[i % n]
        s = torch.from_numpy(val_data.slots[j : j + 1]).to(device)
        m = torch.from_numpy(val_data.mask[j : j + 1]).to(device)
        st = torch.from_numpy(val_data.static[j : j + 1]).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(s, m, st)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            times.append(dt)
    return float(np.mean(times))


def full_eval(y_true, y_score, thresh=0.5) -> dict:
    y_pred = (y_score >= thresh).astype(int)
    m = binary_metrics(y_true, y_pred, y_score)
    m["macro_f1"] = macro_f1_score(y_true, y_pred)
    return m


def build_model(candidate: str, ids_cfg):
    if candidate == "mgat_mil":
        return MGATMIL(
            input_dim=ids_cfg.input_dim, encoder_hidden=ids_cfg.encoder_hidden, encoder_out=ids_cfg.encoder_out,
            lstm_hidden_per_dir=ids_cfg.lstm_hidden_per_dir, static_dim=ids_cfg.static_dim,
            static_hidden=ids_cfg.static_hidden, n_generators=K_GENERATORS, dropout=ids_cfg.dropout,
        )
    if candidate == "msc_mil":
        return MSCMIL(
            input_dim=ids_cfg.input_dim, encoder_hidden=ids_cfg.encoder_hidden, encoder_out=ids_cfg.encoder_out,
            lstm_hidden_per_dir=ids_cfg.lstm_hidden_per_dir, static_dim=ids_cfg.static_dim,
            static_hidden=ids_cfg.static_hidden, dropout=ids_cfg.dropout,
        )
    if candidate == "grma_mil":
        return GRMAMIL(
            input_dim=ids_cfg.input_dim, encoder_hidden=ids_cfg.encoder_hidden, encoder_out=ids_cfg.encoder_out,
            lstm_hidden_per_dir=ids_cfg.lstm_hidden_per_dir, static_dim=ids_cfg.static_dim,
            static_hidden=ids_cfg.static_hidden, n_generators=K_GENERATORS, dropout=ids_cfg.dropout,
        )
    raise ValueError(candidate)


def train_one(candidate: str, model, ids_cfg, base_data, multigen_train, device, seed):
    if candidate == "mgat_mil":
        return train_mgat(model, ids_cfg, multigen_train, base_data["val"], device, seed, max_epochs=MAX_EPOCHS, patience=PATIENCE)
    if candidate == "msc_mil":
        return train_msc(model, ids_cfg, base_data, device, seed, max_epochs=MAX_EPOCHS, patience=PATIENCE)
    if candidate == "grma_mil":
        return train_grma(model, ids_cfg, multigen_train, base_data["val"], device, seed, max_epochs=MAX_EPOCHS, patience=PATIENCE)
    raise ValueError(candidate)


def run_one(candidate, scma_type, seed, base_data, multigen_train, ids_cfg, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(candidate, ids_cfg).to(device)

    t0 = time.time()
    history, best_state, best_epoch, best_val_f1_at05, converged, fail_reason = train_one(
        candidate, model, ids_cfg, base_data, multigen_train, device, seed
    )
    train_time_s = time.time() - t0

    result = {
        "candidate": candidate,
        "scma_type": scma_type,
        "seed": seed,
        "converged": converged,
        "fail_reason": fail_reason,
        "n_params": n_params(model),
        "train_time_s": train_time_s,
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "best_val_macro_f1@0.5_earlystop": best_val_f1_at05,
    }

    if not converged or best_state is None:
        result["val_metrics"] = None
        result["inference_latency_ms"] = None
        return result, None

    model.load_state_dict(best_state)
    val_data = base_data["val"]
    val_score = infer_forward(model, val_data.slots, val_data.mask, val_data.static, device)
    result["val_metrics"] = full_eval(val_data.labels, val_score)
    result["inference_latency_ms"] = measure_latency_ms(model, val_data, device)
    return result, model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stage2_screening] device={device} candidates={CANDIDATES} types={SCMA_TYPES} seeds={SEEDS}")

    cfg = ReproductionConfig()
    ids_cfg = cfg.ids
    manifest = dsmod.load_manifest()
    store = dsmod.SlotStore.get()

    from src.reproduction.ppo_attacker import ActorCritic

    primary_ckpt = torch.load(CKPT_DIR / "ppo_attacker.pt", map_location=device)
    primary_attacker = ActorCritic(cfg.attacker).to(device)
    primary_attacker.load_state_dict(primary_ckpt["model_state"])
    primary_attacker.eval()

    gen_attackers = []
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

    all_results = []
    for scma_type in SCMA_TYPES:
        t0 = time.time()
        base_data = dsmod.build_type_dataset(manifest, store, pools, test_variants, scma_type, primary_attacker, device, cfg, dataset_seed=0)
        print(f"[stage2_screening] SCMA-{scma_type} base dataset built in {time.time()-t0:.1f}s "
              f"(train={len(base_data['train'].labels)}, val={len(base_data['val'].labels)}) -- "
              f"test_n200/test_full NOT read below (VALIDATION-ONLY per task rule)")

        t1 = time.time()
        scalers = mgd.fit_scalers_for_type(manifest, store, pools, scma_type, primary_attacker, device, cfg)
        multigen_train = mgd.build_multigen_train_split(
            manifest, store, pools, scma_type, gen_attackers, primary_attacker, device, cfg, scalers, dataset_seed=0
        )
        print(f"[stage2_screening] SCMA-{scma_type} multigen train dataset built in {time.time()-t1:.1f}s "
              f"(n={len(multigen_train.labels)})")

        for candidate in CANDIDATES:
            for seed in SEEDS:
                out_path = RAW_DIR / f"{candidate}_scma{scma_type}_seed{seed}.json"
                if out_path.exists():
                    print(f"[stage2_screening] SKIP (exists) {out_path}")
                    with open(out_path) as f:
                        all_results.append(json.load(f))
                    continue
                print(f"[stage2_screening] === {candidate} / SCMA-{scma_type} / seed={seed} ===")
                try:
                    result, model = run_one(candidate, scma_type, seed, base_data, multigen_train, ids_cfg, device)
                except Exception as e:
                    print(f"[stage2_screening] EXCEPTION {candidate}/SCMA-{scma_type}/seed{seed}: {e}")
                    traceback.print_exc()
                    result = {
                        "candidate": candidate, "scma_type": scma_type, "seed": seed,
                        "converged": False, "fail_reason": f"exception: {e}",
                        "n_params": None, "train_time_s": None, "epochs_trained": 0,
                        "best_epoch": -1, "best_val_macro_f1@0.5_earlystop": None,
                        "val_metrics": None, "inference_latency_ms": None,
                    }
                    model = None
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2, default=str)
                vm = result.get("val_metrics")
                status = "OK" if vm else "FAIL"
                if vm:
                    print(
                        f"[stage2_screening] {candidate} SCMA-{scma_type} seed={seed}: {status} "
                        f"macro_f1={vm['macro_f1']:.4f} recall={vm['recall']:.4f} "
                        f"far={vm['false_alarm_rate']:.4f} pr_auc={vm.get('pr_auc')} "
                        f"train_time={result['train_time_s']:.1f}s "
                        f"latency={result['inference_latency_ms']:.3f}ms n_params={result['n_params']} "
                        f"-> {out_path}"
                    )
                else:
                    print(f"[stage2_screening] {candidate} SCMA-{scma_type} seed={seed}: {status} ({result['fail_reason']}) -> {out_path}")
                all_results.append(result)

                if seed == 0 and model is not None and scma_type in (1, 6):
                    ckpt_path = CKPT_DIR / f"stage2_{candidate}_scma{scma_type}_seed0.pt"
                    torch.save({"model_state": model.state_dict(), "candidate": candidate, "scma_type": scma_type}, ckpt_path)
                    print(f"[stage2_screening] saved seed0 checkpoint -> {ckpt_path}")

    # ---- aggregate across seeds -> results/aggregate/candidates_stage2_screening.csv ----
    rows = []
    for candidate in CANDIDATES:
        for scma_type in SCMA_TYPES:
            metric_dicts = [
                r["val_metrics"] for r in all_results
                if r["candidate"] == candidate and r["scma_type"] == scma_type and r.get("val_metrics")
            ]
            if not metric_dicts:
                continue
            agg = aggregate_seeds(metric_dicts)
            for metric, stats in agg.items():
                rows.append({"candidate": candidate, "scma_type": scma_type, "metric": metric, **stats})
            # also aggregate train_time_s / inference_latency_ms / n_params across seeds
            extra_dicts = [
                {
                    "train_time_s": r["train_time_s"],
                    "inference_latency_ms": r["inference_latency_ms"],
                    "n_params": r["n_params"],
                }
                for r in all_results
                if r["candidate"] == candidate and r["scma_type"] == scma_type and r.get("val_metrics")
            ]
            if extra_dicts:
                agg2 = aggregate_seeds(extra_dicts)
                for metric, stats in agg2.items():
                    rows.append({"candidate": candidate, "scma_type": scma_type, "metric": metric, **stats})

    agg_df = pd.DataFrame(rows)
    agg_path = AGG_DIR / "candidates_stage2_screening.csv"
    if not agg_df.empty:
        agg_df = agg_df[["candidate", "scma_type", "metric", "mean", "std", "median", "ci95_lo", "ci95_hi", "n"]]
        agg_df.to_csv(agg_path, index=False)
        print(f"[stage2_screening] wrote aggregate -> {agg_path} ({len(agg_df)} rows)")
    print("[stage2_screening] Part A done.")


if __name__ == "__main__":
    main()
