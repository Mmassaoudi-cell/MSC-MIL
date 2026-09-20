"""Part 1 (post-freeze power analysis, Master Prompt Sec.22): extend the TEST
comparison between MSC-MIL and the 4 tree/boosting benchmarks (CatBoost,
XGBoost, LightGBM, RandomForest -- the ones showing the largest, most
consistent gap in BENCHMARK_RESULTS.md) from 5 to 10 seeds (seeds 0-9).

This is PURELY adding statistical power to an already-frozen comparison, not
re-optimizing: MSC-MIL reuses FINAL_MODEL_CONFIG.yaml's literal frozen
hyperparameters (same code path as final_msc_train.py) and the 4 tree models
reuse classical_tree.py's val-tuned best_params verbatim (no retuning).

Only seeds 5-9 are newly trained/fit here; seeds 0-4 are reused from
results/raw/final_eval/ (MSC-MIL) and results/raw/final_eval/all_methods_per_seed_metrics.json
(tree models), which were produced by final_msc_train.py / final_eval.py and
are read back in, not recomputed.

Writes:
  - results/raw/final_eval/msc_mil_final_scma1_seed{5..9}.json (+ checkpoints)
  - results/raw/final_eval/tree_models_n10_seed{5..9}.json (raw per-seed metrics)
  - results/raw/final_eval/n10_power_analysis_test_full.json (combined 10-seed
    test_full records for MSC-MIL + the 4 tree methods, seeds 0-9)
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import yaml

from src.benchmarks.classical_tree import fit_and_predict
from src.benchmarks.data_utils import PROJECT_ROOT, flatten_features, get_device, load_scma_data
from src.candidates.msc_mil import MSCMIL
from src.candidates.smoke_test import infer_forward, n_params, train_msc
from src.reproduction.config import IDSConfig
from src.reproduction.metrics import binary_metrics, macro_f1_score

CONFIG_PATH = PROJECT_ROOT / "FINAL_MODEL_CONFIG.yaml"
RAW_DIR = PROJECT_ROOT / "results" / "raw" / "final_eval"
CKPT_DIR = PROJECT_ROOT / "results" / "checkpoints" / "final"
AGG_DIR = PROJECT_ROOT / "results" / "aggregate"
SCMA_TYPE = 1
OLD_SEEDS = (0, 1, 2, 3, 4)
NEW_SEEDS = (5, 6, 7, 8, 9)
ALL_SEEDS = OLD_SEEDS + NEW_SEEDS
TREE_METHODS = ["catboost", "xgboost", "lightgbm", "random_forest"]


def train_msc_new_seeds(device):
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    arch = cfg["architecture"]
    hp = cfg["hyperparameters"]
    train_cfg = cfg["training"]
    ids_cfg = IDSConfig(
        lr=hp["learning_rate"], l2_reg=hp["l2_reg"], batch_size=hp["batch_size"],
        n_slots=arch["n_slots"], input_dim=arch["input_dim"], encoder_hidden=arch["encoder_hidden"],
        encoder_out=arch["encoder_out"], lstm_hidden_per_dir=arch["lstm_hidden_per_dir"],
        static_dim=arch["static_dim"], static_hidden=arch["static_hidden"], dropout=arch["dropout"],
        max_epochs=train_cfg["max_epochs"], early_stopping_patience=train_cfg["early_stopping_patience"],
        alert_threshold=cfg["alert_threshold"],
    )
    beta = hp["supcon_beta"]
    tau = hp["supcon_temperature"]
    data = load_scma_data(SCMA_TYPE, device)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    test_full_records = []
    for seed in NEW_SEEDS:
        out_path = RAW_DIR / f"msc_mil_final_scma{SCMA_TYPE}_seed{seed}.json"
        ckpt_path = CKPT_DIR / f"msc_mil_final_scma{SCMA_TYPE}_seed{seed}.pt"
        if out_path.exists() and ckpt_path.exists():
            print(f"[extend_seeds_n10] SKIP (exists) MSC-MIL seed={seed}")
        else:
            print(f"[extend_seeds_n10] === MSC-MIL seed={seed} (n=10 power extension) ===")
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = MSCMIL(
                input_dim=ids_cfg.input_dim, encoder_hidden=ids_cfg.encoder_hidden, encoder_out=ids_cfg.encoder_out,
                lstm_hidden_per_dir=ids_cfg.lstm_hidden_per_dir, static_dim=ids_cfg.static_dim,
                static_hidden=ids_cfg.static_hidden, dropout=ids_cfg.dropout,
            ).to(device)
            t0 = time.time()
            history, best_state, best_epoch, best_val_f1, converged, fail_reason = train_msc(
                model, ids_cfg, data, device, seed, beta=beta, tau=tau,
                max_epochs=ids_cfg.max_epochs, patience=ids_cfg.early_stopping_patience,
            )
            train_time_s = time.time() - t0
            result = {
                "scma_type": SCMA_TYPE, "seed": seed, "converged": converged, "fail_reason": fail_reason,
                "n_params": n_params(model), "train_time_s": train_time_s, "epochs_trained": len(history),
                "best_epoch": best_epoch, "best_val_macro_f1@0.5": best_val_f1,
                "config_source": str(CONFIG_PATH), "note": "n=10 power-analysis extension seed (Part 1)",
            }
            if converged and best_state is not None:
                model.load_state_dict(best_state)
                val = data["val"]
                val_score = infer_forward(model, val.slots, val.mask, val.static, device)
                val_pred = (val_score >= 0.5).astype(int)
                result["val_metrics"] = binary_metrics(val.labels, val_pred, val_score)
                result["val_metrics"]["macro_f1"] = macro_f1_score(val.labels, val_pred)
                torch.save({"model_state": model.state_dict(), "scma_type": SCMA_TYPE, "seed": seed}, ckpt_path)
            else:
                result["val_metrics"] = None
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            vm = result.get("val_metrics")
            print(f"[extend_seeds_n10] MSC-MIL seed={seed}: "
                  f"{'OK val_macro_f1=' + format(vm['macro_f1'], '.4f') if vm else 'FAIL ' + str(fail_reason)} "
                  f"time={train_time_s:.1f}s")

        # evaluate this seed's checkpoint on test_full
        ck = torch.load(ckpt_path, map_location=device)
        model = MSCMIL(
            input_dim=arch["input_dim"], encoder_hidden=arch["encoder_hidden"], encoder_out=arch["encoder_out"],
            lstm_hidden_per_dir=arch["lstm_hidden_per_dir"], static_dim=arch["static_dim"],
            static_hidden=arch["static_hidden"], dropout=arch["dropout"],
        ).to(device)
        model.load_state_dict(ck["model_state"])
        d = data["test_full"]
        score = infer_forward(model, d.slots, d.mask, d.static, device)
        pred = (score >= 0.5).astype(int)
        m = binary_metrics(d.labels, pred, score)
        m["macro_f1"] = macro_f1_score(d.labels, pred)
        test_full_records.append({"method": "MSC-MIL", "test_variant": "test_full", "seed": seed, "metrics": m})
        print(f"[extend_seeds_n10] MSC-MIL seed={seed}: test_full macro_f1={m['macro_f1']:.4f}")

    return test_full_records


def fit_tree_new_seeds(device):
    data = load_scma_data(SCMA_TYPE, device)
    with open(AGG_DIR / "benchmark_val_tuning_classical_tree.json") as f:
        tuned = json.load(f)
    Xtr = flatten_features(data["train"])
    ytr = data["train"].labels
    Xtest_full = flatten_features(data["test_full"])
    ytest_full = data["test_full"].labels

    out_path = RAW_DIR / f"tree_models_n10_seeds5to9.json"
    if out_path.exists():
        with open(out_path) as f:
            records = json.load(f)
        print(f"[extend_seeds_n10] SKIP (exists) tree models seeds 5-9 -> {out_path}")
        return records

    records = []
    for name in TREE_METHODS:
        params = tuned[name]["best_params"]
        for seed in NEW_SEEDS:
            est, proba = fit_and_predict(name, params, Xtr, ytr, Xtest_full, seed=seed)
            pred = (proba >= 0.5).astype(int)
            m = binary_metrics(ytest_full, pred, proba)
            m["macro_f1"] = macro_f1_score(ytest_full, pred)
            records.append({"method": name, "test_variant": "test_full", "seed": seed, "metrics": m})
            print(f"[extend_seeds_n10] {name} seed={seed}: test_full macro_f1={m['macro_f1']:.4f}")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"[extend_seeds_n10] wrote {out_path}")
    return records


def main():
    device = get_device()
    print(f"[extend_seeds_n10] device={device}")

    msc_new = train_msc_new_seeds(device)
    tree_new = fit_tree_new_seeds(device)

    # Pull existing seeds 0-4 test_full records for MSC-MIL + the 4 tree methods
    with open(RAW_DIR / "all_methods_per_seed_metrics.json") as f:
        existing = json.load(f)
    old_records = [
        r for r in existing
        if r["test_variant"] == "test_full" and (r["method"] == "MSC-MIL" or r["method"] in TREE_METHODS)
        and r["seed"] in OLD_SEEDS
    ]

    combined = old_records + msc_new + tree_new
    out_path = RAW_DIR / "n10_power_analysis_test_full.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"[extend_seeds_n10] wrote {out_path} ({len(combined)} records)")

    # sanity: 10 seeds per method
    for method in ["MSC-MIL"] + TREE_METHODS:
        n = len([r for r in combined if r["method"] == method])
        print(f"[extend_seeds_n10] {method}: n={n} test_full records (expect 10)")
    print("[extend_seeds_n10] done.")


if __name__ == "__main__":
    main()
