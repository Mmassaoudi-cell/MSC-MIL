"""Part B step 1: train the FINAL, FROZEN MSC-MIL (FINAL_MODEL_CONFIG.yaml)
for all 5 seeds (0-4) on SCMA-1, reading every hyperparameter literally from
the YAML file (source of truth, per task brief) rather than from
src.candidates.stage3_optuna's in-memory best-params JSON, so this run
doubles as a config-file sanity check: if seed=0 does not reproduce Stage 3's
reported val macro-F1 ~0.9636, that is reported as a real finding, not
silently patched.

Only trains on {train, val} -- test_n200/test_full are never read here (that
happens once, together with every benchmark, in final_eval.py).
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import yaml

from src.benchmarks.data_utils import PROJECT_ROOT, get_device, load_scma_data
from src.candidates.msc_mil import MSCMIL
from src.candidates.smoke_test import infer_forward, n_params, train_msc
from src.reproduction.config import IDSConfig
from src.reproduction.metrics import binary_metrics, macro_f1_score

CONFIG_PATH = PROJECT_ROOT / "FINAL_MODEL_CONFIG.yaml"
RAW_DIR = PROJECT_ROOT / "results" / "raw" / "final_eval"
CKPT_DIR = PROJECT_ROOT / "results" / "checkpoints" / "final"
SCMA_TYPE = 1
SEEDS = (0, 1, 2, 3, 4)


def load_final_config():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    cfg = load_final_config()
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

    device = get_device()
    print(f"[final_msc_train] device={device} config={CONFIG_PATH}")
    print(f"[final_msc_train] hyperparams: lr={ids_cfg.lr} l2_reg={ids_cfg.l2_reg} dropout={ids_cfg.dropout} "
          f"beta={beta} tau={tau} max_epochs={ids_cfg.max_epochs} patience={ids_cfg.early_stopping_patience}")

    data = load_scma_data(SCMA_TYPE, device)
    print(f"[final_msc_train] SCMA-{SCMA_TYPE} train={len(data['train'].labels)} val={len(data['val'].labels)}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in SEEDS:
        out_path = RAW_DIR / f"msc_mil_final_scma{SCMA_TYPE}_seed{seed}.json"
        ckpt_path = CKPT_DIR / f"msc_mil_final_scma{SCMA_TYPE}_seed{seed}.pt"
        if out_path.exists():
            print(f"[final_msc_train] SKIP (exists) {out_path}")
            with open(out_path) as f:
                results.append(json.load(f))
            continue
        print(f"[final_msc_train] === seed={seed} ===")
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
            "config_source": str(CONFIG_PATH),
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
        if vm:
            print(f"[final_msc_train] seed={seed}: OK val_macro_f1={vm['macro_f1']:.4f} "
                  f"epochs={result['epochs_trained']} best_epoch={best_epoch} time={train_time_s:.1f}s")
        else:
            print(f"[final_msc_train] seed={seed}: FAIL ({fail_reason})")
        results.append(result)

    seed0 = next((r for r in results if r["seed"] == 0), None)
    if seed0 and seed0.get("val_metrics"):
        reported = 0.9636271889149528  # Stage 3 Part A Optuna best trial's single-seed val macro-F1
        actual = seed0["val_metrics"]["macro_f1"]
        diff = actual - reported
        print(f"[final_msc_train] CONFIG SANITY CHECK: seed0 val_macro_f1={actual:.4f} vs Stage3-reported={reported:.4f} "
              f"(diff={diff:+.4f})")
        with open(RAW_DIR / "config_sanity_check.json", "w") as f:
            json.dump({"stage3_reported_seed0_val_macro_f1": reported, "final_retrained_seed0_val_macro_f1": actual, "diff": diff}, f, indent=2)
    print("[final_msc_train] done.")


if __name__ == "__main__":
    main()
