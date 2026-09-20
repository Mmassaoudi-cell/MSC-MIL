"""Part 2 (post-freeze ablation, Master Prompt Sec.19: 'Full Model vs Full
Model - Component'). Isolates MSC-MIL's two components beyond the base
MIL-BiLSTM+attention backbone (src/reproduction/mil_ids.py):
  (a) the 3-branch multi-scale architecture (pooling at {1,3,12} slots/token)
  (b) the SupCon contrastive loss term (beta*L_con)

All variants use the SAME frozen hyperparameters from FINAL_MODEL_CONFIG.yaml
(dropout, lr, l2_reg, batch_size, max_epochs/patience, alert_threshold) where
applicable -- only pool_sizes / lstm_hidden_per_dir / supcon beta are varied,
exactly per the task brief. Reuses src.candidates.msc_mil.MSCMIL (now generalized
to accept a `pool_sizes` ctor arg, see msc_mil.py) and
src.candidates.smoke_test.train_msc verbatim (the SAME training-loop code path
used for the frozen final model) -- no reimplementation of the training loop.

Variants:
  1. full_msc_mil          : NOT retrained here -- reuses results/raw/final_eval/
                              msc_mil_final_scma1_seed{0-4}.json (5 seeds, frozen).
  2. no_contrastive         : pool_sizes=(1,3,12) [multi-scale kept], beta=0
                              (L = L_det only) -- isolates SupCon's contribution.
  3. no_multiscale          : pool_sizes=(1,) [single branch, architecturally
                              == the base MIL-BiLSTM+attention backbone],
                              beta=<frozen> [SupCon KEPT] -- isolates multi-scale's
                              contribution. NOT the same as SOURCE_METHOD_REPRODUCTION
                              (neither multiscale nor SupCon) or Plain-LSTM-MIL
                              (neither multiscale, SupCon, nor attention) --
                              both already exist in BENCHMARK_RESULTS.md as
                              reference points, not retrained.
  4. param_matched_wide     : pool_sizes=(1,) [no multiscale], beta=0 [no SupCon],
                              lstm_hidden_per_dir=128 (n_params=203777, ~99.4%
                              of MSC-MIL's 205057 -- see search in this module's
                              companion note) -- tests whether multi-scale's
                              benefit (if any) is about multiple resolutions or
                              just more parameters.

Saves per-seed raw JSON + checkpoints to results/raw/ablation/ /
results/checkpoints/ablation/, aggregates to results/aggregate/ablation_results.csv.
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
from src.reproduction.metrics import aggregate_seeds, binary_metrics, macro_f1_score

CONFIG_PATH = PROJECT_ROOT / "FINAL_MODEL_CONFIG.yaml"
RAW_DIR = PROJECT_ROOT / "results" / "raw" / "ablation"
CKPT_DIR = PROJECT_ROOT / "results" / "checkpoints" / "ablation"
AGG_DIR = PROJECT_ROOT / "results" / "aggregate"
FINAL_RAW_DIR = PROJECT_ROOT / "results" / "raw" / "final_eval"
FINAL_CKPT_DIR = PROJECT_ROOT / "results" / "checkpoints" / "final"
SCMA_TYPE = 1
SEEDS = (0, 1, 2, 3, 4)
TEST_VARIANTS = ["test_n200", "test_full"]

PARAM_MATCHED_HIDDEN = 128  # search: MSCMIL(pool_sizes=(1,), hidden=128) -> 203777 params (~99.4% of 205057)


def build_variants(arch, hp):
    return {
        "no_contrastive": dict(pool_sizes=(1, 3, 12), lstm_hidden_per_dir=arch["lstm_hidden_per_dir"],
                                beta=0.0, tau=hp["supcon_temperature"],
                                desc="multi-scale kept, beta=0 (L=L_det only) -- isolates SupCon's contribution"),
        "no_multiscale": dict(pool_sizes=(1,), lstm_hidden_per_dir=arch["lstm_hidden_per_dir"],
                               beta=hp["supcon_beta"], tau=hp["supcon_temperature"],
                               desc="single branch (== base backbone architecture), SupCon KEPT -- isolates multi-scale's contribution"),
        "param_matched_wide": dict(pool_sizes=(1,), lstm_hidden_per_dir=PARAM_MATCHED_HIDDEN,
                                    beta=0.0, tau=hp["supcon_temperature"],
                                    desc="single branch, hidden=128 (n_params~=203777, param-matched to MSC-MIL's 205057), beta=0"),
    }


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    arch = cfg["architecture"]
    hp = cfg["hyperparameters"]
    train_cfg = cfg["training"]
    ids_cfg_base = IDSConfig(
        lr=hp["learning_rate"], l2_reg=hp["l2_reg"], batch_size=hp["batch_size"],
        n_slots=arch["n_slots"], input_dim=arch["input_dim"], encoder_hidden=arch["encoder_hidden"],
        encoder_out=arch["encoder_out"], static_dim=arch["static_dim"], static_hidden=arch["static_hidden"],
        dropout=arch["dropout"], max_epochs=train_cfg["max_epochs"],
        early_stopping_patience=train_cfg["early_stopping_patience"], alert_threshold=cfg["alert_threshold"],
    )

    device = get_device()
    print(f"[ablation_train] device={device}")
    data = load_scma_data(SCMA_TYPE, device)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    variants = build_variants(arch, hp)
    all_test_records = []  # method, test_variant, seed, metrics

    # ---- variant 1: full MSC-MIL, reused (not retrained) ----
    for seed in SEEDS:
        ckpt_path = FINAL_CKPT_DIR / f"msc_mil_final_scma{SCMA_TYPE}_seed{seed}.pt"
        if not ckpt_path.exists():
            print(f"[ablation_train] WARNING missing full MSC-MIL checkpoint seed={seed}")
            continue
        ck = torch.load(ckpt_path, map_location=device)
        model = MSCMIL(
            input_dim=arch["input_dim"], encoder_hidden=arch["encoder_hidden"], encoder_out=arch["encoder_out"],
            lstm_hidden_per_dir=arch["lstm_hidden_per_dir"], static_dim=arch["static_dim"],
            static_hidden=arch["static_hidden"], dropout=arch["dropout"], pool_sizes=(1, 3, 12),
        ).to(device)
        model.load_state_dict(ck["model_state"])
        for tv in TEST_VARIANTS:
            d = data[tv]
            score = infer_forward(model, d.slots, d.mask, d.static, device)
            pred = (score >= 0.5).astype(int)
            m = binary_metrics(d.labels, pred, score)
            m["macro_f1"] = macro_f1_score(d.labels, pred)
            all_test_records.append({"method": "full_msc_mil", "test_variant": tv, "seed": seed, "metrics": m,
                                      "n_params": n_params(model)})
    print(f"[ablation_train] full_msc_mil: reused {len(SEEDS)} seeds from results/raw/final_eval (not retrained)")

    # ---- variants 2-4: train 5 seeds each ----
    for vname, vcfg in variants.items():
        for seed in SEEDS:
            out_path = RAW_DIR / f"{vname}_scma{SCMA_TYPE}_seed{seed}.json"
            ckpt_path = CKPT_DIR / f"{vname}_scma{SCMA_TYPE}_seed{seed}.pt"
            if out_path.exists() and ckpt_path.exists():
                print(f"[ablation_train] SKIP (exists) {vname} seed={seed}")
            else:
                print(f"[ablation_train] === {vname} seed={seed} === ({vcfg['desc']})")
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = MSCMIL(
                    input_dim=ids_cfg_base.input_dim, encoder_hidden=ids_cfg_base.encoder_hidden,
                    encoder_out=ids_cfg_base.encoder_out, lstm_hidden_per_dir=vcfg["lstm_hidden_per_dir"],
                    static_dim=ids_cfg_base.static_dim, static_hidden=ids_cfg_base.static_hidden,
                    dropout=ids_cfg_base.dropout, pool_sizes=vcfg["pool_sizes"],
                ).to(device)
                t0 = time.time()
                history, best_state, best_epoch, best_val_f1, converged, fail_reason = train_msc(
                    model, ids_cfg_base, data, device, seed, beta=vcfg["beta"], tau=vcfg["tau"],
                    max_epochs=ids_cfg_base.max_epochs, patience=ids_cfg_base.early_stopping_patience,
                )
                train_time_s = time.time() - t0
                result = {
                    "variant": vname, "scma_type": SCMA_TYPE, "seed": seed, "converged": converged,
                    "fail_reason": fail_reason, "n_params": n_params(model), "train_time_s": train_time_s,
                    "epochs_trained": len(history), "best_epoch": best_epoch, "best_val_macro_f1@0.5": best_val_f1,
                    "pool_sizes": list(vcfg["pool_sizes"]), "lstm_hidden_per_dir": vcfg["lstm_hidden_per_dir"],
                    "beta": vcfg["beta"], "tau": vcfg["tau"],
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
                print(f"[ablation_train] {vname} seed={seed}: "
                      f"{'OK val_macro_f1=' + format(vm['macro_f1'], '.4f') if vm else 'FAIL ' + str(fail_reason)} "
                      f"n_params={result['n_params']} time={train_time_s:.1f}s")

            ck = torch.load(ckpt_path, map_location=device)
            model = MSCMIL(
                input_dim=ids_cfg_base.input_dim, encoder_hidden=ids_cfg_base.encoder_hidden,
                encoder_out=ids_cfg_base.encoder_out, lstm_hidden_per_dir=vcfg["lstm_hidden_per_dir"],
                static_dim=ids_cfg_base.static_dim, static_hidden=ids_cfg_base.static_hidden,
                dropout=ids_cfg_base.dropout, pool_sizes=vcfg["pool_sizes"],
            ).to(device)
            model.load_state_dict(ck["model_state"])
            for tv in TEST_VARIANTS:
                d = data[tv]
                score = infer_forward(model, d.slots, d.mask, d.static, device)
                pred = (score >= 0.5).astype(int)
                m = binary_metrics(d.labels, pred, score)
                m["macro_f1"] = macro_f1_score(d.labels, pred)
                all_test_records.append({"method": vname, "test_variant": tv, "seed": seed, "metrics": m,
                                          "n_params": n_params(model)})
            print(f"[ablation_train] {vname} seed={seed}: test_full macro_f1="
                  f"{[r['metrics']['macro_f1'] for r in all_test_records if r['method']==vname and r['seed']==seed and r['test_variant']=='test_full'][0]:.4f}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_DIR / "ablation_all_test_records.json", "w") as f:
        json.dump(all_test_records, f, indent=2, default=str)
    print(f"[ablation_train] wrote {RAW_DIR / 'ablation_all_test_records.json'} ({len(all_test_records)} records)")

    # ---- aggregate ----
    rows = []
    methods = ["full_msc_mil"] + list(variants.keys())
    metric_keys = ["macro_f1", "recall", "false_alarm_rate", "precision", "f1", "accuracy"]
    for method in methods:
        for tv in TEST_VARIANTS:
            recs = [r for r in all_test_records if r["method"] == method and r["test_variant"] == tv]
            if not recs:
                continue
            metric_dicts = [r["metrics"] for r in recs]
            agg = aggregate_seeds(metric_dicts)
            for mk in metric_keys:
                if mk in agg:
                    rows.append({"method": method, "test_variant": tv, "metric": mk,
                                 "mean": agg[mk]["mean"], "std": agg[mk]["std"], "n_seeds": agg[mk]["n"]})
            n_params_vals = [r["n_params"] for r in recs]
            rows.append({"method": method, "test_variant": tv, "metric": "n_params",
                         "mean": float(np.mean(n_params_vals)), "std": float(np.std(n_params_vals)), "n_seeds": len(n_params_vals)})

    import pandas as pd
    out_df = pd.DataFrame(rows)
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = AGG_DIR / "ablation_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[ablation_train] wrote {out_path} ({len(out_df)} rows)")
    print("[ablation_train] done.")


if __name__ == "__main__":
    main()
