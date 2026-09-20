"""Stage 3 Part B, step 2 -- harder cross-generator robustness diagnostic,
resolving Stage 2's flagged caveat (MODEL_SELECTION_REPORT.md Stage 2 Part B:
the mild 50-iter/theta_thr=0.30 held-out generator found ~zero degradation for
any model, so the generator-invariance question was inconclusive).

Evaluates the TUNED (Part A best-config) MSC-MIL and GRMA-MIL, seed=0, on
SCMA-1 (and SCMA-6 if Part A's tuned 5-seed run shows it's still
discriminating -- see SCMA6_STILL_DISCRIMINATING below, set after reading
candidates_stage3_tuned.csv) against the harder held-out generator
(train_heldout_attacker_hard.py: theta_thr=0.35, 200 iters/256 rollout,
seed=123) vs. in-distribution val -- same style as Stage 2's Table B1.

VALIDATION-ONLY: only train/val splits are touched, same rule as Stage 2.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from src.candidates import multigen_dataset as mgd
from src.candidates.grma_mil import GRMAMIL
from src.candidates.msc_mil import MSCMIL
from src.candidates.smoke_test import infer_forward, n_params
from src.candidates.stage2_screening import full_eval
from src.reproduction import dataset as dsmod
from src.reproduction.config import ReproductionConfig
from src.reproduction.ppo_attacker import ActorCritic

PROJECT_ROOT = dsmod.PROJECT_ROOT
RAW_DIR = PROJECT_ROOT / "results" / "raw" / "candidates_stage3"
AGG_DIR = PROJECT_ROOT / "results" / "aggregate"
CKPT_DIR = PROJECT_ROOT / "results" / "checkpoints"

K_GENERATORS = 4


def scma6_still_discriminating() -> bool:
    """Reads Part A's tuned 5-seed aggregate to decide whether SCMA-6 carries
    any discriminative signal under tuned hyperparameters (Stage 2 found it
    trivially separable -- macro-F1=1.0, zero variance -- for all untuned
    candidates)."""
    path = AGG_DIR / "candidates_stage3_tuned.csv"
    if not path.exists():
        return False
    df = pd.read_csv(path)
    sub = df[(df.scma_type == 6) & (df.metric == "macro_f1")]
    if sub.empty:
        return False
    # "still discriminating" = any candidate's mean macro-F1 std > 0 or mean < 1.0
    return bool(((sub["mean"] < 0.999) | (sub["std"] > 1e-6)).any())


def load_tuned_checkpoint(candidate: str, scma_type: int, ids_cfg, device):
    ckpt_path = CKPT_DIR / f"stage3_{candidate}_scma{scma_type}_seed0_tuned.pt"
    ck = torch.load(ckpt_path, map_location=device)
    params = ck["params"]
    if candidate == "msc_mil":
        model = MSCMIL(
            input_dim=ids_cfg.input_dim, encoder_hidden=ids_cfg.encoder_hidden, encoder_out=ids_cfg.encoder_out,
            lstm_hidden_per_dir=ids_cfg.lstm_hidden_per_dir, static_dim=ids_cfg.static_dim,
            static_hidden=ids_cfg.static_hidden, dropout=params["dropout"],
        ).to(device)
    else:
        model = GRMAMIL(
            input_dim=ids_cfg.input_dim, encoder_hidden=ids_cfg.encoder_hidden, encoder_out=ids_cfg.encoder_out,
            lstm_hidden_per_dir=ids_cfg.lstm_hidden_per_dir, static_dim=ids_cfg.static_dim,
            static_hidden=ids_cfg.static_hidden, n_generators=K_GENERATORS, dropout=params["dropout"],
        ).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, params


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scma_types = [1]
    if scma6_still_discriminating():
        scma_types.append(6)
        print("[stage3_crossgen_hard] SCMA-6 STILL discriminates under tuned hyperparameters -- including it.")
    else:
        print("[stage3_crossgen_hard] SCMA-6 confirmed trivially separable under tuned hyperparameters (per task brief) -- SCMA-1 only.")

    print(f"[stage3_crossgen_hard] device={device} types={scma_types}")

    cfg = ReproductionConfig()
    ids_cfg = cfg.ids
    manifest = dsmod.load_manifest()
    store = dsmod.SlotStore.get()
    manifest_idx = manifest.set_index("session_uid")

    primary_ckpt = torch.load(CKPT_DIR / "ppo_attacker.pt", map_location=device)
    primary_attacker = ActorCritic(cfg.attacker).to(device)
    primary_attacker.load_state_dict(primary_ckpt["model_state"])
    primary_attacker.eval()

    heldout_hard_path = CKPT_DIR / "ppo_attacker_heldout_hard.pt"
    assert heldout_hard_path.exists(), "run train_heldout_attacker_hard.py first"
    heldout_ckpt = torch.load(heldout_hard_path, map_location=device)
    heldout_attacker = ActorCritic(cfg.attacker).to(device)
    heldout_attacker.load_state_dict(heldout_ckpt["model_state"])
    heldout_attacker.eval()

    pools = dsmod.build_split_pools(manifest, seed=0)
    test_variants = dsmod.build_test_variants(manifest, seed=0)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    AGG_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for scma_type in scma_types:
        print(f"[stage3_crossgen_hard] === SCMA-{scma_type} ===")
        base_data = dsmod.build_type_dataset(manifest, store, pools, test_variants, scma_type, primary_attacker, device, cfg, dataset_seed=0)
        scalers = mgd.fit_scalers_for_type(manifest, store, pools, scma_type, primary_attacker, device, cfg)

        val_benign_bags, val_attack_heldout_bags = dsmod.generate_split_bags(
            pools["val"], manifest_idx, store, scma_type, heldout_attacker, device, cfg.attacker.c_adjust
        )
        heldout_val_data = dsmod._bags_to_arrays(val_benign_bags + val_attack_heldout_bags, scalers, seed=0)
        print(f"[stage3_crossgen_hard] SCMA-{scma_type} HARD held-out-generator val pool built: n={len(heldout_val_data.labels)} "
              f"(pos_rate={heldout_val_data.labels.mean():.3f}) vs in-distribution val n={len(base_data['val'].labels)}")

        for candidate in ("msc_mil", "grma_mil"):
            model, params = load_tuned_checkpoint(candidate, scma_type, ids_cfg, device)

            id_score = infer_forward(model, base_data["val"].slots, base_data["val"].mask, base_data["val"].static, device)
            id_metrics = full_eval(base_data["val"].labels, id_score)

            ho_score = infer_forward(model, heldout_val_data.slots, heldout_val_data.mask, heldout_val_data.static, device)
            ho_metrics = full_eval(heldout_val_data.labels, ho_score)

            result = {
                "model": candidate, "scma_type": scma_type, "n_params": n_params(model), "tuned_params": params,
                "in_distribution": id_metrics, "held_out_generator_hard": ho_metrics,
                "delta_macro_f1": id_metrics["macro_f1"] - ho_metrics["macro_f1"],
                "delta_recall": id_metrics["recall"] - ho_metrics["recall"],
                "delta_false_alarm_rate": ho_metrics["false_alarm_rate"] - id_metrics["false_alarm_rate"],
            }
            out_path = RAW_DIR / f"crossgen_hard_{candidate}_scma{scma_type}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(
                f"[stage3_crossgen_hard] {candidate} SCMA-{scma_type}: "
                f"in-dist macro_f1={id_metrics['macro_f1']:.4f} recall={id_metrics['recall']:.4f} far={id_metrics['false_alarm_rate']:.4f} | "
                f"HARD held-out macro_f1={ho_metrics['macro_f1']:.4f} recall={ho_metrics['recall']:.4f} far={ho_metrics['false_alarm_rate']:.4f} | "
                f"delta_macro_f1={result['delta_macro_f1']:+.4f} delta_recall={result['delta_recall']:+.4f} -> {out_path}"
            )
            rows.append(result)

    flat_rows = []
    for r in rows:
        flat_rows.append({
            "model": r["model"], "scma_type": r["scma_type"], "n_params": r["n_params"],
            "id_macro_f1": r["in_distribution"]["macro_f1"], "id_recall": r["in_distribution"]["recall"],
            "id_precision": r["in_distribution"]["precision"], "id_false_alarm_rate": r["in_distribution"]["false_alarm_rate"],
            "ho_macro_f1": r["held_out_generator_hard"]["macro_f1"], "ho_recall": r["held_out_generator_hard"]["recall"],
            "ho_precision": r["held_out_generator_hard"]["precision"], "ho_false_alarm_rate": r["held_out_generator_hard"]["false_alarm_rate"],
            "delta_macro_f1": r["delta_macro_f1"], "delta_recall": r["delta_recall"], "delta_false_alarm_rate": r["delta_false_alarm_rate"],
        })
    df = pd.DataFrame(flat_rows)
    out_csv = AGG_DIR / "candidates_stage3_crossgen_hard.csv"
    df.to_csv(out_csv, index=False)
    print(f"[stage3_crossgen_hard] wrote summary -> {out_csv}")
    print("[stage3_crossgen_hard] Part B (hard) done.")


if __name__ == "__main__":
    main()
