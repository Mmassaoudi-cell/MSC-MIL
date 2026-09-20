"""Stage 2 Part B -- cross-generator robustness diagnostic (Master Prompt
Sec.18/20; SOURCE_WEAKNESS_ANALYSIS.md sec. A1, the paper's own biggest gap).

Tests whether MGAT-MIL / GRMA-MIL's generator-invariance mechanism (GRL
domain-adversarial head trained against a K=4-generator attack pool) actually
buys anything when facing an attacker style never seen during training,
versus MSC-MIL (no invariance mechanism) and the baseline MIL-BiLSTM IDS.

Procedure (see task brief):
  1. Train a held-out 5th PPO generator, theta_thr=0.30 (outside the K=4
     generators' {0.10,0.15,0.20,0.25} range), seed=99 -- done by
     train_heldout_attacker.py, reused here (not retrained).
  2. For SCMA-1 and SCMA-6, build a validation-style attack pool using this
     held-out generator's predict_c_attack instead of the primary/K=4
     generators (dataset.generate_split_bags, swapping in the held-out
     attacker model) -- SAME val benign pool + SAME scalers as Part A's
     in-distribution val set, so the only thing that changes is which policy
     generated the attack sessions' C_attack.
  3. Evaluate all 3 candidates' Stage 2 seed=0 checkpoints (saved by
     stage2_screening.py to results/checkpoints/stage2_<candidate>_scma
     <type>_seed0.pt) plus a freshly seed=0-trained baseline MIL-BiLSTM IDS
     (no reproduction checkpoint existed on disk -- see module docstring
     below) on BOTH the in-distribution val set and the held-out-generator
     val pool.
  4. Report degradation (in-distribution macro-F1/recall minus held-out
     macro-F1/recall) per model -- the falsifiable test of whether generator-
     invariance training helps.

VALIDATION-ONLY: only `train`/`val` splits are touched; test_n200/test_full
are never read (same rule as stage2_screening.py Part A).

Baseline note: results/checkpoints/ contains only PPO attacker checkpoints --
src/reproduction/train_ids.py never saves an IDS checkpoint. Per the task
brief's documented fallback ("if not, this comparison point can use the
baseline's already-computed val-set numbers ... as the in-distribution
reference, with the held-out-generator eval as the new number"), we train a
fresh seed=0 baseline IDS here, restricted to train/val only (a new, smaller
training loop that reuses MILBiLSTMIDS + bce_l2_loss verbatim but does NOT
call train_ids.py's train_one_seed, since that function's own body reads
test_n200/test_full internally -- reusing it unmodified would violate this
task's "never read the TEST split" rule). Its in-distribution val macro-F1 is
cross-checked against results/raw/reproduction/scma{1,6}_seed0.json's
best_val_macro_f1 (the original reproduction run) as a sanity reference, but
the held-out-generator number necessarily comes from this freshly trained
copy since no earlier checkpoint exists to load.
"""
from __future__ import annotations

import copy
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.candidates import multigen_dataset as mgd
from src.candidates.grma_mil import GRMAMIL
from src.candidates.mgat_mil import MGATMIL
from src.candidates.msc_mil import MSCMIL
from src.candidates.stage2_screening import CANDIDATES as SCREEN_CANDIDATES
from src.candidates.stage2_screening import K_GENERATORS, build_model, full_eval, measure_latency_ms
from src.reproduction import dataset as dsmod
from src.reproduction.config import ReproductionConfig
from src.reproduction.metrics import aggregate_seeds
from src.reproduction.mil_ids import MILBiLSTMIDS, bce_l2_loss
from src.reproduction.ppo_attacker import ActorCritic

PROJECT_ROOT = dsmod.PROJECT_ROOT
RAW_DIR = PROJECT_ROOT / "results" / "raw" / "candidates_stage2"
AGG_DIR = PROJECT_ROOT / "results" / "aggregate"
CKPT_DIR = PROJECT_ROOT / "results" / "checkpoints"

SCMA_TYPES = (1, 6)
BASELINE_SEED = 0
MAX_EPOCHS = 30
PATIENCE = 8


@torch.no_grad()
def infer_forward(model, slots, mask, static, device, batch_size=256):
    model.eval()
    scores = []
    n = len(slots)
    for start in range(0, n, batch_size):
        sl = slice(start, start + batch_size)
        s = torch.from_numpy(slots[sl]).to(device)
        m = torch.from_numpy(mask[sl]).to(device)
        st = torch.from_numpy(static[sl]).to(device)
        y_hat, _ = model(s, m, st)
        scores.append(y_hat.cpu().numpy())
    return np.concatenate(scores)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def train_baseline_train_val_only(scma_type, data, ids_cfg, device, seed=BASELINE_SEED):
    """Mirrors train_ids.py's train_one_seed EPOCH LOOP exactly (same model,
    same loss, same early-stopping-on-val-macro-F1@0.5 logic), but stops
    after loading best_state -- no threshold tuning / test evaluation, since
    this task must not touch test_n200/test_full. Same MAX_EPOCHS/PATIENCE
    budget as Part A's candidates for comparability."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MILBiLSTMIDS(
        input_dim=ids_cfg.input_dim, encoder_hidden=ids_cfg.encoder_hidden, encoder_out=ids_cfg.encoder_out,
        lstm_hidden_per_dir=ids_cfg.lstm_hidden_per_dir, static_dim=ids_cfg.static_dim,
        static_hidden=ids_cfg.static_hidden, dropout=ids_cfg.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ids_cfg.lr)

    ds = TensorDataset(
        torch.from_numpy(data["train"].slots), torch.from_numpy(data["train"].mask),
        torch.from_numpy(data["train"].static), torch.from_numpy(data["train"].labels).float(),
    )
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=ids_cfg.batch_size, shuffle=True, generator=gen)

    from sklearn.metrics import f1_score

    def macro_f1_at(y_true, y_score, thresh=0.5):
        y_pred = (y_score >= thresh).astype(int)
        return f1_score(y_true, y_pred, average="macro", zero_division=0)

    best_val_f1, best_state, best_epoch, epochs_no_improve = -1.0, None, -1, 0
    history = []
    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for slots, mask, static, labels in loader:
            slots, mask, static, labels = slots.to(device), mask.to(device), static.to(device), labels.to(device)
            logits, _ = model.logits(slots, mask, static)
            loss = bce_l2_loss(logits, labels, model, ids_cfg.l2_reg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        val_score = infer_forward(model, data["val"].slots, data["val"].mask, data["val"].static, device)
        val_f1 = macro_f1_at(data["val"].labels, val_score)
        history.append({"epoch": epoch, "train_loss": epoch_loss / max(n_batches, 1), "val_macro_f1@0.5": val_f1})
        if val_f1 > best_val_f1:
            best_val_f1, best_state, best_epoch, epochs_no_improve = val_f1, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return model, best_val_f1, best_epoch, len(history)


def load_candidate_checkpoint(candidate, scma_type, ids_cfg, device):
    ckpt_path = CKPT_DIR / f"stage2_{candidate}_scma{scma_type}_seed0.pt"
    model = build_model(candidate, ids_cfg).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stage2_crossgen] device={device} types={SCMA_TYPES}")

    cfg = ReproductionConfig()
    ids_cfg = cfg.ids
    manifest = dsmod.load_manifest()
    store = dsmod.SlotStore.get()
    manifest_idx = manifest.set_index("session_uid")

    primary_ckpt = torch.load(CKPT_DIR / "ppo_attacker.pt", map_location=device)
    primary_attacker = ActorCritic(cfg.attacker).to(device)
    primary_attacker.load_state_dict(primary_ckpt["model_state"])
    primary_attacker.eval()

    heldout_ckpt_path = CKPT_DIR / "ppo_attacker_heldout.pt"
    assert heldout_ckpt_path.exists(), "run train_heldout_attacker.py first"
    heldout_ckpt = torch.load(heldout_ckpt_path, map_location=device)
    heldout_attacker = ActorCritic(cfg.attacker).to(device)
    heldout_attacker.load_state_dict(heldout_ckpt["model_state"])
    heldout_attacker.eval()

    pools = dsmod.build_split_pools(manifest, seed=0)
    test_variants = dsmod.build_test_variants(manifest, seed=0)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    AGG_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for scma_type in SCMA_TYPES:
        print(f"[stage2_crossgen] === SCMA-{scma_type} ===")
        base_data = dsmod.build_type_dataset(manifest, store, pools, test_variants, scma_type, primary_attacker, device, cfg, dataset_seed=0)
        scalers = mgd.fit_scalers_for_type(manifest, store, pools, scma_type, primary_attacker, device, cfg)

        # held-out-generator VAL pool: same val benign pool + same scalers,
        # attack sessions' C_attack now predicted by the held-out policy.
        val_benign_bags, val_attack_heldout_bags = dsmod.generate_split_bags(
            pools["val"], manifest_idx, store, scma_type, heldout_attacker, device, cfg.attacker.c_adjust
        )
        heldout_val_data = dsmod._bags_to_arrays(val_benign_bags + val_attack_heldout_bags, scalers, seed=0)
        print(f"[stage2_crossgen] SCMA-{scma_type} held-out-generator val pool built: n={len(heldout_val_data.labels)} "
              f"(pos_rate={heldout_val_data.labels.mean():.3f}) vs in-distribution val n={len(base_data['val'].labels)}")

        models = {}
        for candidate in SCREEN_CANDIDATES:
            try:
                models[candidate] = load_candidate_checkpoint(candidate, scma_type, ids_cfg, device)
            except FileNotFoundError:
                print(f"[stage2_crossgen] WARNING: no seed0 checkpoint for {candidate}/SCMA-{scma_type}, skipping")

        t0 = time.time()
        baseline_model, base_val_f1, base_epoch, base_n_epochs = train_baseline_train_val_only(scma_type, base_data, ids_cfg, device)
        print(f"[stage2_crossgen] baseline MIL-BiLSTM IDS trained fresh (seed=0, train/val only) in {time.time()-t0:.1f}s, "
              f"best_epoch={base_epoch}, val_macro_f1@0.5={base_val_f1:.4f}")
        baseline_ckpt_path = CKPT_DIR / f"stage2_baseline_scma{scma_type}_seed0.pt"
        torch.save({"model_state": baseline_model.state_dict(), "scma_type": scma_type}, baseline_ckpt_path)
        models["baseline_mil_bilstm"] = baseline_model

        for model_name, model in models.items():
            model.eval()
            id_score = infer_forward(model, base_data["val"].slots, base_data["val"].mask, base_data["val"].static, device)
            id_metrics = full_eval(base_data["val"].labels, id_score)

            ho_score = infer_forward(model, heldout_val_data.slots, heldout_val_data.mask, heldout_val_data.static, device)
            ho_metrics = full_eval(heldout_val_data.labels, ho_score)

            result = {
                "model": model_name,
                "scma_type": scma_type,
                "n_params": n_params(model),
                "in_distribution": id_metrics,
                "held_out_generator": ho_metrics,
                "delta_macro_f1": id_metrics["macro_f1"] - ho_metrics["macro_f1"],
                "delta_recall": id_metrics["recall"] - ho_metrics["recall"],
                "delta_false_alarm_rate": ho_metrics["false_alarm_rate"] - id_metrics["false_alarm_rate"],
            }
            out_path = RAW_DIR / f"crossgen_{model_name}_scma{scma_type}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(
                f"[stage2_crossgen] {model_name} SCMA-{scma_type}: "
                f"in-dist macro_f1={id_metrics['macro_f1']:.4f} recall={id_metrics['recall']:.4f} far={id_metrics['false_alarm_rate']:.4f} | "
                f"held-out macro_f1={ho_metrics['macro_f1']:.4f} recall={ho_metrics['recall']:.4f} far={ho_metrics['false_alarm_rate']:.4f} | "
                f"delta_macro_f1={result['delta_macro_f1']:+.4f} delta_recall={result['delta_recall']:+.4f} -> {out_path}"
            )
            rows.append(result)

    # flatten to a summary CSV
    flat_rows = []
    for r in rows:
        flat_rows.append({
            "model": r["model"], "scma_type": r["scma_type"], "n_params": r["n_params"],
            "id_macro_f1": r["in_distribution"]["macro_f1"], "id_recall": r["in_distribution"]["recall"],
            "id_precision": r["in_distribution"]["precision"], "id_false_alarm_rate": r["in_distribution"]["false_alarm_rate"],
            "id_pr_auc": r["in_distribution"].get("pr_auc"), "id_balanced_accuracy": r["in_distribution"]["balanced_accuracy"],
            "ho_macro_f1": r["held_out_generator"]["macro_f1"], "ho_recall": r["held_out_generator"]["recall"],
            "ho_precision": r["held_out_generator"]["precision"], "ho_false_alarm_rate": r["held_out_generator"]["false_alarm_rate"],
            "ho_pr_auc": r["held_out_generator"].get("pr_auc"), "ho_balanced_accuracy": r["held_out_generator"]["balanced_accuracy"],
            "delta_macro_f1": r["delta_macro_f1"], "delta_recall": r["delta_recall"], "delta_false_alarm_rate": r["delta_false_alarm_rate"],
        })
    df = pd.DataFrame(flat_rows)
    out_csv = AGG_DIR / "candidates_stage2_crossgen.csv"
    df.to_csv(out_csv, index=False)
    print(f"[stage2_crossgen] wrote summary -> {out_csv}")
    print("[stage2_crossgen] Part B done.")


if __name__ == "__main__":
    main()
