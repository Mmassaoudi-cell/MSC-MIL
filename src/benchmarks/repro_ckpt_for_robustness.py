"""Part 3 prerequisite: src/reproduction/train_ids.py never persisted trained
MILBiLSTMIDS weights to disk (only the resulting metrics JSON) -- there is no
existing checkpoint to load SOURCE_METHOD_REPRODUCTION's trained weights from
for the robustness (noise/missing-data) eval-time perturbation experiments.

This script retrains SOURCE_METHOD_REPRODUCTION for SCMA-1, 5 seeds, using the
IDENTICAL training code path as train_ids.py's train_one_seed (same
IDSConfig defaults: lr=1e-3, l2_reg=1e-4, batch_size=32, max_epochs=200,
patience=20 -- NOT the MSC-MIL 30/8 budget), the ONLY difference being that
the best-val-f1 state_dict is also saved to disk here so robustness.py can
load it. Evaluated at alert_threshold=0.5 (NOT the paper's own per-seed
val-tuned threshold used in REPRODUCTION_REPORT.md/train_ids.py) for
consistency with how every other method in this project's benchmark/ablation/
robustness suite is thresholded -- documented deviation, noted in
ROBUSTNESS_RESULTS.md.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from src.benchmarks.data_utils import PROJECT_ROOT, get_device, load_scma_data
from src.reproduction.config import ReproductionConfig
from src.reproduction.metrics import binary_metrics, macro_f1_score
from src.reproduction.mil_ids import MILBiLSTMIDS, bce_l2_loss
from torch.utils.data import DataLoader, TensorDataset

RAW_DIR = PROJECT_ROOT / "results" / "raw" / "robustness"
CKPT_DIR = PROJECT_ROOT / "results" / "checkpoints" / "reproduction_for_robustness"
SCMA_TYPE = 1
SEEDS = (0, 1, 2, 3, 4)


def make_loader(data, batch_size, seed):
    ds = TensorDataset(
        torch.from_numpy(data.slots), torch.from_numpy(data.mask), torch.from_numpy(data.static), torch.from_numpy(data.labels).float()
    )
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen)


@torch.no_grad()
def infer(model, data, device, batch_size=256):
    model.eval()
    scores = []
    n = len(data.labels)
    for start in range(0, n, batch_size):
        sl = slice(start, start + batch_size)
        slots = torch.from_numpy(data.slots[sl]).to(device)
        mask = torch.from_numpy(data.mask[sl]).to(device)
        static = torch.from_numpy(data.static[sl]).to(device)
        y_hat, _ = model(slots, mask, static)
        scores.append(y_hat.cpu().numpy())
    return np.concatenate(scores)


def main():
    device = get_device()
    cfg = ReproductionConfig()
    ids_cfg = cfg.ids
    data = load_scma_data(SCMA_TYPE, device)
    print(f"[repro_ckpt_for_robustness] device={device} train={len(data['train'].labels)}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        ckpt_path = CKPT_DIR / f"repro_scma{SCMA_TYPE}_seed{seed}.pt"
        out_path = RAW_DIR / f"repro_ckpt_train_seed{seed}.json"
        if ckpt_path.exists() and out_path.exists():
            print(f"[repro_ckpt_for_robustness] SKIP (exists) seed={seed}")
            continue
        print(f"[repro_ckpt_for_robustness] === seed={seed} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = MILBiLSTMIDS(
            input_dim=ids_cfg.input_dim, encoder_hidden=ids_cfg.encoder_hidden, encoder_out=ids_cfg.encoder_out,
            lstm_hidden_per_dir=ids_cfg.lstm_hidden_per_dir, static_dim=ids_cfg.static_dim,
            static_hidden=ids_cfg.static_hidden, dropout=ids_cfg.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=ids_cfg.lr)
        loader = make_loader(data["train"], ids_cfg.batch_size, seed)

        best_val_f1 = -1.0
        best_state = None
        best_epoch = -1
        epochs_no_improve = 0
        t0 = time.time()
        for epoch in range(ids_cfg.max_epochs):
            model.train()
            for slots, mask, static, labels in loader:
                slots, mask, static, labels = slots.to(device), mask.to(device), static.to(device), labels.to(device)
                logits, _ = model.logits(slots, mask, static)
                loss = bce_l2_loss(logits, labels, model, ids_cfg.l2_reg)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            val_score = infer(model, data["val"], device)
            val_pred = (val_score >= 0.5).astype(int)
            val_f1 = macro_f1_score(data["val"].labels, val_pred)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= ids_cfg.early_stopping_patience:
                break
        train_time_s = time.time() - t0
        model.load_state_dict(best_state)
        torch.save({"model_state": model.state_dict(), "scma_type": SCMA_TYPE, "seed": seed}, ckpt_path)

        # sanity eval @0.5 on test_full
        d = data["test_full"]
        score = infer(model, d, device)
        pred = (score >= 0.5).astype(int)
        m = binary_metrics(d.labels, pred, score)
        m["macro_f1"] = macro_f1_score(d.labels, pred)
        result = {"seed": seed, "best_epoch": best_epoch, "best_val_macro_f1@0.5": best_val_f1,
                  "epochs_trained": best_epoch + epochs_no_improve + 1, "train_time_s": train_time_s,
                  "test_full_metrics@0.5": m}
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"[repro_ckpt_for_robustness] seed={seed}: val_f1={best_val_f1:.4f} test_full_macro_f1={m['macro_f1']:.4f} "
              f"time={train_time_s:.1f}s -> {ckpt_path}")
    print("[repro_ckpt_for_robustness] done.")


if __name__ == "__main__":
    main()
