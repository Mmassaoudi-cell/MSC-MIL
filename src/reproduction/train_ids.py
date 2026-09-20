"""Trains MILBiLSTMIDS (Eq. 3-9, BCE+L2 loss Eq. 10/11) for each of the 6 SCMA
types x 5 seeds (0-4), early-stopping on VAL macro-F1, then evaluates on:
  (a) the paper-matched N=200 test subsample (180 benign + 20 attack), and
  (b) the full eligible test pool (7,945 benign-eligible + ~883 attack, 90/10),
using metrics.binary_metrics. Saves per-seed JSON to
results/raw/reproduction/scma{TYPE}_seed{SEED}.json and aggregates across
seeds (mean/std/median/95% CI) into results/aggregate/reproduction_table2.csv.

Alert threshold tau (config.py: IDSConfig.alert_threshold, "ASSUMPTION default,
tuned on validation") is tuned per-seed by a small grid search over VAL macro-F1
(documented resolution of the paper's "tuned on the validation dataset"
language, Sec. V/Table I footnote, which never specifies a tuning method --
we use a coarse threshold grid rather than Optuna here since the search space
is 1-dimensional and a grid is exact/cheap).

Dataset construction (build_type_dataset) is run ONCE per SCMA type and reused
identically across all 5 seeds -- only model weight init / minibatch shuffling
differs per seed, isolating the source of seed-to-seed variance to the IDS
training itself (documented choice, consistent with the fixed dataset_seed=0
used throughout dataset.py).
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import dataset as dsmod
from .config import ReproductionConfig
from .metrics import aggregate_seeds, binary_metrics
from .mil_ids import MILBiLSTMIDS, bce_l2_loss
from .ppo_attacker import ActorCritic

PROJECT_ROOT = dsmod.PROJECT_ROOT
RAW_DIR = PROJECT_ROOT / "results" / "raw" / "reproduction"
AGG_DIR = PROJECT_ROOT / "results" / "aggregate"
CKPT_PATH = PROJECT_ROOT / "results" / "checkpoints" / "ppo_attacker.pt"

THRESH_GRID = np.round(np.arange(0.10, 0.91, 0.05), 2)


def make_loader(data: dsmod.TypeSplitData, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(data.slots), torch.from_numpy(data.mask), torch.from_numpy(data.static), torch.from_numpy(data.labels).float()
    )
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen if shuffle else None)


@torch.no_grad()
def infer(model: MILBiLSTMIDS, data: dsmod.TypeSplitData, device, batch_size: int = 256):
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


def macro_f1_at(y_true, y_score, thresh) -> float:
    from sklearn.metrics import f1_score

    y_pred = (y_score >= thresh).astype(int)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def tune_threshold(y_true, y_score) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in THRESH_GRID:
        f1 = macro_f1_at(y_true, y_score, t)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def train_one_seed(scma_type: int, seed: int, data: dict[str, dsmod.TypeSplitData], cfg: ReproductionConfig, device, max_epochs_override=None):
    ids_cfg = cfg.ids
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MILBiLSTMIDS(
        input_dim=ids_cfg.input_dim,
        encoder_hidden=ids_cfg.encoder_hidden,
        encoder_out=ids_cfg.encoder_out,
        lstm_hidden_per_dir=ids_cfg.lstm_hidden_per_dir,
        static_dim=ids_cfg.static_dim,
        static_hidden=ids_cfg.static_hidden,
        dropout=ids_cfg.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ids_cfg.lr)

    train_loader = make_loader(data["train"], ids_cfg.batch_size, shuffle=True, seed=seed)

    max_epochs = max_epochs_override or ids_cfg.max_epochs
    patience = ids_cfg.early_stopping_patience
    best_val_f1 = -1.0
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for slots, mask, static, labels in train_loader:
            slots, mask, static, labels = slots.to(device), mask.to(device), static.to(device), labels.to(device)
            logits, _ = model.logits(slots, mask, static)
            loss = bce_l2_loss(logits, labels, model, ids_cfg.l2_reg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        val_score = infer(model, data["val"], device)
        val_f1 = macro_f1_at(data["val"].labels, val_score, 0.5)
        history.append({"epoch": epoch, "train_loss": epoch_loss / max(n_batches, 1), "val_macro_f1@0.5": val_f1})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break

    model.load_state_dict(best_state)
    val_score = infer(model, data["val"], device)
    tuned_thresh = tune_threshold(data["val"].labels, val_score)

    n200_score = infer(model, data["test_n200"], device)
    full_score = infer(model, data["test_full"], device)
    n200_pred = (n200_score >= tuned_thresh).astype(int)
    full_pred = (full_score >= tuned_thresh).astype(int)

    result = {
        "scma_type": scma_type,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "tuned_threshold": tuned_thresh,
        "train_size": int(len(data["train"].labels)),
        "train_pos_rate": float(data["train"].labels.mean()),
        "val_size": int(len(data["val"].labels)),
        "epochs_trained": len(history),
        "test_n200": binary_metrics(data["test_n200"].labels, n200_pred, n200_score),
        "test_full": binary_metrics(data["test_full"].labels, full_pred, full_score),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--types", type=str, default="1,2,3,4,5,6")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    types = [int(x) for x in args.types.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    cfg = ReproductionConfig()
    if args.smoke:
        args.max_epochs = args.max_epochs or 3
        cfg.ids.early_stopping_patience = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_ids] device={device}  types={types}  seeds={seeds}")

    manifest = dsmod.load_manifest()
    store = dsmod.SlotStore.get()

    ckpt = torch.load(CKPT_PATH, map_location=device)
    attacker = ActorCritic(cfg.attacker).to(device)
    attacker.load_state_dict(ckpt["model_state"])
    attacker.eval()

    pools = dsmod.build_split_pools(manifest, seed=0)
    test_variants = dsmod.build_test_variants(manifest, seed=0)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    AGG_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for scma_type in types:
        t0 = time.time()
        data = dsmod.build_type_dataset(manifest, store, pools, test_variants, scma_type, attacker, device, cfg, dataset_seed=0)
        print(f"[train_ids] SCMA-{scma_type} dataset built in {time.time()-t0:.1f}s "
              f"(train={len(data['train'].labels)}, val={len(data['val'].labels)}, "
              f"n200={len(data['test_n200'].labels)}, full={len(data['test_full'].labels)})")
        for seed in seeds:
            t0 = time.time()
            result = train_one_seed(scma_type, seed, data, cfg, device, max_epochs_override=args.max_epochs)
            dt = time.time() - t0
            out_path = RAW_DIR / f"scma{scma_type}_seed{seed}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            n200 = result["test_n200"]
            print(
                f"[train_ids] SCMA-{scma_type} seed={seed} done in {dt:.1f}s "
                f"best_epoch={result['best_epoch']} val_f1={result['best_val_macro_f1']:.3f} "
                f"N200: P={n200['precision']:.3f} R={n200['recall']:.3f} F1={n200['f1']:.3f} Acc={n200['accuracy']:.3f} "
                f"-> {out_path}"
            )
            all_results.append(result)

    # ---- aggregate across seeds -> results/aggregate/reproduction_table2.csv ----
    rows = []
    for scma_type in types:
        for variant in ["test_n200", "test_full"]:
            metric_dicts = [r[variant] for r in all_results if r["scma_type"] == scma_type]
            if not metric_dicts:
                continue
            agg = aggregate_seeds(metric_dicts)
            for metric, stats in agg.items():
                rows.append({"scma_type": scma_type, "metric": metric, "test_variant": variant, **stats})
    agg_df = pd.DataFrame(rows)
    agg_path = AGG_DIR / "reproduction_table2.csv"
    if not agg_df.empty:
        agg_df = agg_df[["scma_type", "metric", "mean", "std", "median", "ci95_lo", "ci95_hi", "n", "test_variant"]]
        agg_df.to_csv(agg_path, index=False)
        print(f"[train_ids] wrote aggregate -> {agg_path}  ({len(agg_df)} rows)")


if __name__ == "__main__":
    main()
