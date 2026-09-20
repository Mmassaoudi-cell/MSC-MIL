"""Table III baselines: IsolationForest, OneClassSVM (RBF), LocalOutlierFactor,
and a small MLP reconstruction autoencoder -- all trained UNSUPERVISED on
TRAIN-BENIGN-ONLY tabular session-summary features, thresholded on an anomaly
score, evaluated on SCMA-1 (per SOURCE_PAPER_AUDIT.md sec.8's inference that
Table III's "Proposed IDS" row matches Table II's SCMA-1 row exactly, so
Table III as a whole is read as SCMA-1-only).

Feature vector: the SAME 7-dim per-session summary used for ADASYN in
dataset.py (dataset.summary_vec: [e_demand_kwh, t_avail_h(~duration_hours),
t_start_h, t_end_h, mean/std/max active-slot power]) -- a natural "flattened
per-session tabular summary" consistent with how these baselines are
non-sequential/non-MIL (Sec. Baselines, Table III).

Hyperparameters the paper never specifies for these baselines (library
defaults / documented assumptions, since SOURCE_PAPER_AUDIT.md sec. "Missing"
explicitly flags "whether Table III's benchmarks were tuned/optimized at all,
or run with library defaults" as unknown):
  - IsolationForest: n_estimators=100 (sklearn default), contamination='auto'.
  - OneClassSVM: kernel='rbf', gamma='scale' (sklearn default), nu=0.1
    ASSUMPTION (paper never gives nu; 0.1 matches the ~10% attack prevalence
    used throughout this reproduction's train/val/test composition).
  - LocalOutlierFactor: n_neighbors=20 (sklearn default), novelty=True
    (required for scoring held-out data).
  - MLP autoencoder: 7 -> 16 -> 4 -> 16 -> 7 (bottleneck=4), ReLU, Adam
    lr=1e-3, 150 epochs, MSE reconstruction loss -- architecture/training
    hyperparameters are all ASSUMPTIONs (paper gives no architecture detail
    beyond "Deep MLP Auto-Encoder").

All 4 models are fit once per seed on TRAIN BENIGN summary vectors (a
StandardScaler is also fit on TRAIN BENIGN only, reused for val/test).
IsolationForest and the MLP autoencoder have real seed-to-seed randomness
(random_state / weight init); OneClassSVM and LOF are deterministic given
fixed training data, so their 5-seed std is expected to be ~0 -- this is
reported honestly, not forced to vary.

Anomaly threshold tau is tuned per (model, seed) on VAL macro-F1 via a grid
over score quantiles (same rationale as train_ids.py: the paper says "tuned
on the validation dataset" without a method).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from . import dataset as dsmod
from .config import ReproductionConfig
from .metrics import aggregate_seeds, binary_metrics
from .ppo_attacker import ActorCritic

PROJECT_ROOT = dsmod.PROJECT_ROOT
RAW_DIR = PROJECT_ROOT / "results" / "raw" / "reproduction"
AGG_DIR = PROJECT_ROOT / "results" / "aggregate"
CKPT_PATH = PROJECT_ROOT / "results" / "checkpoints" / "ppo_attacker.pt"

SCMA_TYPE = 1  # Table III == SCMA-1 only (documented inference, see module docstring)
THRESH_QUANTILES = np.linspace(0.50, 0.995, 40)


class MLPAutoencoder(nn.Module):
    def __init__(self, in_dim: int = dsmod.SUMMARY_DIM, bottleneck: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, 16), nn.ReLU(), nn.Linear(16, bottleneck), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(bottleneck, 16), nn.ReLU(), nn.Linear(16, in_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def bags_to_summary(bags: list[dsmod.Bag]) -> np.ndarray:
    return np.stack([dsmod.summary_vec(b.sess, b.slots, b.active) for b in bags]).astype(np.float32)


def tune_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Grid-search a raw anomaly-score threshold over VAL score quantiles,
    maximizing macro-F1 (see module docstring: paper's 'tuned on validation'
    is undefined, so we use a quantile grid rather than a fixed rule)."""
    best_t, best_f1 = float(np.median(scores)), -1.0
    for q in THRESH_QUANTILES:
        t = float(np.quantile(scores, q))
        pred = (scores >= t).astype(int)
        f1 = f1_score(y_true, pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def train_mlp_autoencoder(X_train: np.ndarray, seed: int, epochs: int, device) -> MLPAutoencoder:
    torch.manual_seed(seed)
    model = MLPAutoencoder(in_dim=X_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X_t = torch.from_numpy(X_train).to(device)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        recon = model(X_t)
        loss = nn.functional.mse_loss(recon, X_t)
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def mlp_recon_error(model: MLPAutoencoder, X: np.ndarray, device) -> np.ndarray:
    model.eval()
    X_t = torch.from_numpy(X).to(device)
    recon = model(X_t)
    err = ((recon - X_t) ** 2).mean(dim=1)
    return err.cpu().numpy()


def evaluate(y_true, scores, thresh):
    pred = (scores >= thresh).astype(int)
    return binary_metrics(y_true, pred, scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--mlp-epochs", type=int, default=150)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    if args.smoke:
        args.mlp_epochs = 10
        seeds = seeds[:1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[baselines_table3] device={device} seeds={seeds} SCMA-{SCMA_TYPE}")

    manifest = dsmod.load_manifest()
    store = dsmod.SlotStore.get()
    manifest_idx = manifest.set_index("session_uid")

    ckpt = torch.load(CKPT_PATH, map_location=device)
    cfg = ReproductionConfig()
    attacker = ActorCritic(cfg.attacker).to(device)
    attacker.load_state_dict(ckpt["model_state"])
    attacker.eval()

    pools = dsmod.build_split_pools(manifest, seed=0)
    test_variants = dsmod.build_test_variants(manifest, seed=0)
    c_adjust = cfg.attacker.c_adjust

    t0 = time.time()
    train_benign, train_attack = dsmod.generate_split_bags(pools["train"], manifest_idx, store, SCMA_TYPE, attacker, device, c_adjust)
    val_benign, val_attack = dsmod.generate_split_bags(pools["val"], manifest_idx, store, SCMA_TYPE, attacker, device, c_adjust)
    n200_benign, n200_attack = dsmod.generate_split_bags(test_variants["test_n200"], manifest_idx, store, SCMA_TYPE, attacker, device, c_adjust)
    full_benign, full_attack = dsmod.generate_split_bags(test_variants["test_full"], manifest_idx, store, SCMA_TYPE, attacker, device, c_adjust)
    print(f"[baselines_table3] bags built in {time.time()-t0:.1f}s "
          f"(train_benign={len(train_benign)}, val={len(val_benign)}+{len(val_attack)}, "
          f"n200={len(n200_benign)}+{len(n200_attack)}, full={len(full_benign)}+{len(full_attack)})")

    Xtr_benign = bags_to_summary(train_benign)  # UNSUPERVISED: fit on benign only
    scaler = StandardScaler().fit(Xtr_benign)
    Xtr_benign_s = scaler.transform(Xtr_benign)

    def split_xy(benign_bags, attack_bags):
        Xb = bags_to_summary(benign_bags)
        Xa = bags_to_summary(attack_bags)
        X = np.concatenate([Xb, Xa], axis=0)
        y = np.concatenate([np.zeros(len(Xb), dtype=int), np.ones(len(Xa), dtype=int)])
        return scaler.transform(X), y

    Xval, yval = split_xy(val_benign, val_attack)
    Xn200, yn200 = split_xy(n200_benign, n200_attack)
    Xfull, yfull = split_xy(full_benign, full_attack)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    AGG_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {name: [] for name in ["isolation_forest", "one_class_svm", "local_outlier_factor", "mlp_autoencoder"]}

    for seed in seeds:
        t0 = time.time()
        # --- IsolationForest ---
        iso = IsolationForest(n_estimators=100, contamination="auto", random_state=seed)
        iso.fit(Xtr_benign_s)
        score_val = -iso.decision_function(Xval)
        thresh = tune_threshold(yval, score_val)
        res = {
            "model": "isolation_forest", "seed": seed, "tuned_threshold": thresh,
            "test_n200": evaluate(yn200, -iso.decision_function(Xn200), thresh),
            "test_full": evaluate(yfull, -iso.decision_function(Xfull), thresh),
        }
        all_results["isolation_forest"].append(res)
        _save(res)

        # --- OneClassSVM (RBF, nu=0.1 ASSUMPTION) ---
        ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
        ocsvm.fit(Xtr_benign_s)  # deterministic -- seed has no effect (documented)
        score_val = -ocsvm.decision_function(Xval)
        thresh = tune_threshold(yval, score_val)
        res = {
            "model": "one_class_svm", "seed": seed, "tuned_threshold": thresh,
            "test_n200": evaluate(yn200, -ocsvm.decision_function(Xn200), thresh),
            "test_full": evaluate(yfull, -ocsvm.decision_function(Xfull), thresh),
        }
        all_results["one_class_svm"].append(res)
        _save(res)

        # --- LocalOutlierFactor (novelty=True, deterministic) ---
        lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
        lof.fit(Xtr_benign_s)
        score_val = -lof.decision_function(Xval)
        thresh = tune_threshold(yval, score_val)
        res = {
            "model": "local_outlier_factor", "seed": seed, "tuned_threshold": thresh,
            "test_n200": evaluate(yn200, -lof.decision_function(Xn200), thresh),
            "test_full": evaluate(yfull, -lof.decision_function(Xfull), thresh),
        }
        all_results["local_outlier_factor"].append(res)
        _save(res)

        # --- MLP autoencoder ---
        ae = train_mlp_autoencoder(Xtr_benign_s.astype(np.float32), seed, args.mlp_epochs, device)
        score_val = mlp_recon_error(ae, Xval.astype(np.float32), device)
        thresh = tune_threshold(yval, score_val)
        res = {
            "model": "mlp_autoencoder", "seed": seed, "tuned_threshold": thresh,
            "test_n200": evaluate(yn200, mlp_recon_error(ae, Xn200.astype(np.float32), device), thresh),
            "test_full": evaluate(yfull, mlp_recon_error(ae, Xfull.astype(np.float32), device), thresh),
        }
        all_results["mlp_autoencoder"].append(res)
        _save(res)

        print(f"[baselines_table3] seed={seed} done in {time.time()-t0:.1f}s")

    rows = []
    for name, results in all_results.items():
        for variant in ["test_n200", "test_full"]:
            metric_dicts = [r[variant] for r in results]
            agg = aggregate_seeds(metric_dicts)
            for metric, stats in agg.items():
                rows.append({"model": name, "metric": metric, "test_variant": variant, **stats})
    agg_df = pd.DataFrame(rows)
    agg_path = AGG_DIR / "reproduction_table3.csv"
    agg_df = agg_df[["model", "metric", "mean", "std", "median", "ci95_lo", "ci95_hi", "n", "test_variant"]]
    agg_df.to_csv(agg_path, index=False)
    print(f"[baselines_table3] wrote aggregate -> {agg_path} ({len(agg_df)} rows)")


def _save(res: dict):
    out_path = RAW_DIR / f"baseline_{res['model']}_seed{res['seed']}.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
