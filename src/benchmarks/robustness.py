"""Part 3 (Master Prompt Sec.20-21): robustness / generalization experiments
for the frozen MSC-MIL model vs. CatBoost (top tree benchmark) and
SOURCE_METHOD_REPRODUCTION.

1. Feature/sensor noise robustness: Gaussian noise (std = 5%/15%/30% of each
   feature's TRAIN-set std) injected onto test_full's power/energy features
   at EVAL TIME ONLY (no retraining) -- simulates sensor measurement error.
   Noise is applied to the underlying (slots power, static energy-demand)
   tensors BEFORE any downstream feature derivation, so MSC-MIL/repro (which
   consume the tensors directly) and CatBoost (which consumes
   data_utils.flatten_features() derived from the SAME tensors) see a
   consistent perturbation.
2. Missing-data robustness: randomly mask 10%/25%/50% of a session's
   otherwise-ACTIVE slots at eval time (set slot power to the dataset's own
   empirical "inactive" fill value + mask=False), simulating telemetry gaps.
3. Low-data generalization curve: retrain MSC-MIL and CatBoost on reduced
   TRAIN fractions (10/25/50/100% of the existing capped train pool,
   stratified by label), 3 seeds each, evaluated on the SAME full test_full.

Prereqs (must already exist):
  - results/checkpoints/final/msc_mil_final_scma1_seed{0-4}.pt (final_msc_train.py)
  - results/checkpoints/reproduction_for_robustness/repro_scma1_seed{0-4}.pt
    (repro_ckpt_for_robustness.py -- SOURCE_METHOD_REPRODUCTION never had
    persisted weights before this task)
  - results/aggregate/benchmark_val_tuning_classical_tree.json (CatBoost's
    val-tuned params, reused verbatim, not retuned)

Writes results/raw/robustness/*.json and results/aggregate/robustness_results.csv.
"""
from __future__ import annotations

import copy
import json
import time

import numpy as np
import pandas as pd
import torch
import yaml

from src.benchmarks.classical_tree import build_estimator
from src.benchmarks.data_utils import PROJECT_ROOT, flatten_features, get_device, load_scma_data
from src.candidates.msc_mil import MSCMIL
from src.candidates.smoke_test import infer_forward, train_msc
from src.reproduction import dataset as dsmod
from src.reproduction.config import IDSConfig
from src.reproduction.metrics import aggregate_seeds, binary_metrics, macro_f1_score
from src.reproduction.mil_ids import MILBiLSTMIDS

CONFIG_PATH = PROJECT_ROOT / "FINAL_MODEL_CONFIG.yaml"
RAW_DIR = PROJECT_ROOT / "results" / "raw" / "robustness"
AGG_DIR = PROJECT_ROOT / "results" / "aggregate"
CKPT_FINAL = PROJECT_ROOT / "results" / "checkpoints" / "final"
CKPT_REPRO = PROJECT_ROOT / "results" / "checkpoints" / "reproduction_for_robustness"
SCMA_TYPE = 1
SEEDS5 = (0, 1, 2, 3, 4)
NOISE_LEVELS = [0.0, 0.05, 0.15, 0.30]
MISSING_LEVELS = [0.0, 0.10, 0.25, 0.50]
LOWDATA_FRACS = [0.10, 0.25, 0.50, 1.00]
LOWDATA_SEEDS = (0, 1, 2)
METRIC_KEYS = ["macro_f1", "recall", "false_alarm_rate", "precision", "f1", "accuracy"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_msc_models(arch, device):
    models = {}
    for seed in SEEDS5:
        ck = torch.load(CKPT_FINAL / f"msc_mil_final_scma{SCMA_TYPE}_seed{seed}.pt", map_location=device)
        m = MSCMIL(
            input_dim=arch["input_dim"], encoder_hidden=arch["encoder_hidden"], encoder_out=arch["encoder_out"],
            lstm_hidden_per_dir=arch["lstm_hidden_per_dir"], static_dim=arch["static_dim"],
            static_hidden=arch["static_hidden"], dropout=arch["dropout"],
        ).to(device)
        m.load_state_dict(ck["model_state"])
        m.eval()
        models[seed] = m
    return models


def load_repro_models(device):
    ids_cfg = IDSConfig()
    models = {}
    for seed in SEEDS5:
        ck = torch.load(CKPT_REPRO / f"repro_scma{SCMA_TYPE}_seed{seed}.pt", map_location=device)
        m = MILBiLSTMIDS(
            input_dim=ids_cfg.input_dim, encoder_hidden=ids_cfg.encoder_hidden, encoder_out=ids_cfg.encoder_out,
            lstm_hidden_per_dir=ids_cfg.lstm_hidden_per_dir, static_dim=ids_cfg.static_dim,
            static_hidden=ids_cfg.static_hidden, dropout=ids_cfg.dropout,
        ).to(device)
        m.load_state_dict(ck["model_state"])
        m.eval()
        models[seed] = m
    return models


def fit_catboost_models(Xtr, ytr, params):
    models = {}
    for seed in SEEDS5:
        est = build_estimator("catboost", params, seed=seed)
        est.fit(Xtr, ytr)
        models[seed] = est
    return models


# ---------------------------------------------------------------------------
# Perturbations (eval time only -- copies, never mutate the cached dataset)
# ---------------------------------------------------------------------------
def perturb_noise(d, power_std, energy_std, severity, rng):
    d2 = copy.deepcopy(d)
    if severity > 0:
        active = d2.mask
        noise_p = rng.normal(0.0, severity * power_std, size=d2.slots[:, :, 0].shape).astype(np.float32)
        d2.slots[:, :, 0] = d2.slots[:, :, 0] + noise_p * active
        noise_e = rng.normal(0.0, severity * energy_std, size=d2.static[:, 2].shape).astype(np.float32)
        d2.static[:, 2] = d2.static[:, 2] + noise_e
    return d2


def compute_fill_value(d):
    inactive_vals = d.slots[:, :, 0][~d.mask]
    if inactive_vals.size == 0:
        return 0.0
    return float(inactive_vals.mean())


def perturb_missing(d, frac, rng, fill_value):
    d2 = copy.deepcopy(d)
    if frac > 0:
        for i in range(len(d2.labels)):
            active_idx = np.where(d2.mask[i])[0]
            if len(active_idx) == 0:
                continue
            n_mask = int(round(frac * len(active_idx)))
            if n_mask == 0:
                continue
            drop_idx = rng.choice(active_idx, size=n_mask, replace=False)
            d2.mask[i, drop_idx] = False
            d2.slots[i, drop_idx, 0] = fill_value
    return d2


def subsample_train(d, frac, seed):
    if frac >= 1.0:
        return d
    rng = np.random.default_rng(1_000_000 + seed)
    idx0 = np.where(d.labels == 0)[0]
    idx1 = np.where(d.labels == 1)[0]
    n0 = max(1, int(round(frac * len(idx0))))
    n1 = max(1, int(round(frac * len(idx1))))
    sel0 = rng.choice(idx0, size=n0, replace=False)
    sel1 = rng.choice(idx1, size=n1, replace=False)
    sel = np.sort(np.concatenate([sel0, sel1]))
    return dsmod.TypeSplitData(
        slots=d.slots[sel], mask=d.mask[sel], static=d.static[sel], labels=d.labels[sel],
        uids=[d.uids[i] for i in sel],
    )


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------
def eval_torch(model, d, device):
    score = infer_forward(model, d.slots, d.mask, d.static, device)
    pred = (score >= 0.5).astype(int)
    m = binary_metrics(d.labels, pred, score)
    m["macro_f1"] = macro_f1_score(d.labels, pred)
    return m


def eval_sklearn(est, X, y):
    proba = est.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    m = binary_metrics(y, pred, proba)
    m["macro_f1"] = macro_f1_score(y, pred)
    return m


# ---------------------------------------------------------------------------
def aggregate_records(records, group_keys):
    rows = []
    df_keys = sorted(set(tuple(r[k] for k in group_keys) for r in records))
    for key in df_keys:
        recs = [r for r in records if all(r[k] == v for k, v in zip(group_keys, key))]
        metric_dicts = [r["metrics"] for r in recs]
        agg = aggregate_seeds(metric_dicts)
        for mk in METRIC_KEYS:
            if mk in agg:
                row = dict(zip(group_keys, key))
                row["metric"] = mk
                row["mean"] = agg[mk]["mean"]
                row["std"] = agg[mk]["std"]
                row["n_seeds"] = agg[mk]["n"]
                rows.append(row)
    return rows


def main():
    device = get_device()
    print(f"[robustness] device={device}")
    with open(CONFIG_PATH) as f:
        final_cfg = yaml.safe_load(f)
    arch = final_cfg["architecture"]
    hp = final_cfg["hyperparameters"]
    train_cfg = final_cfg["training"]

    data = load_scma_data(SCMA_TYPE, device)
    train, test_full = data["train"], data["test_full"]

    with open(AGG_DIR / "benchmark_val_tuning_classical_tree.json") as f:
        tuned = json.load(f)
    catboost_params = tuned["catboost"]["best_params"]

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    msc_models = load_msc_models(arch, device)
    repro_models = load_repro_models(device)
    Xtr_full = flatten_features(train)
    ytr_full = train.labels
    catboost_models = fit_catboost_models(Xtr_full, ytr_full, catboost_params)
    print("[robustness] loaded MSC-MIL (5 seeds), SOURCE_METHOD_REPRODUCTION (5 seeds), fit CatBoost (5 seeds)")

    power_std = float(train.slots[:, :, 0][train.mask].std())
    energy_std = float(train.static[:, 2].std())
    fill_value = compute_fill_value(train)
    print(f"[robustness] noise scale refs: power_std={power_std:.4f} energy_std={energy_std:.4f} "
          f"missing fill_value={fill_value:.4f}")

    # ---------------- 1. Noise robustness ----------------
    noise_records = []
    for severity in NOISE_LEVELS:
        for seed in SEEDS5:
            rng = np.random.default_rng(int(severity * 1000) * 100 + seed)
            d_noisy = perturb_noise(test_full, power_std, energy_std, severity, rng)
            Xte_noisy = flatten_features(d_noisy)

            m_msc = eval_torch(msc_models[seed], d_noisy, device)
            noise_records.append({"method": "MSC-MIL", "severity": severity, "seed": seed, "metrics": m_msc})

            m_repro = eval_torch(repro_models[seed], d_noisy, device)
            noise_records.append({"method": "SOURCE_METHOD_REPRODUCTION", "severity": severity, "seed": seed, "metrics": m_repro})

            m_cb = eval_sklearn(catboost_models[seed], Xte_noisy, d_noisy.labels)
            noise_records.append({"method": "CatBoost", "severity": severity, "seed": seed, "metrics": m_cb})
        print(f"[robustness] noise severity={severity}: done (5 seeds x 3 methods)")

    with open(RAW_DIR / "noise_robustness_records.json", "w") as f:
        json.dump(noise_records, f, indent=2, default=str)

    # ---------------- 2. Missing-data robustness ----------------
    missing_records = []
    for frac in MISSING_LEVELS:
        for seed in SEEDS5:
            rng = np.random.default_rng(500_000 + int(frac * 1000) * 100 + seed)
            d_missing = perturb_missing(test_full, frac, rng, fill_value)
            Xte_missing = flatten_features(d_missing)

            m_msc = eval_torch(msc_models[seed], d_missing, device)
            missing_records.append({"method": "MSC-MIL", "missing_frac": frac, "seed": seed, "metrics": m_msc})

            m_repro = eval_torch(repro_models[seed], d_missing, device)
            missing_records.append({"method": "SOURCE_METHOD_REPRODUCTION", "missing_frac": frac, "seed": seed, "metrics": m_repro})

            m_cb = eval_sklearn(catboost_models[seed], Xte_missing, d_missing.labels)
            missing_records.append({"method": "CatBoost", "missing_frac": frac, "seed": seed, "metrics": m_cb})
        print(f"[robustness] missing_frac={frac}: done (5 seeds x 3 methods)")

    with open(RAW_DIR / "missing_robustness_records.json", "w") as f:
        json.dump(missing_records, f, indent=2, default=str)

    # ---------------- 3. Low-data generalization curve ----------------
    ids_cfg_base = IDSConfig(
        lr=hp["learning_rate"], l2_reg=hp["l2_reg"], batch_size=hp["batch_size"],
        n_slots=arch["n_slots"], input_dim=arch["input_dim"], encoder_hidden=arch["encoder_hidden"],
        encoder_out=arch["encoder_out"], lstm_hidden_per_dir=arch["lstm_hidden_per_dir"],
        static_dim=arch["static_dim"], static_hidden=arch["static_hidden"], dropout=arch["dropout"],
        max_epochs=train_cfg["max_epochs"], early_stopping_patience=train_cfg["early_stopping_patience"],
        alert_threshold=final_cfg["alert_threshold"],
    )
    beta, tau = hp["supcon_beta"], hp["supcon_temperature"]

    lowdata_records = []
    for frac in LOWDATA_FRACS:
        for seed in LOWDATA_SEEDS:
            out_path = RAW_DIR / f"lowdata_msc_mil_frac{int(frac*100)}_seed{seed}.json"
            train_sub = subsample_train(train, frac, seed)
            print(f"[robustness] lowdata frac={frac} seed={seed}: train_n={len(train_sub.labels)} "
                  f"(pos_rate={train_sub.labels.mean():.3f})")

            if out_path.exists():
                with open(out_path) as f:
                    cached = json.load(f)
                m_msc = cached["metrics"]
            else:
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = MSCMIL(
                    input_dim=ids_cfg_base.input_dim, encoder_hidden=ids_cfg_base.encoder_hidden,
                    encoder_out=ids_cfg_base.encoder_out, lstm_hidden_per_dir=ids_cfg_base.lstm_hidden_per_dir,
                    static_dim=ids_cfg_base.static_dim, static_hidden=ids_cfg_base.static_hidden,
                    dropout=ids_cfg_base.dropout,
                ).to(device)
                data_sub = {"train": train_sub, "val": data["val"]}
                t0 = time.time()
                history, best_state, best_epoch, best_val_f1, converged, fail_reason = train_msc(
                    model, ids_cfg_base, data_sub, device, seed, beta=beta, tau=tau,
                    max_epochs=ids_cfg_base.max_epochs, patience=ids_cfg_base.early_stopping_patience,
                )
                dt = time.time() - t0
                if converged and best_state is not None:
                    model.load_state_dict(best_state)
                    m_msc = eval_torch(model, test_full, device)
                else:
                    m_msc = None
                with open(out_path, "w") as f:
                    json.dump({"frac": frac, "seed": seed, "converged": converged, "fail_reason": fail_reason,
                               "train_time_s": dt, "epochs_trained": len(history), "best_epoch": best_epoch,
                               "metrics": m_msc}, f, indent=2, default=str)
                print(f"[robustness] lowdata MSC-MIL frac={frac} seed={seed}: "
                      f"{'test_full macro_f1=' + format(m_msc['macro_f1'], '.4f') if m_msc else 'FAILED'} time={dt:.1f}s")
            if m_msc is not None:
                lowdata_records.append({"method": "MSC-MIL", "train_frac": frac, "seed": seed, "metrics": m_msc})

            # CatBoost on the same subsampled train
            Xtr_sub = flatten_features(train_sub)
            ytr_sub = train_sub.labels
            est = build_estimator("catboost", catboost_params, seed=seed)
            est.fit(Xtr_sub, ytr_sub)
            Xte_full = flatten_features(test_full)
            m_cb = eval_sklearn(est, Xte_full, test_full.labels)
            lowdata_records.append({"method": "CatBoost", "train_frac": frac, "seed": seed, "metrics": m_cb})
            print(f"[robustness] lowdata CatBoost frac={frac} seed={seed}: test_full macro_f1={m_cb['macro_f1']:.4f}")

    with open(RAW_DIR / "lowdata_records.json", "w") as f:
        json.dump(lowdata_records, f, indent=2, default=str)

    # ---------------- aggregate everything ----------------
    rows = []
    for r in aggregate_records(noise_records, ["method", "severity"]):
        r["experiment"] = "noise"
        rows.append(r)
    for r in aggregate_records(missing_records, ["method", "missing_frac"]):
        r["experiment"] = "missing_data"
        rows.append(r)
    for r in aggregate_records(lowdata_records, ["method", "train_frac"]):
        r["experiment"] = "lowdata_curve"
        rows.append(r)

    out_df = pd.DataFrame(rows)
    out_path = AGG_DIR / "robustness_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[robustness] wrote {out_path} ({len(out_df)} rows)")
    print("[robustness] done.")


if __name__ == "__main__":
    main()
