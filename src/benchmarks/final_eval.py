"""Part B: the SINGLE, FINAL, untouched TEST evaluation. Reads test_n200 and
test_full for SCMA-1 exactly once, for every method at once:
  - MSC-MIL (5 seeds, FINAL_MODEL_CONFIG.yaml, checkpoints from final_msc_train.py)
  - 4 deep benchmarks (5 seeds each, checkpoints from deep_models.py)
  - 7 classical/tree/boosting benchmarks (val-tuned params from classical_tree.py;
    refit on TRAIN with 5 seeds where the model has any fit-time randomness,
    else 1 seed / std=0, honestly reported per the reproduction phase's own
    OneClassSVM/LOF precedent)
  - SOURCE_METHOD_REPRODUCTION (reused as-is from results/raw/reproduction/scma1_seed*.json
    and results/aggregate/reproduction_table2.csv -- NOT retrained)

Writes:
  - results/aggregate/final_benchmark_comparison.csv (method, test_variant, metric, mean, std, n_seeds)
  - results/raw/final_eval/all_methods_per_seed_metrics.json (full detail, for Part C's paired tests)
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from src.benchmarks.classical_tree import GRIDS, RANDOM_MODELS, fit_and_predict
from src.benchmarks.data_utils import PROJECT_ROOT, flatten_features, get_device, load_scma_data
from src.benchmarks.deep_models import build_model as build_deep_model
from src.benchmarks.deep_models import infer_forward as deep_infer_forward
from src.benchmarks.deep_models import n_params as deep_n_params
from src.candidates.msc_mil import MSCMIL
from src.candidates.smoke_test import infer_forward as msc_infer_forward
from src.candidates.smoke_test import n_params as msc_n_params
from src.reproduction.config import IDSConfig
from src.reproduction.metrics import aggregate_seeds, binary_metrics, macro_f1_score
from src.reproduction.mil_ids import MILBiLSTMIDS

AGG_DIR = PROJECT_ROOT / "results" / "aggregate"
RAW_DIR = PROJECT_ROOT / "results" / "raw" / "final_eval"
CKPT_FINAL = PROJECT_ROOT / "results" / "checkpoints" / "final"
CKPT_BENCH = PROJECT_ROOT / "results" / "checkpoints" / "benchmarks"
REPRO_RAW = PROJECT_ROOT / "results" / "raw" / "reproduction"

SCMA_TYPE = 1
SEEDS = (0, 1, 2, 3, 4)
DEEP_MODELS = ["mlp", "cnn1d", "plain_lstm_mil", "transformer_mil"]
TEST_VARIANTS = ["test_n200", "test_full"]

with open(PROJECT_ROOT / "FINAL_MODEL_CONFIG.yaml") as _f:
    import yaml
    _FINAL_CFG = yaml.safe_load(_f)


@torch.no_grad()
def measure_latency_ms(predict_fn, n_sessions, warmup=5):
    times = []
    for i in range(warmup + n_sessions):
        t0 = time.perf_counter()
        predict_fn()
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            times.append(dt)
    return float(np.mean(times))


def eval_torch_model(model, data, device):
    """Returns {test_variant: (metrics_dict_with_macro_f1, latency_ms)}."""
    out = {}
    for tv in TEST_VARIANTS:
        d = data[tv]
        score = deep_infer_forward(model, d.slots, d.mask, d.static, device)
        pred = (score >= 0.5).astype(int)
        m = binary_metrics(d.labels, pred, score)
        m["macro_f1"] = macro_f1_score(d.labels, pred)
        out[tv] = m
    # latency: batch=1 on test_full, 50 sessions
    d = data["test_full"]
    n = min(50, len(d.labels))
    idx = np.arange(n)

    def _one():
        j = idx[0]
        s = torch.from_numpy(d.slots[j:j+1]).to(device)
        mk = torch.from_numpy(d.mask[j:j+1]).to(device)
        st = torch.from_numpy(d.static[j:j+1]).to(device)
        model(s, mk, st)
        if device.type == "cuda":
            torch.cuda.synchronize()

    lat = measure_latency_ms(_one, n)
    return out, lat


def main():
    device = get_device()
    print(f"[final_eval] device={device}")
    data = load_scma_data(SCMA_TYPE, device)
    print(f"[final_eval] SCMA-{SCMA_TYPE} test_n200={len(data['test_n200'].labels)} test_full={len(data['test_full'].labels)}")

    per_seed_records = []  # rows: method, test_variant, seed, metric_dict, n_params, latency_ms

    # ---------------- MSC-MIL (final, frozen) ----------------
    arch = _FINAL_CFG["architecture"]
    for seed in SEEDS:
        ckpt_path = CKPT_FINAL / f"msc_mil_final_scma{SCMA_TYPE}_seed{seed}.pt"
        if not ckpt_path.exists():
            print(f"[final_eval] WARNING missing MSC-MIL checkpoint seed={seed}, skipping")
            continue
        ck = torch.load(ckpt_path, map_location=device)
        model = MSCMIL(
            input_dim=arch["input_dim"], encoder_hidden=arch["encoder_hidden"], encoder_out=arch["encoder_out"],
            lstm_hidden_per_dir=arch["lstm_hidden_per_dir"], static_dim=arch["static_dim"],
            static_hidden=arch["static_hidden"], dropout=arch["dropout"],
        ).to(device)
        model.load_state_dict(ck["model_state"])
        metrics_by_tv, lat = eval_torch_model(model, data, device)
        for tv, m in metrics_by_tv.items():
            per_seed_records.append({"method": "MSC-MIL", "test_variant": tv, "seed": seed, "metrics": m,
                                      "n_params": msc_n_params(model), "latency_ms": lat})
        print(f"[final_eval] MSC-MIL seed={seed}: test_full macro_f1={metrics_by_tv['test_full']['macro_f1']:.4f} "
              f"test_n200 macro_f1={metrics_by_tv['test_n200']['macro_f1']:.4f}")

    # ---------------- Deep benchmarks 8-11 ----------------
    ids_cfg = IDSConfig()
    for name in DEEP_MODELS:
        for seed in SEEDS:
            ckpt_path = CKPT_BENCH / f"{name}_scma{SCMA_TYPE}_seed{seed}.pt"
            if not ckpt_path.exists():
                print(f"[final_eval] WARNING missing {name} checkpoint seed={seed}, skipping")
                continue
            ck = torch.load(ckpt_path, map_location=device)
            model = build_deep_model(name, ids_cfg).to(device)
            model.load_state_dict(ck["model_state"])
            metrics_by_tv, lat = eval_torch_model(model, data, device)
            for tv, m in metrics_by_tv.items():
                per_seed_records.append({"method": name, "test_variant": tv, "seed": seed, "metrics": m,
                                          "n_params": deep_n_params(model), "latency_ms": lat})
        n_done = sum(1 for r in per_seed_records if r["method"] == name and r["test_variant"] == "test_full")
        print(f"[final_eval] {name}: {n_done}/5 seeds evaluated")

    # ---------------- SOURCE_METHOD_REPRODUCTION (reused, not retrained) ----------------
    repro_arch_cfg = IDSConfig()  # unmodified reproduction architecture
    repro_model = MILBiLSTMIDS(
        input_dim=repro_arch_cfg.input_dim, encoder_hidden=repro_arch_cfg.encoder_hidden,
        encoder_out=repro_arch_cfg.encoder_out, lstm_hidden_per_dir=repro_arch_cfg.lstm_hidden_per_dir,
        static_dim=repro_arch_cfg.static_dim, static_hidden=repro_arch_cfg.static_hidden,
        dropout=repro_arch_cfg.dropout,
    ).to(device)
    repro_n_params = sum(p.numel() for p in repro_model.parameters())
    d = data["test_full"]
    n = min(50, len(d.labels))

    def _one_repro():
        s = torch.from_numpy(d.slots[0:1]).to(device)
        mk = torch.from_numpy(d.mask[0:1]).to(device)
        st = torch.from_numpy(d.static[0:1]).to(device)
        repro_model(s, mk, st)
        if device.type == "cuda":
            torch.cuda.synchronize()

    repro_latency = measure_latency_ms(_one_repro, n)
    print(f"[final_eval] SOURCE_METHOD_REPRODUCTION: n_params={repro_n_params} "
          f"latency (fresh-arch measurement, weights irrelevant to wall-clock)={repro_latency:.4f}ms")

    for seed in SEEDS:
        p = REPRO_RAW / f"scma{SCMA_TYPE}_seed{seed}.json"
        if not p.exists():
            print(f"[final_eval] WARNING missing reproduction raw json seed={seed}")
            continue
        with open(p) as f:
            rj = json.load(f)
        for tv in TEST_VARIANTS:
            m = dict(rj[tv])  # accuracy/precision/recall/f1/false_alarm_rate/balanced_accuracy/tp/fp/tn/fn/pr_auc/roc_auc
            tp, fp, tn, fn = m["tp"], m["fp"], m["tn"], m["fn"]
            prec_neg = tn / max(tn + fn, 1)
            rec_neg = tn / max(tn + fp, 1)
            f1_neg = 2 * prec_neg * rec_neg / max(prec_neg + rec_neg, 1e-12) if (prec_neg + rec_neg) > 0 else 0.0
            m["macro_f1"] = (m["f1"] + f1_neg) / 2.0
            per_seed_records.append({"method": "SOURCE_METHOD_REPRODUCTION", "test_variant": tv, "seed": seed,
                                      "metrics": m, "n_params": repro_n_params, "latency_ms": repro_latency})

    # ---------------- Classical/tree/boosting benchmarks 1-7 ----------------
    with open(AGG_DIR / "benchmark_val_tuning_classical_tree.json") as f:
        tuned = json.load(f)
    Xtr = flatten_features(data["train"]); ytr = data["train"].labels
    Xtest = {tv: flatten_features(data[tv]) for tv in TEST_VARIANTS}
    for name, info in tuned.items():
        params = info["best_params"]
        has_rand = info["has_randomness"]
        seeds_to_run = list(SEEDS) if has_rand else [0]
        for seed in seeds_to_run:
            t_fit0 = time.time()
            est, _ = fit_and_predict(name, params, Xtr, ytr, Xtest["test_n200"], seed=seed)
            fit_time = time.time() - t_fit0
            # latency: single-sample predict_proba wall time, averaged
            x0 = Xtest["test_full"][0:1]

            def _one_cls():
                est.predict_proba(x0)

            lat = measure_latency_ms(_one_cls, 50)
            for tv in TEST_VARIANTS:
                proba = est.predict_proba(Xtest[tv])[:, 1]
                pred = (proba >= 0.5).astype(int)
                m = binary_metrics(data[tv].labels, pred, proba)
                m["macro_f1"] = macro_f1_score(data[tv].labels, pred)
                per_seed_records.append({"method": name, "test_variant": tv, "seed": seed, "metrics": m,
                                          "n_params": None, "latency_ms": lat})
        print(f"[final_eval] {name}: {len(seeds_to_run)} seed(s) evaluated (has_randomness={has_rand})")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_DIR / "all_methods_per_seed_metrics.json", "w") as f:
        json.dump(per_seed_records, f, indent=2, default=str)
    print(f"[final_eval] wrote per-seed detail -> {RAW_DIR / 'all_methods_per_seed_metrics.json'} ({len(per_seed_records)} records)")

    # ---------------- Aggregate into the central CSV ----------------
    rows = []
    methods = sorted(set(r["method"] for r in per_seed_records))
    metric_keys = ["macro_f1", "f1", "precision", "recall", "accuracy", "false_alarm_rate",
                   "balanced_accuracy", "pr_auc", "roc_auc"]
    for method in methods:
        for tv in TEST_VARIANTS:
            recs = [r for r in per_seed_records if r["method"] == method and r["test_variant"] == tv]
            if not recs:
                continue
            metric_dicts = [r["metrics"] for r in recs]
            agg = aggregate_seeds(metric_dicts)
            for mk in metric_keys:
                if mk in agg:
                    rows.append({"method": method, "test_variant": tv, "metric": mk,
                                 "mean": agg[mk]["mean"], "std": agg[mk]["std"], "n_seeds": agg[mk]["n"]})
            # n_params / latency (single value per method, not per test_variant duplication needed,
            # but included per test_variant row set for convenience of downstream joins)
            n_params_vals = [r["n_params"] for r in recs if r["n_params"] is not None]
            if n_params_vals:
                rows.append({"method": method, "test_variant": tv, "metric": "n_params",
                             "mean": float(np.mean(n_params_vals)), "std": float(np.std(n_params_vals)), "n_seeds": len(n_params_vals)})
            lat_vals = [r["latency_ms"] for r in recs]
            rows.append({"method": method, "test_variant": tv, "metric": "latency_ms",
                         "mean": float(np.mean(lat_vals)), "std": float(np.std(lat_vals)), "n_seeds": len(lat_vals)})

    out_df = pd.DataFrame(rows)
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = AGG_DIR / "final_benchmark_comparison.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[final_eval] wrote {out_path} ({len(out_df)} rows, {len(methods)} methods)")
    print("[final_eval] done.")


if __name__ == "__main__":
    main()
