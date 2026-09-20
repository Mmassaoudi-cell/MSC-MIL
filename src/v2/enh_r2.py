"""Round 2 (validation only): PF-tree baselines + E5 PF-Fusion. See ENHANCEMENT_PROTOCOL.md 'ROUND 2'.
python -m src.v2.enh_r2 pf_trees | e4_s2 | e5
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch

from src.v2 import baselines_lib as L
from src.v2 import enhanced_model as em
from src.v2 import enh_run as R
from src.v2.common import RESULTS_V2

PF_OUT = RESULTS_V2 / "baselines_tuning_pf"
PF_OUT.mkdir(exist_ok=True)
MODELS = ["catboost", "xgboost", "lightgbm", "random_forest"]
DISPLAY = {"catboost": "CatBoost-PF", "xgboost": "XGBoost-PF", "lightgbm": "LightGBM-PF", "random_forest": "RandomForest-PF"}
SEEDS = (0, 1, 2)
TYPES = (1, 2, 4)


def pf_feats(d, meta):
    X = L.feats(d)
    c = em._consts(meta, "cpu")
    with torch.no_grad():
        ph = em.physics_features(torch.from_numpy(d.slots[..., 0]), torch.from_numpy(d.mask), torch.from_numpy(d.static), c).numpy()
    return np.hstack([X, ph])


def tune(name):
    fp = PF_OUT / f"best_{name}.json"
    if fp.exists():
        return json.loads(fp.read_text())
    data = L.load_trainval(1)
    Xtr = pf_feats(data["train"], data["meta"]); ytr = data["train"].labels
    Xva = pf_feats(data["val"], data["meta"]); yva = data["val"].labels

    def obj(trial):
        p = L.suggest_tree(name, trial)
        est = L.fit_tree(name, p, Xtr, ytr, 0)
        return L.metrics_from_score(yva, est.predict_proba(Xva)[:, 1])["macro_f1"]

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0),
                             storage=f"sqlite:///{(PF_OUT / 'optuna.db').as_posix()}", study_name=name, load_if_exists=True)
    todo = 15 - len([t for t in st.trials if t.state.name == "COMPLETE"])
    if todo > 0:
        st.optimize(obj, n_trials=todo)
    best = dict(best_params=st.best_params, best_value=st.best_value, n_trials=len(st.trials))
    # int params that suggest_tree returns unchanged; add fixed keys
    if name == "lightgbm":
        best["best_params"]["subsample_freq"] = 1
    fp.write_text(json.dumps(best, indent=1))
    print(f"[PF tune] {name}: {best}", flush=True)
    return best


def final_runs():
    csv = PF_OUT / "final_runs.csv"
    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    done = {(r["model"], r["type"], r["seed"]) for r in rows}
    sr = L.sigma_r()
    for t in TYPES:
        data = L.load_trainval(t)
        Xtr = pf_feats(data["train"], data["meta"]); ytr = data["train"].labels
        for name in MODELS:
            p = json.loads((PF_OUT / f"best_{name}.json").read_text())["best_params"]
            for s in SEEDS:
                if (name, t, s) in done:
                    continue
                est = L.fit_tree(name, p, Xtr, ytr, s)
                y = data["val"].labels
                m = L.metrics_from_score(y, est.predict_proba(pf_feats(data["val"], data["meta"]))[:, 1])
                row = dict(model=name, type=t, seed=s, clean=m["macro_f1"], recall_clean=m["recall"], far_clean=m["false_alarm_rate"])
                for pname, kind, prm in L.noise_protocols():
                    nd = L.make_noisy(data["val"], data, kind, prm, s, sr)
                    row[pname] = L.metrics_from_score(y, est.predict_proba(pf_feats(nd, data["meta"]))[:, 1])["macro_f1"]
                rows.append(row)
                pd.DataFrame(rows).to_csv(csv, index=False)
                print(f"[PF final] {name} SCMA-{t} seed{s} clean={row['clean']:.4f} cal4x={row['cal_4x']:.4f} leg15={row['legacy_15']:.4f}", flush=True)


def pf_trees():
    for n in MODELS:
        tune(n)
    final_runs()
    # append NEW entries to frozen_baseline_configs.json (existing entries untouched)
    fp = RESULTS_V2 / "frozen_baseline_configs.json"
    cfg = json.loads(fp.read_text())
    for n in MODELS:
        k = f"{n}_pf"
        if k in cfg["models"]:
            continue
        b = json.loads((PF_OUT / f"best_{n}.json").read_text())
        cfg["models"][k] = dict(display=DISPLAY[n], base_model=n, noise_augmented=False, physics_features=True,
                                features="7-dim summary_features_v2 + 8 physics features (src/v2/enhanced_model.py::physics_features), 15 dims",
                                search_space="same as the non-PF model (baselines_lib.suggest_tree)", best_params=b["best_params"],
                                best_tuning_val_macro_f1_seed0=b["best_value"], n_trials=b["n_trials"],
                                code="src/v2/enh_r2.py (pf_feats / tune / final_runs); ENHANCEMENT_PROTOCOL.md Round 2")
    fp.write_text(json.dumps(cfg, indent=1))
    print("[PF] frozen_baseline_configs.json updated (new *_pf entries)", flush=True)


def e4_s2():
    b = R.optuna_nn("E4")
    cfg = R.apply_params(R.base_cfg("E4"), b["best_params"])
    for s in SEEDS:
        R.run_nn("E4", cfg, 2, s, "E4")


# ------------------------------------------------------------------ E5
def tree_size(est, name):
    if name == "catboost":
        return int(est.tree_count_ * 2 ** est.get_param("depth")) if est.get_param("depth") else int(est.tree_count_ * 64)
    if name == "random_forest":
        return int(sum(e.tree_.node_count for e in est.estimators_))
    if name == "xgboost":
        return int(est.get_params()["n_estimators"] * 2 ** est.get_params()["max_depth"])
    return int(est.get_params()["n_estimators"] * est.get_params()["num_leaves"])


def fit_predict_tree(name, params, t, s):
    d, meta = R.data(t)
    X, y = R.cb_train_set(d["train"], meta, True, s, k=4)
    est = L.fit_tree(name, params, X, y, s)
    refs = em.noise_refs(d["train"])
    noisy = em.noisy_copies(d["val"], refs, seed_offset=s)
    out = {"clean": est.predict_proba(R.cb_features(d["val"].slots, d["val"].mask, d["val"].static, meta, True))[:, 1]}
    for lv, ns in zip(em.NOISE_LEVELS, noisy):
        out[f"n{lv}"] = est.predict_proba(R.cb_features(ns.slots, ns.mask, ns.static, meta, True))[:, 1]
    return est, out


def e4_probs(t, s):
    z = np.load(R.OUT / "probs" / f"E4_scma{t}_seed{s}.npz")
    return {k: z[k] for k in z.files if k != "y"}


def e5():
    df = pd.read_csv(PF_OUT / "final_runs.csv")
    d1 = df[df.type == 1].groupby("model").clean.mean()
    name = d1.idxmax()
    params = json.loads((PF_OUT / f"best_{name}.json").read_text())["best_params"]
    print(f"[E5] best PF tree by SCMA-1 clean val F1: {name} ({d1.to_dict()})", flush=True)
    fpw = R.OUT / "E5_weight.json"
    y1 = R.data(1)[0]["val"].labels
    cache = {}

    def tp(t, s):
        if (t, s) not in cache:
            cache[(t, s)] = fit_predict_tree(name, params, t, s)
        return cache[(t, s)]

    if fpw.exists():
        wj = json.loads(fpw.read_text())
    else:
        _, pc = tp(1, 0)
        grid = []
        for w in np.round(np.arange(0, 1.0001, 0.05), 2):
            r = R.fuse_eval(pc, e4_probs(1, 0), float(w), y1)
            grid.append(dict(w=float(w), S_T=r["S_T"], clean=r["clean_f1"], noisy=r["noisy_mean"]))
        bw = max(grid, key=lambda g: g["S_T"])
        wj = dict(tree=name, params=params, w=bw["w"], grid=grid, note="w = weight on the tree; chosen on SCMA-1 validation, seed 0 (model-selection use of val)")
        fpw.write_text(json.dumps(wj, indent=1))
    w = wj["w"]
    print(f"[E5] tuned w={w}", flush=True)
    out_rows = []
    for t in (1, 4, 2):
        y = R.data(t)[0]["val"].labels
        for s in SEEDS:
            fp = R.OUT / f"E5_scma{t}_seed{s}.json"
            if fp.exists():
                out_rows.append(json.loads(fp.read_text())); continue
            est, pc = tp(t, s)
            r = R.fuse_eval(pc, e4_probs(t, s), w, y)
            e4 = json.loads((R.OUT / f"E4_scma{t}_seed{s}.json").read_text())
            o = dict(cand="E5", scma=t, seed=s, w=w, tree=name, tree_params=params, n_params=e4["n_params"] + tree_size(est, name),
                     tree_size=tree_size(est, name), converged=True, **r)
            if t == 1 and s == 0:
                d, meta = R.data(1); v = d["val"]
                for _ in range(10): est.predict_proba(R.cb_features(v.slots[:1], v.mask[:1], v.static[:1], meta, True))
                ts = []
                for _ in range(200):
                    t1 = time.perf_counter(); est.predict_proba(R.cb_features(v.slots[:1], v.mask[:1], v.static[:1], meta, True)); ts.append(time.perf_counter() - t1)
                lt = float(np.median(ts) * 1e3)
                e41 = json.loads((R.OUT / "E4_scma1_seed0.json").read_text())
                o.update(latency_ms_tree_part=lt, latency_ms_cuda=e41["latency_ms_cuda"] + lt, latency_ms_cpu=e41["latency_ms_cpu"] + lt)
            fp.write_text(json.dumps(o, indent=1, default=float))
            out_rows.append(o)
            print(f"[E5] SCMA-{t} seed{s}: clean={r['clean_f1']:.4f} noisy={r['noisy_f1']} S_T={r['S_T']:.4f} rec={r['recall_clean']:.3f}", flush=True)


if __name__ == "__main__":
    for a in sys.argv[1:]:
        {"pf_trees": pf_trees, "e4_s2": e4_s2, "e5": e5}[a]()
