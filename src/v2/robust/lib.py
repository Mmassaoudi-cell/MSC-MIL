"""Post-freeze analysis library (robustness / ablation / low-data / multiclass). Nothing here changes a frozen model.
Test arrays (test_full) are read here for the post-freeze analyses requested by the peer review."""
from __future__ import annotations

import copy
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from src.benchmarks.robustness import perturb_noise, perturb_missing, compute_fill_value
from src.v2 import baselines_lib as L
from src.v2 import enhanced_model as em
from src.v2.common import DATA_V2, PROJECT_ROOT

FIN = PROJECT_ROOT / "results" / "v2" / "final"
FCK = FIN / "ckpt"
ROB = PROJECT_ROOT / "results" / "v2" / "robust"
RCK = ROB / "ckpt"
RAW = ROB / "raw"
PROBS = ROB / "probs"
for _p in (ROB, RCK, RAW, PROBS, ROB / "logs"):
    _p.mkdir(parents=True, exist_ok=True)
FROZEN = json.loads((PROJECT_ROOT / "results/v2/frozen_baseline_configs.json").read_text())["models"]
SIGMA_R = float(json.loads((PROJECT_ROOT / "results/v2/noise_calibration.json").read_text())["sigma_r"])
MISSING_REAL = 0.0015
SEEDS5 = (0, 1, 2, 3, 4)
TYPES = (1, 4)

# condition name -> (kind, param)
CONDS = {
    "clean": ("clean", 0.0),
    "cal_1x": ("cal", 1.0), "cal_2x": ("cal", 2.0), "cal_4x": ("cal", 4.0),
    "legacy_5": ("legacy", 0.05), "legacy_15": ("legacy", 0.15), "legacy_30": ("legacy", 0.30),
    "miss_real": ("miss_real", MISSING_REAL), "miss_10": ("miss", 0.10), "miss_25": ("miss", 0.25), "miss_50": ("miss", 0.50),
}
COND_IDX = {c: i for i, c in enumerate(CONDS)}


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_obj(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


def load_all(t):
    """train/val/meta (frozen loader) + test_full."""
    data = L.load_trainval(t)
    z = np.load(DATA_V2 / f"scma{t}_v2.npz", allow_pickle=False)
    test = L.D(z["test_full__slots"], z["test_full__mask"], z["test_full__static"], z["test_full__labels"])
    return data, test


def make_condition(cond, test, train, meta, t, s):
    """Deterministic perturbed copy of `test` (same draw for every model of the same (cond,type,seed))."""
    kind, par = CONDS[cond]
    if kind == "clean":
        return test
    rng = np.random.default_rng([20260920, COND_IDX[cond], t, s])
    if kind == "cal":
        return L.mult_noise_scaled(test, meta, par * SIGMA_R, rng)
    if kind == "legacy":
        pstd = float(train.slots[:, :, 0][train.mask].std()); estd = float(train.static[:, 2].std())
        return perturb_noise(test, pstd, estd, par, rng)   # the legacy function itself
    fill = compute_fill_value(train)
    if kind == "miss":
        return perturb_missing(test, par, rng, fill)       # the legacy function itself (fixed fraction of active slots, rounded)
    if kind == "miss_real":
        # Bernoulli per active slot: with round(0.0015*n_active) the legacy fixed-fraction rule masks nothing for n_active<334.
        d2 = copy.deepcopy(test)
        drop = (rng.random(d2.mask.shape) < par) & d2.mask
        d2.mask = d2.mask & ~drop
        d2.slots[:, :, 0] = np.where(drop, np.float32(fill), d2.slots[:, :, 0])
        return d2
    raise ValueError(cond)


def metrics(y, p, thr=0.5):
    pred = (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    f1p = 2 * tp / max(2 * tp + fp + fn, 1); f1n = 2 * tn / max(2 * tn + fn + fp, 1)
    return dict(macro_f1=(f1p + f1n) / 2, recall=tp / max(tp + fn, 1), far=fp / max(fp + tn, 1), precision=tp / max(tp + fp, 1),
                tp=tp, fp=fp, tn=tn, fn=fn)


# ------------------------------------------------------------------ model loading
def dev():
    return L.get_device()


def load_pi(t, s, meta, device):
    net = em.EnhancedMSCMIL(meta, physics=False, dropout=em.FROZEN_NET_CFG["dropout"]).to(device)
    net.load_state_dict(torch.load(FCK / f"pi_fusion_scma{t}_seed{s}_net.pt", map_location=device))
    net.eval()
    with open(FCK / f"pi_fusion_scma{t}_seed{s}_xgb.pkl", "rb") as f:
        tree = pickle.load(f)
    return net, tree


def deep_path(tag, t, s):
    p = RCK / f"{tag}_scma{t}_seed{s}.pt"
    return p if tag.endswith("_aug") else FCK / f"{tag}_scma{t}_seed{s}.pt"


def tree_path(tag, t, s):
    return RCK / f"{tag}_scma{t}_seed{s}.pkl" if tag.endswith("_aug") else FCK / f"{tag}_scma{t}_seed{s}.pkl"


def load_deep(tag, t, s, device):
    base = FROZEN[tag]["base_model"]
    m = L.build_deep(base, FROZEN[tag]["best_params"]).to(device)
    m.load_state_dict(torch.load(deep_path(tag, t, s), map_location=device))
    m.eval()
    return m


def load_tree(tag, t, s):
    with open(tree_path(tag, t, s), "rb") as f:
        return pickle.load(f)


DEEP_TAGS = ["msc_mil", "source_bilstm_attn", "msc_mil_aug", "source_bilstm_attn_aug"]
PF_TREES = ["xgboost_pf", "lightgbm_pf", "catboost_pf"]
F7_TREES = ["catboost", "random_forest", "xgboost", "xgboost_aug", "catboost_aug", "random_forest_aug"]
DISPLAY = {"pi_fusion": "PI-Fusion (E5)", "e4_net": "E4 network alone", "msc_mil": "MSC-MIL (orig.)", "source_bilstm_attn": "Source BiLSTM+attn",
           "xgboost_pf": "XGBoost-PF", "lightgbm_pf": "LightGBM-PF", "catboost_pf": "CatBoost-PF", "catboost": "CatBoost (7-feat)",
           "random_forest": "RandomForest", "xgboost": "XGBoost", "msc_mil_aug": "MSC-MIL +noise-aug", "source_bilstm_attn_aug": "Source BiLSTM+attn +noise-aug",
           "xgboost_aug": "XGBoost +noise-aug", "catboost_aug": "CatBoost +noise-aug", "random_forest_aug": "RandomForest +noise-aug"}
MODELS = list(DISPLAY)


def bundle_available(t, s):
    need = [FCK / f"pi_fusion_scma{t}_seed{s}_net.pt", FCK / f"pi_fusion_scma{t}_seed{s}_xgb.pkl"]
    need += [deep_path(k, t, s) for k in DEEP_TAGS] + [tree_path(k, t, s) for k in PF_TREES + F7_TREES]
    return [str(p) for p in need if not p.exists()]


def load_bundle(t, s, meta, device):
    b = dict(pi=load_pi(t, s, meta, device))
    b["deep"] = {k: load_deep(k, t, s, device) for k in DEEP_TAGS}
    b["tree"] = {k: load_tree(k, t, s) for k in PF_TREES + F7_TREES}
    return b


def predict_all(b, d, meta, device):
    """-> dict model -> per-session attack probability on (possibly perturbed) d; features recomputed from d."""
    net, xgb = b["pi"]
    F15 = em.tree_features(d.slots, d.mask, d.static, meta)
    F7 = L.feats(d)
    out = {}
    pn = em._net_proba(net, d, device)
    pt = xgb.predict_proba(F15)[:, 1]
    out["e4_net"] = pn
    out["pi_fusion"] = em.FROZEN_W_TREE * pt + (1 - em.FROZEN_W_TREE) * pn
    out["_pi_tree"] = pt
    for k, m in b["deep"].items():
        out[k] = L.infer(m, d, device)
    for k in PF_TREES:
        out[k] = b["tree"][k].predict_proba(F15)[:, 1]
    for k in F7_TREES:
        out[k] = b["tree"][k].predict_proba(F7)[:, 1]
    return out


def agg(rows, keys=("macro_f1", "recall", "far")):
    """list of dicts -> {key: (mean, std)}"""
    o = {}
    for k in keys:
        v = np.array([r[k] for r in rows], float)
        o[k] = (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan"))
    return o
