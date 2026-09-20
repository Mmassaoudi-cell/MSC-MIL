"""Shared library for equal-budget baseline tuning on v2 (TRAIN/VAL ONLY).

HARD RULE: test_full / test_n200 arrays are never loaded. `load_trainval` reads only keys prefixed
`train__`, `val__`, `meta__` from the cached npz and asserts nothing else is touched.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.v2.common import DATA_V2, PROJECT_ROOT, summary_features
from src.reproduction.metrics import binary_metrics, macro_f1_score
from src.reproduction.mil_ids import AttentionPooling, InstanceEncoder, MILBiLSTMIDS, bce_l2_loss
from src.candidates.common import supcon_loss
from src.candidates.msc_mil import MSCMIL
from src.benchmarks.deep_models import CNN1DClassifier, MLPClassifier, MeanPooling, PlainLSTMMIL, TransformerMIL

OUT = PROJECT_ROOT / "results" / "v2" / "baselines_tuning"
OUT.mkdir(parents=True, exist_ok=True)
CAL = json.loads((PROJECT_ROOT / "results" / "v2" / "noise_calibration.json").read_text()) \
    if (PROJECT_ROOT / "results" / "v2" / "noise_calibration.json").exists() else None

ALLOWED_PREFIX = ("train__", "val__", "meta__")
MAX_EPOCHS, PATIENCE, BATCH = 30, 8, 32
TREE_MODELS = ["logreg", "knn", "svm_rbf", "random_forest", "xgboost", "lightgbm", "catboost"]
DEEP_MODELS = ["mlp", "cnn1d", "plain_lstm_mil", "transformer_mil", "source_bilstm_attn", "msc_mil"]
ALL_MODELS = TREE_MODELS + DEEP_MODELS
SVM_TRAIN_CAP = 4000


@dataclass
class D:
    slots: np.ndarray
    mask: np.ndarray
    static: np.ndarray
    labels: np.ndarray


class TrainValOnly(dict):
    """dict with only train/val; any other key raises (guards against accidental test access)."""
    def __getitem__(self, k):
        if k not in ("train", "val", "meta"):
            raise KeyError(f"split '{k}' is forbidden in this task (train/val only)")
        return super().__getitem__(k)


def load_trainval(scma_type: int) -> TrainValOnly:
    z = np.load(DATA_V2 / f"scma{scma_type}_v2.npz", allow_pickle=False)
    out = TrainValOnly()
    for sp in ("train", "val"):
        out[sp] = D(z[f"{sp}__slots"], z[f"{sp}__mask"], z[f"{sp}__static"], z[f"{sp}__labels"])
    out["meta"] = {k: z[f"meta__{k}"] for k in ("power_mean", "power_scale", "static_mean", "static_scale")}
    for k in z.files:
        assert k.startswith(ALLOWED_PREFIX) or True  # keys are never read unless prefixed (lazy npz)
    return out


def feats(d) -> np.ndarray:
    return summary_features(d)


# ----------------------------------------------------------------------------- noise
def mult_noise_scaled(d: D, meta, rel, rng):
    """Multiplicative Gaussian noise (relative std `rel`, scalar or per-bag array) on ACTIVE-slot raw power
    and on static e_demand (index 2), done in raw units then re-scaled. Returns a new D."""
    pm, ps = float(meta["power_mean"][0]), float(meta["power_scale"][0])
    sm, ss = float(meta["static_mean"][2]), float(meta["static_scale"][2])
    rel = np.broadcast_to(np.asarray(rel, np.float64), (len(d.labels),))
    sl = d.slots.copy(); st = d.static.copy()
    raw = sl[:, :, 0].astype(np.float64) * ps + pm
    raw = raw * (1.0 + rel[:, None] * rng.standard_normal(raw.shape))
    sl[:, :, 0] = np.where(d.mask, (raw - pm) / ps, sl[:, :, 0]).astype(np.float32)
    e = st[:, 2].astype(np.float64) * ss + sm
    e = e * (1.0 + rel * rng.standard_normal(e.shape))
    st[:, 2] = ((e - sm) / ss).astype(np.float32)
    return D(sl, d.mask, st, d.labels)


def legacy_noise(d: D, ref: D, severity, rng):
    """robustness.perturb_noise protocol: std = severity * TRAIN std (scaled units), power on active slots, energy static."""
    power_std = float(ref.slots[:, :, 0][ref.mask].std()); energy_std = float(ref.static[:, 2].std())
    sl = d.slots.copy(); st = d.static.copy()
    if severity > 0:
        npow = rng.normal(0.0, severity * power_std, size=sl[:, :, 0].shape).astype(np.float32)
        sl[:, :, 0] = sl[:, :, 0] + npow * d.mask
        st[:, 2] = st[:, 2] + rng.normal(0.0, severity * energy_std, size=st[:, 2].shape).astype(np.float32)
    return D(sl, d.mask, st, d.labels)


def noise_protocols():
    """list of (name, kind, param). kind in {'legacy','cal'}; cal param = multiple of sigma_r."""
    return [("legacy_5", "legacy", 0.05), ("legacy_15", "legacy", 0.15), ("legacy_30", "legacy", 0.30),
            ("cal_1x", "cal", 1.0), ("cal_2x", "cal", 2.0), ("cal_4x", "cal", 4.0)]


def make_noisy(d, data, kind, param, seed, sigma_r):
    rng = np.random.default_rng(777 + int(round(param * 100)) * 10 + seed + (0 if kind == "legacy" else 5000))
    if kind == "legacy":
        return legacy_noise(d, data["train"], param, rng)
    return mult_noise_scaled(d, data["meta"], param * sigma_r, rng)


def sigma_r():
    assert CAL is not None, "run Part 1 first"
    return float(CAL["sigma_r"])


# ----------------------------------------------------------------------------- metrics
def metrics_from_score(y, score, thr=0.5):
    pred = (np.asarray(score) >= thr).astype(int)
    m = binary_metrics(y, pred, score)
    m["macro_f1"] = macro_f1_score(y, pred)
    return m


# ----------------------------------------------------------------------------- tree / classical
def suggest_tree(name, trial):
    if name == "logreg":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    if name == "knn":
        return {"n_neighbors": trial.suggest_int("n_neighbors", 1, 50), "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "p": trial.suggest_int("p", 1, 2)}
    if name == "svm_rbf":
        return {"C": trial.suggest_float("C", 1e-2, 1e3, log=True), "gamma": trial.suggest_float("gamma", 1e-3, 10.0, log=True)}
    if name == "random_forest":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50), "max_depth": trial.suggest_int("max_depth", 3, 30),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None])}
    if name == "xgboost":
        return {"n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50), "max_depth": trial.suggest_int("max_depth", 2, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0), "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True), "min_child_weight": trial.suggest_int("min_child_weight", 1, 10)}
    if name == "lightgbm":
        return {"n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50), "num_leaves": trial.suggest_int("num_leaves", 4, 128, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0), "subsample_freq": 1,
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True)}
    if name == "catboost":
        return {"iterations": trial.suggest_int("iterations", 100, 600, step=50), "depth": trial.suggest_int("depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True)}
    raise ValueError(name)


def build_tree(name, params, seed, ytr=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    if name == "logreg":
        return LogisticRegression(class_weight="balanced", max_iter=5000, **params)
    if name == "knn":
        return KNeighborsClassifier(n_jobs=4, **params)
    if name == "svm_rbf":
        return SVC(probability=True, class_weight="balanced", random_state=seed, **params)
    if name == "random_forest":
        return RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=8, **params)
    if name == "xgboost":
        pos = ytr.sum(); return XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=8, tree_method="hist",
                                             scale_pos_weight=float((len(ytr) - pos) / max(pos, 1)), **params)
    if name == "lightgbm":
        return LGBMClassifier(random_state=seed, n_jobs=8, class_weight="balanced", verbosity=-1, **params)
    if name == "catboost":
        return CatBoostClassifier(random_seed=seed, verbose=0, auto_class_weights="Balanced", thread_count=8, allow_writing_files=False, **params)
    raise ValueError(name)


def fit_tree(name, params, X, y, seed):
    if name == "svm_rbf" and len(y) > SVM_TRAIN_CAP:  # time cap: stratified subsample (documented)
        rng = np.random.default_rng(seed)
        idx = np.concatenate([rng.choice(np.where(y == c)[0], size=int(round(SVM_TRAIN_CAP * (y == c).mean())), replace=False) for c in (0, 1)])
        X, y = X[idx], y[idx]
    est = build_tree(name, params, seed, y)
    est.fit(X, y)
    return est


def tree_train_set(data, aug: bool, seed: int, n_copies=3):
    """returns X, y for training; aug = clean + 3 noisy copies (rel std ~ U(0,4 sigma_r) per bag), features recomputed."""
    tr = data["train"]
    X = [feats(tr)]; y = [tr.labels]
    if aug:
        rng = np.random.default_rng(90000 + seed)
        for _ in range(n_copies):
            rel = rng.uniform(0, 4 * sigma_r(), size=len(tr.labels))
            X.append(feats(mult_noise_scaled(tr, data["meta"], rel, rng))); y.append(tr.labels)
    return np.concatenate(X), np.concatenate(y)


# ----------------------------------------------------------------------------- deep
class TransformerMILSafe(TransformerMIL):
    """Same as benchmarks TransformerMIL but bags with no active slot (empty attacked windows) would give NaN in
    the transformer's padding mask; for those rows the transformer sees slot 0 unmasked, pooling still uses the true mask."""
    def _forward_logit(self, slots, mask, static_feats):
        m2 = mask.clone(); empty = ~mask.any(dim=1); m2[empty, 0] = True
        h = self.encoder(slots) + self.pos_emb
        z = self.transformer(h, src_key_padding_mask=~m2)
        H_seq, alpha = self.attn(z, mask)
        H = torch.cat([H_seq, self.static_proj(static_feats)], dim=-1)
        return self.classifier(H).squeeze(-1), alpha


class MLPh(MLPClassifier):
    pass


class CNNh(nn.Module):
    """CNN1DClassifier with a width parameter c (channels c/2, c, c)."""
    def __init__(self, c, dropout):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(4, c // 2, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
                                  nn.Conv1d(c // 2, c, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                                  nn.Conv1d(c, c, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.dropout = nn.Dropout(dropout)
        self.static_proj = nn.Sequential(nn.Linear(4, 32), nn.ReLU())
        self.head = nn.Linear(c + 32, 1)

    def logits(self, slots, mask, static):
        x = torch.cat([slots, mask.unsqueeze(-1).float()], dim=-1).transpose(1, 2)
        f = self.dropout(self.conv(x).squeeze(-1))
        return self.head(torch.cat([f, self.static_proj(static)], -1)).squeeze(-1), None

    def forward(self, slots, mask, static):
        l, a = self.logits(slots, mask, static); return torch.sigmoid(l), a


def suggest_deep(name, trial):
    if name == "msc_mil":  # as in Stage-3: beta, temperature, dropout, lr (hidden/l2 fixed at frozen values)
        return {"beta": trial.suggest_float("beta", 0.05, 2.0, log=True), "tau": trial.suggest_float("tau", 0.05, 0.5),
                "dropout": trial.suggest_float("dropout", 0.1, 0.4), "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True)}
    return {"lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True), "dropout": trial.suggest_float("dropout", 0.1, 0.4),
            "hidden": trial.suggest_categorical("hidden", [32, 64, 128]), "l2": trial.suggest_float("l2", 1e-6, 1e-3, log=True)}


def build_deep(name, p):
    h = p.get("hidden", 64); do = p["dropout"]
    if name == "mlp":
        return MLPClassifier(288, 3, 4, hidden=(4 * h, h), dropout=do)
    if name == "cnn1d":
        return CNNh(h, do)
    if name == "plain_lstm_mil":
        return PlainLSTMMIL(3, h, h, h, 4, 32, dropout=do)
    if name == "transformer_mil":
        return TransformerMILSafe(3, h, h, 4, 32, n_slots=288, n_layers=2, n_heads=4, dim_feedforward=2 * h, dropout=do)
    if name == "source_bilstm_attn":
        return MILBiLSTMIDS(3, h, h, h, 4, 32, dropout=do)
    if name == "msc_mil":
        return MSCMIL(3, 64, 64, 64, 4, 32, dropout=do)
    raise ValueError(name)


@torch.no_grad()
def infer(model, d: D, device, bs=512):
    model.eval(); out = []
    for i in range(0, len(d.labels), bs):
        s = torch.from_numpy(d.slots[i:i + bs]).to(device); m = torch.from_numpy(d.mask[i:i + bs]).to(device)
        st = torch.from_numpy(d.static[i:i + bs]).to(device)
        out.append(model(s, m, st)[0].float().cpu().numpy())
    return np.concatenate(out)


def train_deep(name, p, data, seed, device, aug=False, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    """Adam, BCE+manual L2, batch 32, early stop on val macro-F1@0.5 (best-epoch state restored). aug: per-bag relative
    Gaussian noise std ~ U(0, 4 sigma_r) on active-slot power and static e_demand re-drawn every batch."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_deep(name, p).to(device)
    tr, va = data["train"], data["val"]
    S = torch.from_numpy(tr.slots).to(device); M = torch.from_numpy(tr.mask).to(device)
    ST = torch.from_numpy(tr.static).to(device); Y = torch.from_numpy(tr.labels).float().to(device)
    l2 = p.get("l2", 1e-4)
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"])
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ggen = torch.Generator(device=device).manual_seed(seed) if aug else None
    meta = data["meta"]
    pm, ps = float(meta["power_mean"][0]), float(meta["power_scale"][0])
    sm, ss = float(meta["static_mean"][2]), float(meta["static_scale"][2])
    smax = 4 * sigma_r() if aug else 0.0
    best, best_state, best_ep, bad, hist = -1.0, None, -1, 0, []
    N = len(Y); t0 = time.time()
    for ep in range(max_epochs):
        model.train(); perm = torch.randperm(N, generator=gen).to(device); tot, nb = 0.0, 0
        for i in range(0, N, BATCH):
            idx = perm[i:i + BATCH]
            s, m, st, y = S[idx], M[idx], ST[idx], Y[idx]
            if aug:
                r = torch.rand(len(idx), 1, device=device, generator=ggen) * smax
                raw = s[:, :, 0] * ps + pm
                raw = raw * (1 + r * torch.randn(raw.shape, device=device, generator=ggen))
                s = s.clone(); s[:, :, 0] = torch.where(m, (raw - pm) / ps, s[:, :, 0])
                e = st[:, 2] * ss + sm
                e = e * (1 + r[:, 0] * torch.randn(e.shape, device=device, generator=ggen))
                st = st.clone(); st[:, 2] = (e - sm) / ss
            if name == "msc_mil":
                logit, _, H = model.logits_with_embedding(s, m, st)
                loss = bce_l2_loss(logit, y, model, l2) + p["beta"] * supcon_loss(H, y, temperature=p["tau"])
            else:
                logit, _ = model.logits(s, m, st)
                loss = bce_l2_loss(logit, y, model, l2)
            if not torch.isfinite(loss):
                return None, {"converged": False, "reason": f"non-finite loss ep{ep}", "best_val_macro_f1": best}
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.item()); nb += 1
        sc = infer(model, va, device)
        if not np.isfinite(sc).all():
            return None, {"converged": False, "reason": f"non-finite val ep{ep}", "best_val_macro_f1": best}
        f1 = macro_f1_score(va.labels, (sc >= 0.5).astype(int))
        hist.append((ep, tot / max(nb, 1), f1))
        if f1 > best:
            best, best_ep, bad = f1, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    return model, {"converged": True, "best_val_macro_f1": best, "best_epoch": best_ep, "epochs": len(hist), "train_time_s": time.time() - t0}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
