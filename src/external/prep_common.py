"""Shared windowed-MIL construction for the external (real-attack) validation.

bag = W consecutive flows/packets of ONE source segment in capture order (non-overlapping);
bag label = 1 iff ANY instance is an attack. Chronological 70/15/15 split of the bag sequence
WITHIN each source segment, then union. Exact duplicate bags removed across splits.
Transform: signed log1p on every feature, train-median imputation, StandardScaler fit on train
instances only, clip to +-10. Constant (train) columns dropped.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed_external"
RES = ROOT / "results" / "external"
SPLITS = ("train", "val", "test")


def window_segment(X, y, W):
    """X (n,d) raw float32 in capture order; y (n,) {0,1}. -> bags (nb,W,d), labels (nb,)."""
    nb = len(X) // W
    Xb = X[: nb * W].reshape(nb, W, X.shape[1])
    yb = y[: nb * W].reshape(nb, W).max(axis=1)
    return Xb, yb.astype(np.int8)


def chrono_split_idx(nb, fr=(0.70, 0.15, 0.15)):
    a = int(round(nb * fr[0])); b = int(round(nb * (fr[0] + fr[1])))
    return {"train": np.arange(0, a), "val": np.arange(a, b), "test": np.arange(b, nb)}


def bag_hash(bag):
    return hashlib.md5(np.nan_to_num(bag, nan=-1e30).tobytes()).digest()


def build_splits(segments, W):
    """segments: list of (name, X_raw, y_inst). Returns dict split -> dict(X,y,src) raw, plus stats."""
    out = {s: {"X": [], "y": [], "src": []} for s in SPLITS}
    stats = {}
    for si, (name, X, y) in enumerate(segments):
        Xb, yb = window_segment(X, y, W)
        idx = chrono_split_idx(len(Xb))
        stats[name] = {"n_rows": int(len(X)), "n_bags": int(len(Xb)), "row_prev": float(y.mean()),
                       **{f"{s}_bags": int(len(idx[s])) for s in SPLITS},
                       **{f"{s}_pos": int(yb[idx[s]].sum()) for s in SPLITS}}
        for s in SPLITS:
            out[s]["X"].append(Xb[idx[s]]); out[s]["y"].append(yb[idx[s]])
            out[s]["src"].append(np.full(len(idx[s]), si, dtype=np.int16))
    for s in SPLITS:
        for k in ("X", "y", "src"):
            out[s][k] = np.concatenate(out[s][k])
    return out, stats


def dedup_across(out):
    """Drop exact-duplicate bags: repeats within train (keep first); val/test bags equal to any
    earlier-split bag (train, or val for test)."""
    seen = set(); removed = {}
    for s in SPLITS:
        hs = [bag_hash(b) for b in out[s]["X"]]
        if s == "train":
            first = set(); keep = np.ones(len(hs), bool)
            for i, h in enumerate(hs):
                if h in first:
                    keep[i] = False
                first.add(h)
        else:
            keep = np.array([h not in seen for h in hs])
        removed[s] = int((~keep).sum())
        for k in out[s]:
            out[s][k] = out[s][k][keep]
        seen.update(hs)
    return removed


def cap_train(out, cap, seed=0):
    """Seeded class-aware subsample of TRAIN to <= cap bags: keep all minority-class bags, subsample majority."""
    y = out["train"]["y"]
    if len(y) <= cap:
        return {"applied": False}
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    minor, major = (pos, neg) if len(pos) < len(neg) else (neg, pos)
    n_major = max(cap - len(minor), 0)
    keep = np.sort(np.concatenate([minor, rng.choice(major, size=min(n_major, len(major)), replace=False)]))
    for k in out["train"]:
        out["train"][k] = out["train"][k][keep]
    return {"applied": True, "cap": cap, "kept": int(len(keep))}


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def fit_transform(out, feat_names, clip=10.0):
    """Fit imputer+scaler on train instances only; transform all splits. Returns arrays & meta."""
    d = out["train"]["X"].shape[2]
    T = {s: slog(out[s]["X"]).astype(np.float32) for s in SPLITS}  # inf must already be NaN
    flat = T["train"].reshape(-1, d)
    med = np.nan_to_num(np.nanmedian(flat, axis=0), nan=0.0)
    nan_frac = float(np.isnan(flat).mean())
    for s in SPLITS:
        a = T[s]; m = np.isnan(a)
        if m.any():
            a[m] = np.broadcast_to(med, a.shape)[m]
    flat = T["train"].reshape(-1, d)
    mu = flat.mean(0); sd = flat.std(0)
    keep = sd > 1e-6
    for s in SPLITS:
        T[s] = np.clip((T[s][:, :, keep] - mu[keep]) / sd[keep], -clip, clip).astype(np.float32)
    names = [n for n, k in zip(feat_names, keep) if k]
    dropped_const = [n for n, k in zip(feat_names, keep) if not k]
    flat = T["train"].reshape(-1, T["train"].shape[2])
    samp = flat[np.random.default_rng(0).choice(len(flat), size=min(len(flat), 200000), replace=False)]
    cont = np.array([len(np.unique(samp[:, j])) > 2 for j in range(samp.shape[1])])
    meta = {"feature_names": names, "dropped_constant": dropped_const, "cont_mask": cont.tolist(),
            "train_nan_frac_before_impute": nan_frac, "clip": clip}
    return T, meta


def save(ds, T, out, meta):
    d = PROC / ds; d.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        np.save(d / f"X_{s}.npy", T[s]); np.save(d / f"y_{s}.npy", out[s]["y"]); np.save(d / f"src_{s}.npy", out[s]["src"])
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
