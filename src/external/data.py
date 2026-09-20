"""Loading cached windowed arrays, window-summary features, and the evaluation-time perturbations."""
from __future__ import annotations
import json
import numpy as np
from src.external.prep_common import PROC, SPLITS

DATASETS = {"cic2017": "cic2017_W64", "edge_iiot": "edge_iiot_W64"}


def load(ds, splits=("train", "val")):
    """Only the requested splits are read from disk. TEST is read only when explicitly asked for."""
    d = PROC / DATASETS.get(ds, ds)
    meta = json.loads((d / "meta.json").read_text())
    out = {"meta": meta}
    for s in splits:
        out[s] = {"X": np.load(d / f"X_{s}.npy"), "y": np.load(d / f"y_{s}.npy").astype(np.int64),
                  "src": np.load(d / f"src_{s}.npy")}
    return out


def summary_features(X, mask=None, chunk=4096):
    """(n,W,d) [+ (n,W) bool mask of retained instances] -> (n,4d): per-feature mean,std,min,max over
    the retained instances of the window. Bags with no retained instance -> zeros."""
    n, W, d = X.shape
    res = np.zeros((n, 4 * d), np.float32)
    for a in range(0, n, chunk):
        x = X[a:a + chunk]
        if mask is None:
            res[a:a + chunk] = np.concatenate([x.mean(1), x.std(1), x.min(1), x.max(1)], 1)
        else:
            m = mask[a:a + chunk][:, :, None]
            cnt = m.sum(1).clip(min=1)
            mean = (x * m).sum(1) / cnt
            var = (((x - mean[:, None]) ** 2) * m).sum(1) / cnt
            mn = np.where(m, x, np.inf).min(1); mx = np.where(m, x, -np.inf).max(1)
            empty = ~mask[a:a + chunk].any(1)
            mn[empty] = 0; mx[empty] = 0
            res[a:a + chunk] = np.concatenate([mean, np.sqrt(var), mn, mx], 1)
    return res


# ---------------- perturbation protocol (evaluation time only) ----------------
def train_feature_std(Xtr):
    """Per-feature std over all train instances (in the scaled space) -- noise scale reference."""
    return Xtr.reshape(-1, Xtr.shape[2]).std(0).astype(np.float32)


def perturb_noise(X, severity, feat_std, cont_mask, rng):
    """Add N(0, (severity*train_std)^2) to every continuous feature of every instance. Binary/flag
    features (cont_mask False) are left untouched."""
    sd = (severity * feat_std * np.asarray(cont_mask, np.float32))[None, None, :]
    return X + rng.standard_normal(X.shape).astype(np.float32) * sd


def perturb_missing(X, frac, rng, fill=0.0):
    """Randomly remove round(frac*W) instances from every bag. Returns (X', mask): removed instances are set
    to `fill` (=0 = train mean in scaled space) and mask=False (deep models mask them out of pooling /
    attention; trees recompute summaries over the retained instances only)."""
    n, W, d = X.shape
    k = int(round(frac * W))
    mask = np.ones((n, W), bool)
    if k > 0:
        r = rng.random((n, W))
        drop = np.argsort(r, axis=1)[:, :k]
        np.put_along_axis(mask, drop, False, axis=1)
    X2 = np.where(mask[:, :, None], X, np.float32(fill)).astype(np.float32)
    return X2, mask
