"""Train-or-load (on-disk cache) for every evaluated model; resumable. Uses train + val ONLY (val for early stopping)."""
from __future__ import annotations
import json, pickle
import numpy as np, torch
from src.external.prep_common import RES
from src.external.models import DEEP, TREE, build_deep, fit_model, FittedDeep, FittedTree
from src.external import aug

BASE = ["logreg", "random_forest", "xgboost", "lightgbm", "catboost", "plain_lstm_mil", "bilstm_attn_mil", "transformer_mil", "msc_mil"]
NEW = ["tree_aug", "msc_aug_tuned", "msc_e4"]
ALL = BASE + NEW
DEEPNAMES = DEEP + ["msc_aug_tuned", "msc_e4"]


def cfgs(ds):
    base = json.loads((RES / f"tuned_configs_{ds}.json").read_text())
    f = RES / f"tuned_configs_aug_{ds}.json"
    return base, (json.loads(f.read_text()) if f.exists() else {})


class Ctx:
    def __init__(self, ds, tr, va, meta, dev):
        self.ds, self.tr, self.va, self.dev = ds, tr, va, dev
        self.cont = np.array(meta["cont_mask"]); self.fstd = aug.train_feature_std(tr["X"])
        self.noisy_val = aug.val_sets(va, self.fstd, self.cont)
        self.base, self.augc = cfgs(ds)
        self._S = None; self._Saug = None
        self.dir = RES / "ckpt" / ds; self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def S(self):
        from src.external.data import summary_features
        if self._S is None: self._S = summary_features(self.tr["X"])
        return self._S

    @property
    def Saug(self):
        if self._Saug is None: self._Saug = aug.build_tree_aug_matrix(self.tr, self.fstd, self.cont)
        return self._Saug


def get(ctx: Ctx, name, seed):
    """returns fitted object with predict_proba(X, mask)"""
    d = ctx.tr["X"].shape[2]; W = ctx.tr["X"].shape[1]
    if name in ("tree_aug",) or name in TREE:
        f = ctx.dir / f"{name}_s{seed}.pkl"
        if f.exists(): return FittedTree(pickle.loads(f.read_bytes()))
        if name == "tree_aug":
            fit = aug.fit_tree_aug("xgboost", ctx.augc["tree_aug"]["params"], ctx.tr, seed, ctx.fstd, ctx.cont, ctx.Saug)
        else:
            fit = fit_model(name, ctx.base[name]["params"], ctx.tr, ctx.va, seed, ctx.dev, ctx.S)[0]
        f.write_bytes(pickle.dumps(fit.est)); return fit
    # deep / logreg pipeline handled above
    if name == "logreg":
        pass
    f = ctx.dir / f"{name}_s{seed}.pt"
    if name == "msc_e4": p, arch = ctx.base["msc_mil"]["params"], "msc_mil"
    elif name == "msc_aug_tuned": p, arch = ctx.augc["msc_aug_tuned"]["params"], "msc_mil"
    else: p, arch = ctx.base[name]["params"], name
    if f.exists():
        m = build_deep(arch, d, W, p).to(ctx.dev); m.load_state_dict(torch.load(f, map_location=ctx.dev)); return FittedDeep(m, ctx.dev)
    if name in ("msc_e4", "msc_aug_tuned"):
        fit, info = aug.fit_deep_e4(arch, p, ctx.tr, ctx.va, seed, ctx.dev, ctx.fstd, ctx.cont, ctx.noisy_val)
    else:
        fit, info = fit_model(name, p, ctx.tr, ctx.va, seed, ctx.dev)
    if fit is None: raise RuntimeError(f"{name} seed {seed} diverged: {info}")
    torch.save(fit.model.state_dict(), f)
    (ctx.dir / f"{name}_s{seed}.info.json").write_text(json.dumps({k: v for k, v in info.items() if k != "history"}, default=float))
    return fit
