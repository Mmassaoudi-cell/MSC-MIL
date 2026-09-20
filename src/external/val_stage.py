"""VALIDATION-ONLY stage before the test evaluation: (1) 12-trial TPE tuning of tree_aug (xgboost) and msc_aug_tuned
on the val score; (2) train msc_e4 + tree_aug (+ msc_aug_tuned) seeds 0-4 into the cache; (3) choose the fusion weight
on validation. Test split is never loaded. python -m src.external.val_stage --dataset cic2017"""
import argparse, json, time
import numpy as np, optuna, torch
from src.external.data import load, summary_features
from src.external.prep_common import RES
from src.external import aug, zoo
from src.external.tune import space

SEEDS = (0, 1, 2, 3, 4); GRID = [round(0.1 * i, 1) for i in range(11)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); a = ap.parse_args(); ds = a.dataset
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = load(ds, ("train", "val")); tr, va = D["train"], D["val"]
    ctx = zoo.Ctx(ds, tr, va, D["meta"], dev)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    out = RES / f"tuned_configs_aug_{ds}.json"
    cfg = json.loads(out.read_text()) if out.exists() else {}
    # ---- tune tree_aug (xgboost space)
    if "tree_aug" not in cfg:
        Saug = ctx.Saug; t0 = time.time()
        def obj(t):
            p = space("xgboost", t)
            f = aug.fit_tree_aug("xgboost", p, tr, 0, ctx.fstd, ctx.cont, Saug)
            s, c, n = aug.val_score(f.predict_proba, va, ctx.noisy_val)
            print(f"[{ds}/tree_aug] trial {t.number} score={s:.4f} clean={c:.4f} noisy={n:.4f}", flush=True); return s
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0)); st.optimize(obj, n_trials=12)
        cfg["tree_aug"] = {"params": st.best_params, "val_score": st.best_value, "secs": time.time() - t0}
        out.write_text(json.dumps(cfg, indent=1)); ctx.augc = cfg
    # ---- tune msc_aug_tuned
    if "msc_aug_tuned" not in cfg:
        t0 = time.time()
        def obj(t):
            p = space("msc_mil", t); t1 = time.time()
            f, info = aug.fit_deep_e4("msc_mil", p, tr, va, 0, dev, ctx.fstd, ctx.cont, ctx.noisy_val)
            s = info.get("val_score", 0.0) if f is not None else 0.0
            print(f"[{ds}/msc_aug_tuned] trial {t.number} score={s:.4f} {p} ({time.time()-t1:.0f}s)", flush=True); return s
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0)); st.optimize(obj, n_trials=12)
        cfg["msc_aug_tuned"] = {"params": st.best_params, "val_score": st.best_value, "secs": time.time() - t0}
        out.write_text(json.dumps(cfg, indent=1)); ctx.augc = cfg
    # ---- train/cached models, val probabilities, fusion weight
    vp = {}
    for name in ("tree_aug", "msc_e4", "msc_aug_tuned"):
        t0 = time.time()
        for s in SEEDS:
            f = zoo.get(ctx, name, s)
            vp[(name, s)] = [f.predict_proba(va["X"], None)] + [f.predict_proba(X, None) for X in ctx.noisy_val]
        print(f"[{ds}] trained/loaded {name} x5 ({time.time()-t0:.0f}s)", flush=True)
    from src.external.models import macro_f1
    def score(w):
        vals = []
        for s in SEEDS:
            ps = [w * a_ + (1 - w) * b_ for a_, b_ in zip(vp[("tree_aug", s)], vp[("msc_e4", s)])]
            f = [macro_f1(va["y"], (p >= 0.5).astype(int)) for p in ps]
            vals.append(0.5 * f[0] + 0.5 * float(np.mean(f[1:])))
        return float(np.mean(vals))
    sc = {w: score(w) for w in GRID}
    best = max(sc.values()); wbest = min([w for w in GRID if abs(sc[w] - best) < 1e-12], key=lambda w: abs(w - 0.5))
    (RES / f"fusion_weights_{ds}.json").write_text(json.dumps({"w_tree": wbest, "val_score_grid": sc, "grid": GRID,
        "selection": "max mean-over-seeds val score (0.5 clean + 0.5 mean noisy 5/15/30 macro-F1@0.5); ties -> w closest to 0.5"}, indent=1))
    print(f"[{ds}] fusion weight on tree = {wbest}; grid = {sc}", flush=True)


if __name__ == "__main__":
    main()
