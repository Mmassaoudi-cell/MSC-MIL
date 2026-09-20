"""Part 1: diagnose the flat-rate shortcut on v1 data, TRAIN/VAL only."""
import json
import time
import time

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, f1_score, make_scorer

from src.v2.common import *  # noqa
from src.benchmarks.robustness import perturb_noise  # the exact v1 noise protocol
from src.benchmarks.classical_tree import build_estimator

CB = json.load(open(PROJECT_ROOT / "results/aggregate/benchmark_val_tuning_classical_tree.json"))["catboost"]["best_params"]
SEEDS = (0, 1, 2, 3, 4)
LEVELS = [0.0, 0.05, 0.15, 0.30]
VARIANTS = {
    "full (7)": [0, 1, 2, 3, 4, 5, 6],
    "no std": [0, 1, 2, 3, 4, 6],
    "no max": [0, 1, 2, 3, 4, 5],
    "no std, no max": [0, 1, 2, 3, 4],
    "no std/max/mean (extra)": [0, 1, 2, 3],
    "std only (extra)": [5],
}


def mf1(y, p):
    return float(f1_score(y, p, average="macro", zero_division=0))


def main():
    t0 = time.time()
    train, val = build_v1_train_val(1)
    Xtr, ytr = summary_features(train), train.labels
    Xv, yv = summary_features(val), val.labels
    print(f"train {Xtr.shape} pos={ytr.mean():.3f}; val {Xv.shape} pos={yv.mean():.3f}  ({time.time()-t0:.0f}s)")
    out = {"catboost_params": CB, "n_train": int(len(ytr)), "n_val": int(len(yv)), "train_pos_rate": float(ytr.mean()), "val_pos_rate": float(yv.mean())}

    dist = {}
    for c, nm in ((0, "benign"), (1, "attack")):
        dist[nm] = {FEATURE_NAMES[j]: [float(v) for v in np.percentile(Xv[yv == c, j], [0, 5, 50, 95, 100])] for j in (4, 5, 6)}
    for thr in ("1e-6", "1e-3"):
        t = float(thr)
        dist[f"attack_frac_std_lt_{thr}"] = float((Xv[yv == 1, 5] < t).mean())
        dist[f"benign_frac_std_lt_{thr}"] = float((Xv[yv == 0, 5] < t).mean())
    out["val_feature_percentiles(0,5,50,95,100)"] = dist

    au = {}
    for j, nm in enumerate(FEATURE_NAMES):
        a = roc_auc_score(yv, Xv[:, j])
        au[nm] = {"auroc_raw": float(a), "auroc_oriented": float(max(a, 1 - a)), "attack_higher": bool(a >= 0.5)}
    out["single_feature_auroc_val"] = au

    nat = []
    for s in SEEDS:
        est = build_estimator("catboost", CB, seed=s)
        est.fit(Xtr, ytr)
        nat.append(est.get_feature_importance())
        if s == 0:
            r = permutation_importance(est, Xv, yv, scoring=make_scorer(mf1), n_repeats=10, random_state=0)
            perm = (r.importances_mean, r.importances_std)
            base = mf1(yv, est.predict(Xv))
    nat = np.array(nat)
    out["importance"] = {
        "native_PredictionValuesChange_mean_over_5seeds": dict(zip(FEATURE_NAMES, map(float, nat.mean(0)))),
        "native_std": dict(zip(FEATURE_NAMES, map(float, nat.std(0)))),
        "permutation_macroF1_drop_val_seed0": dict(zip(FEATURE_NAMES, map(float, perm[0]))),
        "permutation_std": dict(zip(FEATURE_NAMES, map(float, perm[1]))),
        "seed0_val_macroF1": base,
    }

    power_std = float(train.slots[:, :, 0][train.mask].std())
    energy_std = float(train.static[:, 2].std())
    out["noise_refs"] = {"power_std": power_std, "energy_std": energy_std}
    res = {}
    for vname, cols in VARIANTS.items():
        models = []
        for s in SEEDS:
            est = build_estimator("catboost", CB, seed=s)
            est.fit(Xtr[:, cols], ytr)
            models.append(est)
        res[vname] = {}
        for lv in LEVELS:
            f1s, recs, fars = [], [], []
            for s, est in zip(SEEDS, models):
                rng = np.random.default_rng(int(lv * 1000) * 100 + s)
                dn = perturb_noise(val, power_std, energy_std, lv, rng)
                Xn = summary_features(dn)
                p = est.predict(Xn[:, cols]).astype(int)
                f1s.append(mf1(yv, p))
                recs.append(float(((p == 1) & (yv == 1)).sum() / (yv == 1).sum()))
                fars.append(float(((p == 1) & (yv == 0)).sum() / (yv == 0).sum()))
            res[vname][str(lv)] = {"macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s, ddof=1)),
                                    "recall_mean": float(np.mean(recs)), "far_mean": float(np.mean(fars)), "per_seed_f1": f1s}
        print(vname, {k: round(v["macro_f1_mean"], 4) for k, v in res[vname].items()})
    out["noise_val"] = res
    json.dump(out, open(RESULTS_V2 / "diagnosis_v1.json", "w"), indent=2)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
