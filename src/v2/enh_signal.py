"""Step 0: what separates attacked from benign bags in v2 (train fit / VAL analysis only)."""
from __future__ import annotations
import json, sys
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from catboost import CatBoostClassifier
from src.v2.enh_data import load_v2_trainval
from src.v2.common import RESULTS_V2, summary_features

DT = 5 / 60.0
NAMES = ["e_decl", "e_curve", "e_resid_rel", "e_ratio", "pmr(max/mean)", "load_factor(mean/max)", "clip_frac(>=19.9kW)",
         "n_active_h", "win_over_tavail", "cv(std/mean)", "plateau_frac(>=0.95max)", "max_slot_jump", "zero_active_frac",
         "first_slot_over_max", "raw_max", "raw_mean", "raw_std", "t_avail", "conn_hod"]


def phys(d, meta):
    p = d.slots[:, :, 0].astype(np.float64) * meta["power_scale"][0] + meta["power_mean"][0]
    m = d.mask
    st = d.static.astype(np.float64) * meta["static_scale"] + meta["static_mean"]
    n = m.sum(1); sn = np.maximum(n, 1)
    pa = np.where(m, p, 0.0)
    ecurve = pa.sum(1) * DT
    edecl = st[:, 2]
    mean = pa.sum(1) / sn
    mx = np.where(m, p, -np.inf).max(1); mx[n == 0] = 0
    std = np.sqrt((((p - mean[:, None]) ** 2) * m).sum(1) / sn)
    eps = 1e-6
    jump = np.abs(np.diff(p, axis=1)) * (m[:, 1:] & m[:, :-1])
    first = np.array([p[i, np.argmax(m[i])] if n[i] else 0 for i in range(len(p))])
    F = np.stack([
        edecl, ecurve, (ecurve - edecl) / np.maximum(edecl, 1.0), ecurve / np.maximum(edecl, 0.5),
        mx / np.maximum(mean, eps), mean / np.maximum(mx, eps), (np.where(m, p, 0) >= 19.9).sum(1) / sn,
        n * DT, n * DT / np.maximum(st[:, 3], 0.1), std / np.maximum(mean, eps),
        (np.where(m, p, 0) >= 0.95 * mx[:, None]).sum(1) / sn, jump.max(1), ((p <= 1e-6) & m).sum(1) / sn,
        first / np.maximum(mx, eps), mx, mean, std, st[:, 3], st[:, 0]], 1)
    return F


def oauc(x, y):
    a = roc_auc_score(y, np.nan_to_num(x, nan=0, posinf=1e6, neginf=-1e6))
    return a, max(a, 1 - a)


def main():
    out = {}
    lines = []
    for t in (1, 4):
        data, meta = load_v2_trainval(t)
        tr, va = data["train"], data["val"]
        Ftr, Fva = phys(tr, meta), phys(va, meta)
        S_tr, S_va = summary_features(tr), summary_features(va)
        y, ytr = va.labels, tr.labels
        res = {"aucs": {}, "n_val": int(len(y)), "n_pos": int(y.sum())}
        for j, nme in enumerate(NAMES):
            a, o = oauc(Fva[:, j], y)
            res["aucs"][nme] = dict(raw=a, oriented=o, mean_benign=float(Fva[y == 0, j].mean()), mean_attack=float(Fva[y == 1, j].mean()))
        # catboost base vs extended
        cfgs = {}
        for tag, (Xtr, Xva, names) in {"base7": (S_tr, S_va, ["e_demand", "t_avail", "conn_hod", "t_end", "pmean", "pstd", "pmax"]),
                                        "ext(7+phys)": (np.hstack([S_tr, Ftr]), np.hstack([S_va, Fva]), ["e_demand", "t_avail", "conn_hod", "t_end", "pmean", "pstd", "pmax"] + NAMES),
                                        "phys_only": (Ftr, Fva, NAMES)}.items():
            f1s, imps, aucs = [], [], []
            for s in range(3):
                cb = CatBoostClassifier(depth=4, iterations=200, learning_rate=0.1, auto_class_weights="Balanced", random_seed=s, verbose=0)
                cb.fit(Xtr, ytr)
                pr = cb.predict_proba(Xva)[:, 1]
                f1s.append(f1_score(y, pr >= 0.5, average="macro")); aucs.append(roc_auc_score(y, pr))
                imps.append(cb.get_feature_importance())
            # permutation importance (seed 2 model), macro-F1 drop
            rng = np.random.default_rng(0)
            base = f1_score(y, cb.predict_proba(Xva)[:, 1] >= 0.5, average="macro")
            perm = []
            for j in range(Xva.shape[1]):
                dr = []
                for _ in range(5):
                    Xp = Xva.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
                    dr.append(base - f1_score(y, cb.predict_proba(Xp)[:, 1] >= 0.5, average="macro"))
                perm.append(float(np.mean(dr)))
            cfgs[tag] = dict(f1_mean=float(np.mean(f1s)), f1_std=float(np.std(f1s)), auroc=float(np.mean(aucs)),
                             native_imp=dict(zip(names, np.mean(imps, 0).round(2).tolist())), perm_imp=dict(zip(names, np.round(perm, 4).tolist())))
        res["catboost"] = cfgs
        # energy consistency detail
        ex = load_v2_trainval(t, with_extras=True)[2]["val"]
        res["energy_consistency"] = {
            "resid_rel_benign_p5_p50_p95": np.percentile(Fva[y == 0, 2], [5, 50, 95]).round(4).tolist(),
            "resid_rel_attack_p5_p50_p95": np.percentile(Fva[y == 1, 2], [5, 50, 95]).round(4).tolist(),
            "frac_attack_clipped": float(ex["clipped"][y == 1].mean()),
            "frac_attack_empty": float((va.mask[y == 1].sum(1) == 0).mean()),
            "frac_benign_empty": float((va.mask[y == 0].sum(1) == 0).mean()),
            "n_active_h_benign_p5_50_95": np.percentile(Fva[y == 0, 7], [5, 50, 95]).round(3).tolist(),
            "n_active_h_attack_p5_50_95": np.percentile(Fva[y == 1, 7], [5, 50, 95]).round(3).tolist(),
            "win_over_tavail_benign_p5_50_95": np.percentile(Fva[y == 0, 8], [5, 50, 95]).round(3).tolist(),
            "win_over_tavail_attack_p5_50_95": np.percentile(Fva[y == 1, 8], [5, 50, 95]).round(3).tolist(),
        }
        # sensitivity of each phys feature to 15% legacy noise (relative change in AUROC)
        from src.benchmarks.robustness import perturb_noise
        pstd = float(tr.slots[:, :, 0][tr.mask].std()); estd = float(tr.static[:, 2].std())
        rng = np.random.default_rng(1500)
        vn = perturb_noise(va, pstd, estd, 0.15, rng)
        Fn = phys(vn, meta)
        res["auc_under_15pct_noise"] = {nme: oauc(Fn[:, j], y)[1] for j, nme in enumerate(NAMES)}
        out[f"SCMA-{t}"] = res
    (RESULTS_V2 / "signal_analysis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:12000])


if __name__ == "__main__":
    main()
