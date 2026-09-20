"""Part 2 sanity checks (a)-(d). VAL/TRAIN only; test arrays are never loaded into any decision.
Writes results/v2/v2_sanity.json (rendered into V2_DATA_REPORT.md by make_v2_report.py)."""
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.v2.common import *  # noqa
from src.v2 import dataset_v2 as D2
from src.v2.attack_realize import DT_H
from src.reproduction.scma_env import SCMA_MODES
from src.benchmarks.classical_tree import build_estimator
from src.benchmarks.robustness import perturb_noise

CB_TUNED = json.load(open(PROJECT_ROOT / "results/aggregate/benchmark_val_tuning_classical_tree.json"))["catboost"]["best_params"]


def mf1(y, p):
    return float(f1_score(y, p, average="macro", zero_division=0))


def auroc_oriented(y, x):
    a = roc_auc_score(y, x)
    return float(a), float(max(a, 1 - a))


def probe(Xtr, ytr, Xv, yv):
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(class_weight="balanced", max_iter=5000).fit(sc.transform(Xtr), ytr)
    p_lr = lr.predict_proba(sc.transform(Xv))[:, 1]
    cb = build_estimator("catboost", {}, seed=0)  # CatBoost defaults (+ Balanced class weights, as in the v1 benchmark builder)
    cb.fit(Xtr, ytr)
    p_cb = cb.predict_proba(Xv)[:, 1]
    return {"logreg_macroF1": mf1(yv, (p_lr >= .5).astype(int)), "logreg_auroc": float(roc_auc_score(yv, p_lr)),
            "catboost_macroF1": mf1(yv, (p_cb >= .5).astype(int)), "catboost_auroc": float(roc_auc_score(yv, p_cb))}


def main():
    t0 = time.time()
    out = {}
    store = dsmod.SlotStore.get()
    manifest_idx = dsmod.load_manifest().set_index("session_uid")

    # ---------------- (a) single-feature AUROC v1 vs v2 (SCMA-1, val) ----------------
    tr1, va1 = build_v1_train_val(1)
    d2 = D2.build_type_dataset_v2(1)
    X1, y1 = summary_features(va1), va1.labels
    X2, y2 = summary_features(d2["val"]), d2["val"].labels
    a = {}
    for j, nm in enumerate(FEATURE_NAMES):
        r1, o1 = auroc_oriented(y1, X1[:, j])
        r2, o2 = auroc_oriented(y2, X2[:, j])
        a[nm] = {"v1_raw": r1, "v1_oriented": o1, "v2_raw": r2, "v2_oriented": o2}
    out["a_auroc_scma1_val"] = a
    ex_v = D2.load_extras(1)["val"]
    raw = ex_v["raw_summary"]
    lab = d2["val"].labels
    out["a_std_zero_frac_v2_val"] = {"attack_std_lt_1e-6": float((X2[lab == 1, 5] < 1e-6).mean()),
                                     "benign_std_lt_1e-6": float((X2[lab == 0, 5] < 1e-6).mean()),
                                     "attack_std_lt_1e-3": float((X2[lab == 1, 5] < 1e-3).mean()),
                                     "attack_active_empty": float((d2["val"].mask[lab == 1].sum(1) == 0).mean())}
    # v2 catboost (tuned v1 config) clean + noise on v2 SCMA-1 val (extra, val only)
    trd = d2["train"]
    Xt, yt = summary_features(trd), trd.labels
    pstd = float(trd.slots[:, :, 0][trd.mask].std()); estd = float(trd.static[:, 2].std())
    cols_sets = {"full (7)": list(range(7)), "no std, no max": [0, 1, 2, 3, 4]}
    nz = {}
    for nmv, cols in cols_sets.items():
        ms = []
        for s in range(5):
            e = build_estimator("catboost", CB_TUNED, seed=s); e.fit(Xt[:, cols], yt); ms.append(e)
        nz[nmv] = {}
        for lv in [0.0, 0.05, 0.15, 0.30]:
            f = []
            for s, e in enumerate(ms):
                rng = np.random.default_rng(int(lv * 1000) * 100 + s)
                dn = perturb_noise(d2["val"], pstd, estd, lv, rng)
                f.append(mf1(y2, e.predict(summary_features(dn)[:, cols]).astype(int)))
            nz[nmv][str(lv)] = {"mean": float(np.mean(f)), "std": float(np.std(f, ddof=1))}
    out["extra_v2_scma1_catboost_noise_val"] = nz

    # ---------------- (b),(c) per type ----------------
    b, c = {}, {}
    for t in D2.SCMA_TYPES:
        d = D2.build_type_dataset_v2(t)
        ex = D2.load_extras(t)
        z = np.load(D2.cache_path(t))
        pm, ps = float(z["meta__power_mean"][0]), float(z["meta__power_scale"][0])
        b[t], c[t] = {}, {}
        for sp in ("train", "val"):
            x, e = d[sp], ex[sp]
            att = x.labels == 1
            corr = e["corr"][att]
            fin = np.isfinite(corr)
            e_new, e_tgt = np.nan_to_num(e["e_new"][att], nan=0.0), e["e_target"][att]  # empty window -> 0 kWh delivered
            e_att = e["raw_summary"][att, 0]
            ok = e_tgt > 1e-6
            rel = np.abs(e_new[ok] - e_tgt[ok]) / e_tgt[ok]
            rel_att = np.abs(e_new - e_att)[e_att > 1e-6] / e_att[e_att > 1e-6]
            unclipped = ~e["clipped"][att]
            cu = e["corr"][att][unclipped]
            b[t][sp] = {
                "n_attack": int(att.sum()),
                "frac_empty_window": float((e["n_new"][att] == 0).mean()),
                "frac_attack_raw_max_eq_20kW": float((e["raw_summary"][att, 6] >= 20.0 - 1e-3).mean()),
                "frac_benign_raw_max_ge_20kW": float((e["raw_summary"][~att, 6] >= 20.0 - 1e-3).mean()),
                "frac_attack_raw_std_lt_1e-6": float((e["raw_summary"][att, 5] < 1e-6).mean()),
                "frac_benign_raw_std_lt_1e-6": float((e["raw_summary"][~att, 5] < 1e-6).mean()),
                "clipped_energy_shortfall_median": float(np.median(((e_tgt - e_new)[e["clipped"][att] & (e_tgt > 1e-6)]) / e_tgt[e["clipped"][att] & (e_tgt > 1e-6)])) if (e["clipped"][att] & (e_tgt > 1e-6)).any() else float("nan"),
                "e_full_over_kwh_pcts_5_50_95": [float(v) for v in np.nanpercentile(e["e_full"][att] / np.maximum(e["e_orig_kwh"][att], 1e-9), [5, 50, 95])],
                "corr_mean_all": float(np.nanmean(corr)), "corr_median": float(np.nanmedian(corr)),
                "corr_defined_frac": float(fin.mean()),
                "corr_mean_unclipped": float(np.nanmean(cu)),
                "frac_clipped": float(e["clipped"][att].mean()),
                "energy_rel_err_vs_target_median": float(np.median(rel)), "energy_within5pct_of_target": float((rel < 0.05).mean()),
                "energy_rel_err_vs_attacked_agg_median": float(np.median(rel_att)),
                "energy_within5pct_of_attacked_agg": float((rel_att < 0.05).mean()),
                "unclipped_energy_within1pct_of_target": float((rel[unclipped[ok]] < 0.01).mean()) if unclipped[ok].any() else float("nan"),
                "scale_pcts_5_50_95": [float(v) for v in np.nanpercentile(e["scale"][att], [5, 50, 95])],
                "n_new_frac_of_orig_pcts_5_50_95": [float(v) for v in np.nanpercentile(e["n_new"][att] / np.maximum(e["n_orig"][att], 1), [5, 50, 95])],
                "c_attack_pcts_5_50_95": [float(v) for v in np.percentile(e["c_attack"][att], [5, 50, 95])],
            }
        # (c) mask checks on train + val attacked bags
        chk = {"n": 0, "inactive_zero": 0, "single_run": 0, "empty": 0, "window_slot_count_match": 0,
               "subset_of_real_mask": 0, "type_specific_edge_ok": 0, "shape_exact_match_unclipped": 0, "n_unclipped_checked": 0,
               "mask_count_eq_extras_n_new": 0}
        modes = SCMA_MODES[t]
        zero_scaled = (0.0 - pm) / ps
        for sp in ("train", "val"):
            x, e = d[sp], ex[sp]
            for i in np.flatnonzero(x.labels == 1):
                chk["n"] += 1
                m = x.mask[i]
                sl = x.slots[i, :, 0]
                rp, ra = store.get_slots(str(e["src_uid"][i]))
                if np.allclose(sl[~m], zero_scaled, atol=1e-4):
                    chk["inactive_zero"] += 1
                if m.sum() == 0:
                    chk["empty"] += 1
                    chk["single_run"] += 1
                    chk["subset_of_real_mask"] += 1
                    chk["type_specific_edge_ok"] += 1
                    chk["window_slot_count_match"] += 1
                    chk["mask_count_eq_extras_n_new"] += int(e["n_new"][i] == 0)
                    continue
                idx = np.flatnonzero(m)
                if idx[-1] - idx[0] + 1 == m.sum():
                    chk["single_run"] += 1
                ridx = np.flatnonzero(ra)
                if not (m & ~ra).any():
                    chk["subset_of_real_mask"] += 1
                chk["mask_count_eq_extras_n_new"] += int(m.sum() == e["n_new"][i])
                rs = e["raw_summary"][i]  # e, t_avail, t_start, t_end, mean, std, max (attacked aggregate, raw)
                slot_h = np.arange(288) * DT_H
                cnt = int(((slot_h >= rs[2]) & (slot_h < rs[3])).sum())
                if abs(cnt - m.sum()) <= 1 or abs(rs[3] - rs[2]) >= 24 - 1e-6:
                    chk["window_slot_count_match"] += 1
                ok_edge = True
                if "start" not in modes:
                    ok_edge &= (idx[0] == ridx[0])
                if "end" not in modes:
                    ok_edge &= (idx[-1] == ridx[-1])
                if modes == ("demand",):
                    ok_edge &= bool((m == ra).all())
                chk["type_specific_edge_ok"] += int(ok_edge)
                if not e["clipped"][i] and e["scale"][i] > 0:
                    chk["n_unclipped_checked"] += 1
                    rawp = sl[m] * ps + pm
                    ref = rp[ridx[0]:ridx[0] + int(e["n_orig"][i])][:m.sum()] * e["scale"][i]
                    if np.allclose(rawp, ref, rtol=1e-3, atol=1e-3):
                        chk["shape_exact_match_unclipped"] += 1
        c[t] = chk
        print(f"[sanity] SCMA-{t} done ({time.time()-t0:.0f}s)")
    out["b_shape_energy"] = b
    out["c_mask"] = c

    # ---------------- (d) separability probes per type, v2 (and v1 for context) ----------------
    dd = {}
    for t in D2.SCMA_TYPES:
        d = D2.build_type_dataset_v2(t)
        Xt_, yt_ = summary_features(d["train"]), d["train"].labels
        Xv_, yv_ = summary_features(d["val"]), d["val"].labels
        p2 = probe(Xt_, yt_, Xv_, yv_)
        tr1_, va1_ = build_v1_train_val(t)
        p1 = probe(summary_features(tr1_), tr1_.labels, summary_features(va1_), va1_.labels)
        nostd = probe(Xt_[:, [0, 1, 2, 3, 4]], yt_, Xv_[:, [0, 1, 2, 3, 4]], yv_)
        # AUROC of each feature in v2 val
        fa = {nm: auroc_oriented(yv_, Xv_[:, j])[1] for j, nm in enumerate(FEATURE_NAMES)}
        dd[t] = {"v2": p2, "v1": p1, "v2_no_std_no_max": nostd, "v2_feature_auroc_oriented": fa}
        print(f"[probe] SCMA-{t} v2 {p2}  v1 {p1}")
    out["d_probe"] = dd
    json.dump(out, open(RESULTS_V2 / "v2_sanity.json", "w"), indent=2, default=float)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
