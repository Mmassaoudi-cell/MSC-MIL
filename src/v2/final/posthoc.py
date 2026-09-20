"""POST-HOC DIAGNOSTIC analysis of the final v2 test results. Analysis only: no model, config or selection is changed.
usage: python -m src.v2.final.posthoc [stage ...]   stages: shift errors svm recenter (default all). Outputs -> results/v2/final/posthoc/ (resumable cache)."""
from __future__ import annotations

import pickle, sys, warnings
import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp
from sklearn.metrics import f1_score

from src.v2 import baselines_lib as L
from src.v2 import enhanced_model as em
from src.v2.common import DATA_V2, summary_features
from src.v2.final import run_eval as R

warnings.filterwarnings("ignore")
PH = R.FIN / "posthoc"; CACHE = PH / "cache"
PH.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
KEY = ["pi_fusion", "e4_net", "msc_mil", "source_bilstm_attn", "xgboost_pf", "catboost_pf", "random_forest", "xgboost", "svm_rbf"]
SITES = ["caltech", "jpl", "office_01"]
FEATS = ["e_demand", "t_avail", "conn_hod", "t_end", "pow_mean", "pow_std", "pow_max",
         "phys_energy_resid", "phys_log_peak_mean", "phys_load_factor", "phys_clip_frac", "phys_log_hours", "phys_hours_over_tavail", "phys_plateau_frac", "phys_log_cv"]
DEV = L.get_device()


def site_of(uids):
    return np.array([u.split("::")[0].split("/")[0] for u in uids])


def load_all(t):
    z = np.load(DATA_V2 / f"scma{t}_v2.npz", allow_pickle=False)
    out = {}
    for sp in ("train", "val", "test_full"):
        d = L.D(z[f"{sp}__slots"], z[f"{sp}__mask"], z[f"{sp}__static"], z[f"{sp}__labels"])
        out[sp] = (d, site_of([str(u) for u in z[f"{sp}__uids"]]))
    meta = {k: z[f"meta__{k}"] for k in ("power_mean", "power_scale", "static_mean", "static_scale")}
    return out, meta


# ------------------------------------------------------------------ stage 1
def psi(a, b, nb=10):
    edges = np.unique(np.quantile(a, np.linspace(0, 1, nb + 1)[1:-1]))
    ca = np.bincount(np.searchsorted(edges, a), minlength=len(edges) + 1) / len(a)
    cb = np.bincount(np.searchsorted(edges, b), minlength=len(edges) + 1) / len(b)
    ca, cb = np.clip(ca, 1e-4, None), np.clip(cb, 1e-4, None)
    return float(((cb - ca) * np.log(cb / ca)).sum())


def cmp(a, b, rng):
    sd = np.sqrt((a.var() + b.var()) / 2)
    smd = (b.mean() - a.mean()) / sd if sd > 0 else np.nan
    aa = a if len(a) < 200000 else rng.choice(a, 200000, replace=False)
    bb = b if len(b) < 200000 else rng.choice(b, 200000, replace=False)
    return smd, float(ks_2samp(aa, bb).statistic), psi(a, b)


def stage_shift():
    data, meta = load_all(1)
    rng = np.random.default_rng(0)
    ben = {}
    for sp, (d, st) in data.items():
        m = d.labels == 0
        ben[sp] = (d.slots[m], d.mask[m], d.static[m], st[m])
    F, P = {}, {}
    for sp, (sl, mk, sta, st) in ben.items():
        F[sp] = em.tree_features(sl, mk, sta, meta)
        P[sp] = sl[..., 0]
    share = []
    for sp in ben:
        s = ben[sp][3]
        for site in SITES:
            share.append(dict(split=sp, site=site, n_benign=int((s == site).sum()), share=float((s == site).mean())))
    pd.DataFrame(share).to_csv(PH / "posthoc_site_share.csv", index=False)
    rows = []
    for site in ["all"] + SITES:
        sel = {sp: (np.ones(len(ben[sp][3]), bool) if site == "all" else ben[sp][3] == site) for sp in ben}
        for tgt in ("val", "test_full"):
            if sel["train"].sum() < 20 or sel[tgt].sum() < 20:
                continue
            for j, name in enumerate(FEATS):
                smd, ks, p = cmp(F["train"][sel["train"], j], F[tgt][sel[tgt], j], rng)
                rows.append(dict(site=site, comparison=f"train->{tgt}", feature=name, smd=smd, ks=ks, psi=p))
            for name, arr_fn in (("slot_power_active", lambda sp, s: P[sp][s][ben[sp][1][s]]), ("static_energy_scaled", lambda sp, s: ben[sp][2][s][:, 2])):
                smd, ks, p = cmp(arr_fn("train", sel["train"]), arr_fn(tgt, sel[tgt]), rng)
                rows.append(dict(site=site, comparison=f"train->{tgt}", feature=name, smd=smd, ks=ks, psi=p))
    pd.DataFrame(rows).to_csv(PH / "posthoc_shift_benign.csv", index=False)
    print("shift done", flush=True)


# ------------------------------------------------------------------ predictors
def make_pred(model, t, s, meta):
    if model in R.TREES:
        est = pickle.load(open(R.CKPT / f"{model}_scma{t}_seed{s}.pkl", "rb"))
        return lambda d: est.predict_proba(R.tree_feats(model, d, meta))[:, 1]
    if model in ("pi_fusion", "e4_net"):
        net = em.EnhancedMSCMIL(meta, physics=False, dropout=em.FROZEN_NET_CFG["dropout"])
        net.load_state_dict(torch.load(R.CKPT / f"pi_fusion_scma{t}_seed{s}_net.pt", map_location="cpu")); net.to(DEV).eval()
        if model == "e4_net":
            return lambda d: em._net_proba(net, d, DEV)
        xgb = pickle.load(open(R.CKPT / f"pi_fusion_scma{t}_seed{s}_xgb.pkl", "rb"))
        return lambda d: em.FROZEN_W_TREE * xgb.predict_proba(em.tree_features(d.slots, d.mask, d.static, meta))[:, 1] + (1 - em.FROZEN_W_TREE) * em._net_proba(net, d, DEV)
    net = L.build_deep(model, R.FROZEN[model]["best_params"])
    net.load_state_dict(torch.load(R.CKPT / f"{model}_scma{t}_seed{s}.pt", map_location="cpu")); net.to(DEV).eval()
    return lambda d: L.infer(net, d, DEV)


def seedavg(model, t, d, meta, tag):
    fp = CACHE / f"{tag}_{model}_scma{t}.npy"
    if fp.exists():
        return np.load(fp)
    p = np.mean([make_pred(model, t, s, meta)(d) for s in range(R.n_seeds(model, t))], axis=0)
    np.save(fp, p)
    return p


def rates(y, p, thr=0.5):
    pr = p >= thr
    rec = (pr & (y == 1)).sum() / max((y == 1).sum(), 1); far = (pr & (y == 0)).sum() / max((y == 0).sum(), 1)
    return float(rec), float(far), float(f1_score(y, pr.astype(int), average="macro"))


# ------------------------------------------------------------------ stage 2
def stage_errors():
    rows, srows = [], []
    for t in (1, 2, 4):
        data, meta = load_all(t)
        dv, dt = data["val"][0], data["test_full"][0]; st_t = data["test_full"][1]
        for m in KEY:
            pv = seedavg(m, t, dv, meta, "val")
            pt = np.mean([np.load(R.PREDS / f"{m}_scma{t}_seed{s}.npy") for s in range(R.n_seeds(m, t))], axis=0)
            for name, y, p in (("val", dv.labels, pv), ("test_full", dt.labels, pt)):
                rec, far, f1 = rates(y, p)
                rows.append(dict(model=m, scma=t, split=name, recall=rec, far=far, macro_f1=f1, mean_p_benign=float(p[y == 0].mean())))
            for site in SITES:
                b = (dt.labels == 0) & (st_t == site); a = (dt.labels == 1) & (st_t == site)
                srows.append(dict(model=m, scma=t, site=site, n_benign=int(b.sum()), far=float((pt[b] >= .5).mean()) if b.sum() else np.nan,
                                  n_attack=int(a.sum()), recall=float((pt[a] >= .5).mean()) if a.sum() else np.nan))
        print("errors scma", t, flush=True)
    pd.DataFrame(rows).to_csv(PH / "posthoc_error_decomposition.csv", index=False)
    pd.DataFrame(srows).to_csv(PH / "posthoc_far_by_site_test.csv", index=False)


# ------------------------------------------------------------------ stage 3
def stage_svm():
    rng = np.random.default_rng(0)
    rows = []
    for t in (1, 2, 4):
        data, meta = load_all(t)
        dv = data["val"][0]; y = dv.labels
        Xv = R.tree_feats("svm_rbf", dv, meta)
        for m in ("svm_rbf", "xgboost", "random_forest"):
            for s in range(3):
                est = pickle.load(open(R.CKPT / f"{m}_scma{t}_seed{s}.pkl", "rb"))
                base = f1_score(y, (est.predict_proba(Xv)[:, 1] >= .5).astype(int), average="macro")
                for j, name in enumerate(FEATS[:7]):
                    drops = []
                    for _ in range(10):
                        X2 = Xv.copy(); X2[:, j] = rng.permutation(X2[:, j])
                        drops.append(base - f1_score(y, (est.predict_proba(X2)[:, 1] >= .5).astype(int), average="macro"))
                    rows.append(dict(scma=t, model=m, seed=s, feature=name, val_macro_f1=base, perm_drop=float(np.mean(drops))))
    pi = pd.DataFrame(rows).groupby(["scma", "model", "feature"]).perm_drop.mean().reset_index()
    pi.to_csv(PH / "posthoc_permutation_importance_val.csv", index=False)
    # FAR of test benign as a function of distance from the train-benign feature cloud (max |z| over the 7 frozen features)
    out = []
    for t in (1, 2, 4):
        d, meta = load_all(t)
        tr, te = d["train"][0], d["test_full"][0]
        Xtr = R.tree_feats("svm_rbf", tr, meta)[tr.labels == 0]
        mu, sd = Xtr.mean(0), Xtr.std(0)
        Xte = R.tree_feats("svm_rbf", te, meta); b = te.labels == 0
        z = np.abs((Xte - mu) / sd).max(1)
        bins = [0, 2, 3, 5, 10, 1e9]; lab = ["<2", "2-3", "3-5", "5-10", ">10"]
        cut = pd.cut(z[b], bins, labels=lab)
        for m in ("svm_rbf", "xgboost", "random_forest", "xgboost_pf", "pi_fusion", "msc_mil"):
            p = np.mean([np.load(R.PREDS / f"{m}_scma{t}_seed{s}.npy") for s in range(R.n_seeds(m, t))], axis=0)[b]
            for l in lab:
                k = (cut == l)
                out.append(dict(scma=t, model=m, zbin=l, n_benign=int(k.sum()), far=float((p[k] >= .5).mean()) if k.sum() else np.nan, mean_p=float(p[k].mean()) if k.sum() else np.nan))
        # attacks: recall by the same bins
        za = z[~b]
        for m in ("svm_rbf", "xgboost", "pi_fusion"):
            p = np.mean([np.load(R.PREDS / f"{m}_scma{t}_seed{s}.npy") for s in range(R.n_seeds(m, t))], axis=0)[~b]
            cuta = pd.cut(za, bins, labels=lab)
            for l in lab:
                k = (cuta == l)
                out.append(dict(scma=t, model=m + "(attacks:recall)", zbin=l, n_benign=int(k.sum()), far=float((p[k] >= .5).mean()) if k.sum() else np.nan, mean_p=float(p[k].mean()) if k.sum() else np.nan))
    pd.DataFrame(out).to_csv(PH / "posthoc_far_by_distance_from_train_benign.csv", index=False)
    # SVM model summary
    est = pickle.load(open(R.CKPT / "svm_rbf_scma1_seed0.pkl", "rb"))
    pd.DataFrame([dict(gamma=est.gamma, C=est.C, n_support=int(est.n_support_.sum()), features="7 summary features (frozen set; same as the 7-feature set, no PF variant of SVM exists)")]).to_csv(PH / "posthoc_svm_info.csv", index=False)
    print("svm done", flush=True)


# ------------------------------------------------------------------ stage 4
def recenter(dte, site_te, ref_stats, src_stats):
    """Per-site affine map of static columns and active-slot power (scaled units): test-site stats -> train-benign-site stats."""
    sl, st = dte.slots.copy(), dte.static.copy()
    for site in SITES:
        r = site_te == site
        if not r.any():
            continue
        (pm_r, ps_r, sm_r, ss_r), (pm_s, ps_s, sm_s, ss_s) = ref_stats[site], src_stats[site]
        a = sl[r, :, 0]; m = dte.mask[r]
        sl[r, :, 0] = np.where(m, (a - pm_s) / ps_s * ps_r + pm_r, a)
        st[r] = (st[r] - sm_s) / ss_s * ss_r + sm_r
    return L.D(sl.astype(np.float32), dte.mask, st.astype(np.float32), dte.labels)


def site_stats(d, site, sel):
    out = {}
    for s in SITES:
        r = sel & (site == s)
        pw = d.slots[r][:, :, 0][d.mask[r]]
        out[s] = (float(pw.mean()), float(pw.std()), d.static[r].mean(0), d.static[r].std(0) + 1e-9)
    return out


def stage_recenter():
    rows = []
    for t in (1, 2, 4):
        data, meta = load_all(t)
        tr, st_tr = data["train"]; te, st_te = data["test_full"]
        ref = site_stats(tr, st_tr, tr.labels == 0)
        srcA = site_stats(te, st_te, te.labels == 0)          # benign test statistics (as requested; uses labels to pick benign)
        srcB = site_stats(te, st_te, np.ones(len(te.labels), bool))   # all test sessions of the site (strictly label-free, includes attacks)
        variants = {"none": te, "recentered_benign_stats": recenter(te, st_te, ref, srcA), "recentered_all_sessions_stats": recenter(te, st_te, ref, srcB)}
        for m in KEY:
            preds = {}
            for s in range(R.n_seeds(m, t)):
                f = make_pred(m, t, s, meta)
                for v, d in variants.items():
                    preds.setdefault(v, []).append(f(d))
            for v, ps in preds.items():
                p = np.mean(ps, axis=0); rec, far, f1 = rates(te.labels, p)
                rows.append(dict(model=m, scma=t, variant=v, macro_f1=f1, recall=rec, far=far))
            print("recenter", t, m, flush=True)
        pd.DataFrame(rows).to_csv(PH / "posthoc_recentered_macro_f1.csv", index=False)


if __name__ == "__main__":
    st = sys.argv[1:] or ["shift", "errors", "svm", "recenter"]
    for s in st:
        {"shift": stage_shift, "errors": stage_errors, "svm": stage_svm, "recenter": stage_recenter}[s]()
