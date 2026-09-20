"""Part 2: ablation of PI-Fusion on test_full (SCMA-1, SCMA-4; seeds 0-4). Post-hoc; the weight sweep does NOT change the model.
Only new training: the 7-feature XGBoost trained with the identical recipe as the PI-Fusion tree (frozen params, clean + 4 augmented copies)."""
import json, pickle, time, numpy as np, torch
from xgboost import XGBClassifier
from src.v2 import baselines_lib as L
from src.v2 import enhanced_model as em
from src.v2.robust import lib

CONDS = ["clean", "cal_4x", "legacy_15"]
WEIGHTS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]


def fit_tree7_aug(train, meta, seed):
    """== em.fit_frozen_tree but with the 7 summary features only (same params, same augmentation stream)."""
    refs = em.noise_refs(train); fill = float(-meta["power_mean"][0] / meta["power_scale"][0])
    gen = torch.Generator().manual_seed(seed)
    S, M, ST = torch.from_numpy(train.slots), torch.from_numpy(train.mask), torch.from_numpy(train.static)
    Xs = [L.feats(train)]
    for _ in range(em.TREE_AUG_COPIES):
        s, m, st = em.augment_batch(S, M, ST, fill, refs[0], refs[1], 0.3, 0.25, gen)
        Xs.append(L.feats(L.D(s.numpy(), m.numpy(), st.numpy(), train.labels)))
    X = np.vstack(Xs); y = np.tile(train.labels, em.TREE_AUG_COPIES + 1); pos = y.sum()
    est = XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=8, tree_method="hist",
                        scale_pos_weight=float((len(y) - pos) / max(pos, 1)), **em.FROZEN_TREE_PARAMS)
    est.fit(X, y)
    return est


def rows_for(P):
    """P: dict of component probabilities -> dict row_name -> probs."""
    pn, pt, pm = P["net"], P["tree_aug"], P["msc"]
    R = {
        "A. Full PI-Fusion (aug tree 15-feat + E4 net, w=0.2)": 0.2 * pt + 0.8 * pn,
        "B. Tree only: XGBoost-PF (clean-trained baseline)": P["xgb_pf_clean"],
        "B'. Tree only: PI-Fusion tree component (noise-aug trained)": pt,
        "C. Network only: E4 (noise-aug trained)": pn,
        "D. Fusion w/o physics: 7-feat aug XGBoost (same recipe) + E4, w=0.2": 0.2 * P["xgb7_aug"] + 0.8 * pn,
        "D'. Fusion w/o physics: frozen 7-feat XGBoost baseline (clean-trained) + E4, w=0.2": 0.2 * P["xgb7_clean"] + 0.8 * pn,
        "E. Fusion w/o noise-aug net: orig. MSC-MIL + PI-Fusion tree component, w=0.2": 0.2 * pt + 0.8 * pm,
        "E'. Fusion w/o noise-aug net: orig. MSC-MIL + XGBoost-PF (clean), w=0.2": 0.2 * P["xgb_pf_clean"] + 0.8 * pm,
        "G1. 7-feat aug XGBoost alone": P["xgb7_aug"],
        "G2. Original MSC-MIL alone": pm,
    }
    for w in WEIGHTS:
        R[f"W. sweep w_tree={w}"] = w * pt + (1 - w) * pn
    return R


def main():
    device = lib.dev()
    for t in lib.TYPES:
        data, test = lib.load_all(t); y = test.labels
        for s in lib.SEEDS5:
            out = lib.RAW / f"ablation_scma{t}_seed{s}.json"
            if out.exists(): continue
            need = [lib.deep_path("msc_mil", t, s), lib.tree_path("xgboost_pf", t, s), lib.tree_path("xgboost", t, s)]
            if any(not p.exists() for p in need):
                print(f"SKIP SCMA-{t} seed{s}: missing {[str(p) for p in need if not p.exists()]}", flush=True); continue
            t0 = time.time()
            ck = lib.RCK / f"xgb7aug_scma{t}_seed{s}.pkl"
            if ck.exists():
                x7 = pickle.load(open(ck, "rb"))
            else:
                x7 = fit_tree7_aug(data["train"], data["meta"], s); pickle.dump(x7, open(ck, "wb"))
            net, xgb = lib.load_pi(t, s, data["meta"], device)
            msc = lib.load_deep("msc_mil", t, s, device)
            xpf, x7c = lib.load_tree("xgboost_pf", t, s), lib.load_tree("xgboost", t, s)
            res = {}
            for cond in CONDS:
                d = lib.make_condition(cond, test, data["train"], data["meta"], t, s)
                F15 = em.tree_features(d.slots, d.mask, d.static, data["meta"]); F7 = L.feats(d)
                P = dict(net=em._net_proba(net, d, device), tree_aug=xgb.predict_proba(F15)[:, 1], msc=L.infer(msc, d, device),
                         xgb_pf_clean=xpf.predict_proba(F15)[:, 1], xgb7_aug=x7.predict_proba(F7)[:, 1], xgb7_clean=x7c.predict_proba(F7)[:, 1])
                res[cond] = {k: lib.metrics(y, v) for k, v in rows_for(P).items()}
            out.write_text(json.dumps(dict(scma=t, seed=s, results=res), default=float))
            print(f"ablation SCMA-{t} seed{s} done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
