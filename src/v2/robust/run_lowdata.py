"""Part 3: low-data curve, SCMA-1, test_full clean. PI-Fusion / XGBoost-PF / original MSC-MIL retrained on 10/25/50 % of the train pool
(stratified; val subsampled the same way, used only for the model's own frozen early stopping), seeds 0-2. 100 % = the frozen final-evaluation
runs of the same seeds (identical procedure; test preds in results/v2/final/preds), not retrained."""
import json, time, numpy as np, torch
from src.v2 import baselines_lib as L
from src.v2 import enhanced_model as em
from src.v2.robust import lib

FRACS = (0.10, 0.25, 0.50)
SEEDS = (0, 1, 2)
T = 1


def strat_sub(d, frac, seed, split_id):
    rng = np.random.default_rng([424242, seed, int(round(frac * 100)), split_id])
    idx = np.sort(np.concatenate([rng.choice(np.where(d.labels == c)[0], size=max(1, int(round(frac * (d.labels == c).sum()))), replace=False) for c in (0, 1)]))
    return L.D(d.slots[idx], d.mask[idx], d.static[idx], d.labels[idx])


def main():
    device = lib.dev()
    data, test = lib.load_all(T); y = test.labels; meta = data["meta"]
    xp = lib.FROZEN["xgboost_pf"]; mp = lib.FROZEN["msc_mil"]
    for frac in FRACS:
        for s in SEEDS:
            out = lib.RAW / f"lowdata_scma{T}_frac{int(frac*100)}_seed{s}.json"
            if out.exists(): continue
            t0 = time.time()
            tr, va = strat_sub(data["train"], frac, s, 0), strat_sub(data["val"], frac, s, 1)
            sub = dict(train=tr, val=va, meta=meta)
            res = dict(frac=frac, seed=s, n_train=int(len(tr.labels)), n_val=int(len(va.labels)), pos_rate_train=float(tr.labels.mean()))
            # PI-Fusion
            r = em.train_frozen(tr, va, meta, s, device)
            fd = r["model"]
            pn = em._net_proba(fd.net, test, device)
            pt = fd.tree.predict_proba(em.tree_features(test.slots, test.mask, test.static, meta))[:, 1]
            res["pi_fusion"] = lib.metrics(y, fd.w_tree * pt + (1 - fd.w_tree) * pn)
            res["e4_net"] = lib.metrics(y, pn); res["pi_tree_component"] = lib.metrics(y, pt)
            res["pi_best_epoch"] = r["net_result"]["best_epoch"]
            # XGBoost-PF (clean-trained frozen baseline)
            est = L.fit_tree("xgboost", xp["best_params"], em.tree_features(tr.slots, tr.mask, tr.static, meta), tr.labels, s)
            res["xgboost_pf"] = lib.metrics(y, est.predict_proba(em.tree_features(test.slots, test.mask, test.static, meta))[:, 1])
            # original MSC-MIL
            net = None
            for _ in range(2):
                net, info = L.train_deep("msc_mil", mp["best_params"], sub, s, device, aug=False)
                if net is not None: break
            if net is None:
                res["msc_mil"] = None; res["msc_info"] = info
            else:
                res["msc_mil"] = lib.metrics(y, L.infer(net, test, device)); res["msc_best_epoch"] = info["best_epoch"]
            out.write_text(json.dumps(res, default=float))
            print(f"lowdata frac={frac} seed={s} {time.time()-t0:.0f}s PI={res['pi_fusion']['macro_f1']:.4f} XGB-PF={res['xgboost_pf']['macro_f1']:.4f} "
                  f"MSC={res['msc_mil']['macro_f1'] if res['msc_mil'] else None}", flush=True)


if __name__ == "__main__":
    main()
