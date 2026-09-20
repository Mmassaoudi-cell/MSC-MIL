"""SINGLE final test evaluation on the v2 benchmark (frozen configs; no tuning).
usage: python -m src.v2.final.run_eval --group trees|deep [--types 1,2,4] [--models a,b]
Resumable: a (model,type,seed) whose runs/<...>.json exists is skipped.
The test arrays are first read by `load_test` (which stamps TEST_FIRST_LOAD.json atomically)."""
from __future__ import annotations

import argparse, datetime, hashlib, json, pickle, time, traceback

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score

from src.v2 import baselines_lib as L
from src.v2 import enhanced_model as em
from src.v2.common import DATA_V2, PROJECT_ROOT

FIN = PROJECT_ROOT / "results" / "v2" / "final"
RUNS, PREDS, CKPT = FIN / "runs", FIN / "preds", FIN / "ckpt"
CFG_PATH = PROJECT_ROOT / "results" / "v2" / "frozen_baseline_configs.json"
FROZEN = json.loads(CFG_PATH.read_text())["models"]

TREES = ["logreg", "knn", "svm_rbf", "random_forest", "xgboost", "lightgbm", "catboost",
         "random_forest_pf", "xgboost_pf", "lightgbm_pf", "catboost_pf"]
DEEP_LOW = ["mlp", "cnn1d", "plain_lstm_mil", "transformer_mil"]
DEEP_HI = ["pi_fusion", "msc_mil", "source_bilstm_attn"]   # pi_fusion run also yields e4_net (same trained network)


def n_seeds(model, t):
    if model in DEEP_LOW:
        return 5 if t == 1 else 3
    return 10 if t == 1 else 5


def entry_hash(model):
    if model in FROZEN:
        return hashlib.sha256(json.dumps(FROZEN[model], sort_keys=True).encode()).hexdigest()
    return hashlib.sha256(json.dumps(em.FROZEN_CFG, sort_keys=True, default=str).encode()).hexdigest()


def load_test(t):
    """The ONLY place the test arrays are read. Stamps the first-load time on first use."""
    stamp = FIN / "TEST_FIRST_LOAD.json"
    if not stamp.exists():
        try:
            with open(stamp, "x") as f:
                json.dump(dict(first_test_load_local=datetime.datetime.now().astimezone().isoformat(), first_type=t,
                               by="src/v2/final/run_eval.py::load_test",
                               note="test_full/test_n200 arrays of the v2 cache first deserialised at this time"), f, indent=1)
        except FileExistsError:
            pass
    z = np.load(DATA_V2 / f"scma{t}_v2.npz", allow_pickle=False)
    return {sp: L.D(z[f"{sp}__slots"], z[f"{sp}__mask"], z[f"{sp}__static"], z[f"{sp}__labels"]) for sp in ("test_full", "test_n200")}


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    rec = tp / max(tp + fn, 1); prec = tp / max(tp + fp, 1); far = fp / max(fp + tn, 1)
    return dict(macro_f1=float(f1_score(y, pred, average="macro")), recall=rec, precision=prec, far=far,
                pr_auc=float(average_precision_score(y, p)), balanced_accuracy=(rec + (1 - far)) / 2, tp=tp, fp=fp, tn=tn, fn=fn)


def tree_feats(model, d, meta):
    if model.endswith("_pf"):
        return em.tree_features(d.slots, d.mask, d.static, meta)
    return L.feats(d)


def tree_nodes(est, base):
    try:
        if base == "logreg":
            return int(est.coef_.size + 1)
        if base == "knn":
            return int(est._fit_X.shape[0] * est._fit_X.shape[1])
        if base == "svm_rbf":
            return int(est.n_support_.sum())
        if base == "random_forest":
            return int(sum(e.tree_.node_count for e in est.estimators_))
        if base == "xgboost":
            return int(len(est.get_booster().trees_to_dataframe()))
        if base == "lightgbm":
            return int(len(est.booster_.trees_to_dataframe()))
        if base == "catboost":
            return int(est.tree_count_ * 2 ** est.get_param("depth"))
    except Exception:
        return -1
    return -1


SIZE_KIND = {"logreg": "coefficients", "knn": "stored training values", "svm_rbf": "support vectors", "random_forest": "tree nodes",
             "xgboost": "tree nodes", "lightgbm": "tree nodes", "catboost": "leaves (trees*2^depth)"}


def save_run(model, t, s, test, probs, extra):
    y = test["test_full"].labels
    rec = dict(model=model, scma=t, seed=s, config_hash=entry_hash(model),
               test_full=metrics(y, probs["test_full"]), test_n200=metrics(test["test_n200"].labels, probs["test_n200"]), **extra)
    np.save(PREDS / f"{model}_scma{t}_seed{s}.npy", probs["test_full"].astype(np.float32))
    np.save(PREDS / f"{model}_scma{t}_seed{s}_n200.npy", probs["test_n200"].astype(np.float32))
    (RUNS / f"{model}_scma{t}_seed{s}.json").write_text(json.dumps(rec, indent=1, default=float))
    print(f"[{model}] SCMA-{t} seed{s} test_full macroF1={rec['test_full']['macro_f1']:.4f} rec={rec['test_full']['recall']:.3f} "
          f"far={rec['test_full']['far']:.4f} ({extra.get('train_time_s', 0):.0f}s)", flush=True)


def done(model, t, s):
    return (RUNS / f"{model}_scma{t}_seed{s}.json").exists()


def run_tree(model, t, s, data, test):
    ent = FROZEN[model]; base, p = ent["base_model"], ent["best_params"]
    t0 = time.time()
    if model.endswith("_pf"):
        Xtr = tree_feats(model, data["train"], data["meta"]); ytr = data["train"].labels
    else:
        Xtr, ytr = L.tree_train_set(data, False, s)
    est = L.fit_tree(base, p, Xtr, ytr, s)
    tt = time.time() - t0
    probs = {sp: est.predict_proba(tree_feats(model, d, data["meta"]))[:, 1] for sp, d in test.items()}
    with open(CKPT / f"{model}_scma{t}_seed{s}.pkl", "wb") as f:
        pickle.dump(est, f)
    save_run(model, t, s, test, probs, dict(train_time_s=tt, size=tree_nodes(est, base), size_kind=SIZE_KIND[base]))


def run_deep(model, t, s, data, test, dev):
    p = FROZEN[model]["best_params"]
    for attempt in (0, 1):   # SAME config re-run once if the run fails (non-finite loss); no config change
        net, info = L.train_deep(model, p, data, s, dev, aug=False)
        if net is not None:
            break
    if net is None:
        (RUNS / f"{model}_scma{t}_seed{s}.FAILED.json").write_text(json.dumps(info, default=float))
        print("FAILED", model, t, s, info, flush=True)
        return
    probs = {sp: L.infer(net, d, dev) for sp, d in test.items()}
    torch.save(net.state_dict(), CKPT / f"{model}_scma{t}_seed{s}.pt")
    save_run(model, t, s, test, probs, dict(train_time_s=info["train_time_s"], size=int(em.n_params(net)), size_kind="parameters",
                                             best_epoch=info["best_epoch"], epochs=info["epochs"]))


def run_fusion(t, s, data, test, dev):
    """One training run of E5 (network E4 + XGBoost-PF); E4-alone = its network component, PI-Fusion = fused output."""
    t0 = time.time()
    r = em.train_frozen(data["train"], data["val"], data["meta"], s, dev)
    tt = time.time() - t0
    fd = r["model"]; nr = r["net_result"]
    pn, pt, pf = {}, {}, {}
    for sp, d in test.items():
        pn[sp] = em._net_proba(fd.net, d, dev)
        pt[sp] = fd.tree.predict_proba(em.tree_features(d.slots, d.mask, d.static, data["meta"]))[:, 1]
        pf[sp] = fd.w_tree * pt[sp] + (1.0 - fd.w_tree) * pn[sp]
    torch.save(fd.net.state_dict(), CKPT / f"pi_fusion_scma{t}_seed{s}_net.pt")
    with open(CKPT / f"pi_fusion_scma{t}_seed{s}_xgb.pkl", "wb") as f:
        pickle.dump(fd.tree, f)
    xn = tree_nodes(fd.tree, "xgboost")
    common = dict(best_epoch=nr["best_epoch"], converged=bool(r["converged"]))
    np.save(PREDS / f"pi_fusion_tree_scma{t}_seed{s}_full.npy", pt["test_full"].astype(np.float32))
    save_run("e4_net", t, s, test, pn, dict(train_time_s=nr["train_time_s"], size=int(nr["n_params"]), size_kind="parameters",
                                             note="network component of the pi_fusion run of the same seed (identical training)", **common))
    save_run("pi_fusion", t, s, test, pf, dict(train_time_s=tt, size=int(nr["n_params"]) + xn, size_kind="net parameters + xgboost nodes",
                                                net_params=int(nr["n_params"]), xgb_nodes=xn, tree_train_time_s=tt - nr["train_time_s"], **common))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--group", required=True); ap.add_argument("--types", default="1,2,4"); ap.add_argument("--models", default="")
    a = ap.parse_args()
    dev = L.get_device()
    types = [int(x) for x in a.types.split(",")]
    models = a.models.split(",") if a.models else (TREES if a.group == "trees" else DEEP_HI + DEEP_LOW)
    for t in types:
        data = L.load_trainval(t)
        test = load_test(t)
        np.save(PREDS / f"labels_scma{t}_test_full.npy", test["test_full"].labels)
        np.save(PREDS / f"labels_scma{t}_test_n200.npy", test["test_n200"].labels)
        for model in models:
            for s in range(n_seeds(model, t)):
                if done(model, t, s) and (model != "pi_fusion" or done("e4_net", t, s)):
                    continue
                try:
                    if model == "pi_fusion":
                        run_fusion(t, s, data, test, dev)
                    elif model in TREES:
                        run_tree(model, t, s, data, test)
                    else:
                        run_deep(model, t, s, data, test, dev)
                except Exception:
                    traceback.print_exc(); print("CRASH", model, t, s, "-> rerun the same command to retry", flush=True)


if __name__ == "__main__":
    main()
