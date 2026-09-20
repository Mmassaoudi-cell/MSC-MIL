"""Driver for the enhancement study (validation only). Resumable: every unit of work writes a JSON and is skipped if present.

python -m src.v2.enh_run all     # full pipeline sequentially
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna
import torch
from catboost import CatBoostClassifier

from src.reproduction.metrics import macro_f1_score
from src.v2.common import RESULTS_V2, summary_features
from src.v2.enh_data import load_v2_trainval, get_split
from src.v2 import enhanced_model as em

OUT = RESULTS_V2 / "enh"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "probs").mkdir(exist_ok=True)
(OUT / "ckpt").mkdir(exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = (0, 1, 2)
TYPES = (1, 4)
N_TRIALS = 15
_DATA = {}


def data(t):
    if t not in _DATA:
        d, meta = load_v2_trainval(t)
        get_split(d, "train"); get_split(d, "val")
        _DATA[t] = (d, meta)
    return _DATA[t]


def jdump(p, o):
    Path(p).write_text(json.dumps(o, indent=1, default=float))


# ------------------------------------------------------------------ configs
def base_cfg(name):
    c = dict(em.REFERENCE_CFG)
    c["name"] = name
    if name != "REF":
        c.update(augment=True, noise_max=0.3, drop_max=0.25, es_metric="score")
    if name in ("E2",):
        c["physics"] = True
    if name == "E4":
        c.update(max_epochs=60, patience=12, cosine=True)
    return c


def apply_params(cfg, p):
    c = dict(cfg)
    c.update(lr=p["lr"], dropout=p["dropout"], supcon_beta=p["beta"], supcon_temperature=p["temperature"])
    return c


# ------------------------------------------------------------------ evaluation of one NN run
def latency(model, ncalls=200):
    d, _ = data(1)
    s = torch.from_numpy(d["val"].slots[:1]); m = torch.from_numpy(d["val"].mask[:1]); st = torch.from_numpy(d["val"].static[:1])
    res = {}
    for dev in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
        mm = model.to(dev).eval(); a, b, c = s.to(dev), m.to(dev), st.to(dev)
        with torch.no_grad():
            for _ in range(20): mm(a, b, c)
            ts = []
            for _ in range(ncalls):
                if dev == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter(); mm(a, b, c)
                if dev == "cuda": torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
        res[f"latency_ms_{dev}"] = float(np.median(ts) * 1e3)
    model.to(DEV)
    return res


def run_nn(cand, cfg, t, seed, tag):
    """Train cfg on type t / seed, evaluate S components with the reporting noise draws, save probs + json."""
    fp = OUT / f"{tag}_scma{t}_seed{seed}.json"
    if fp.exists():
        return json.loads(fp.read_text())
    d, meta = data(t)
    r = em.train_enhanced(cfg, d["train"], d["val"], meta, seed, DEV)
    model = r["model"]
    refs = em.noise_refs(d["train"])
    noisy = em.noisy_copies(d["val"], refs, seed_offset=seed)
    ev = em.eval_clean_noisy(model, d["val"], noisy, DEV)
    y = d["val"].labels
    probs = {"clean": em.predict_proba(model, d["val"], DEV)}
    for lv, ns in zip(em.NOISE_LEVELS, noisy):
        probs[f"n{lv}"] = em.predict_proba(model, ns, DEV)
    np.savez(OUT / "probs" / f"{tag}_scma{t}_seed{seed}.npz", y=y, **probs)
    torch.save(model.state_dict(), OUT / "ckpt" / f"{tag}_scma{t}_seed{seed}.pt")
    out = dict(cand=cand, tag=tag, scma=t, seed=seed, clean_f1=ev["clean_f1"], noisy_f1=ev["noisy_f1"], noisy_mean=ev["noisy_mean"],
               S_T=0.5 * ev["clean_f1"] + 0.5 * ev["noisy_mean"], recall_clean=ev["recall_clean"], best_epoch=r["best_epoch"],
               epochs=len(r["history"]), n_params=r["n_params"], train_time_s=r["train_time_s"], converged=r["converged"], cfg=cfg)
    if t == 1 and seed == 0:
        out.update(latency(model))
    jdump(fp, out)
    print(f"[{tag}] SCMA-{t} seed{seed}: clean={ev['clean_f1']:.4f} noisy={ev['noisy_f1']} S_T={out['S_T']:.4f} rec={ev['recall_clean']:.3f} ep={out['best_epoch']}/{out['epochs']} ({r['train_time_s']:.0f}s)", flush=True)
    return out


# ------------------------------------------------------------------ Optuna for NN candidates
def optuna_nn(cand):
    fp = OUT / f"optuna_{cand}_best.json"
    if fp.exists():
        return json.loads(fp.read_text())
    cfg0 = base_cfg(cand)

    def obj(trial):
        p = dict(beta=trial.suggest_float("beta", 0.05, 2.0, log=True), temperature=trial.suggest_float("temperature", 0.05, 0.5),
                 dropout=trial.suggest_float("dropout", 0.1, 0.4), lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True))
        o = run_nn(cand, apply_params(cfg0, p), 1, 0, f"trial_{cand}_{trial.number}")
        trial.set_user_attr("clean", o["clean_f1"]); trial.set_user_attr("noisy", o["noisy_mean"])
        return o["S_T"] if o["converged"] else 0.0

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0),
                             storage=f"sqlite:///{(OUT / 'optuna.db').as_posix()}", study_name=f"{cand}", load_if_exists=True)
    todo = N_TRIALS - len([t for t in st.trials if t.state.name == "COMPLETE"])
    if todo > 0:
        st.optimize(obj, n_trials=todo)
    best = dict(best_params=st.best_params, best_value=st.best_value, trials=[dict(n=t.number, value=t.value, params=t.params, clean=t.user_attrs.get("clean"), noisy=t.user_attrs.get("noisy")) for t in st.trials])
    jdump(fp, best)
    return best


def seeds_nn(cand):
    b = optuna_nn(cand)
    cfg = apply_params(base_cfg(cand), b["best_params"])
    for t in TYPES:
        for s in SEEDS:
            run_nn(cand, cfg, t, s, cand)


def seeds_ref():
    for t in TYPES:
        for s in SEEDS:
            run_nn("REF", base_cfg("REF"), t, s, "REF")


# ------------------------------------------------------------------ E3 fusion
def cb_features(slots, mask, static, meta, ext):
    ns = SimpleNamespace(slots=slots, mask=mask, static=static, labels=np.zeros(len(mask)))
    X = summary_features(ns)
    if ext:
        c = em._consts(meta, "cpu")
        with torch.no_grad():
            ph = em.physics_features(torch.from_numpy(slots[..., 0]), torch.from_numpy(mask), torch.from_numpy(static), c).numpy()
        X = np.hstack([X, ph])
    return X


def cb_train_set(train, meta, ext, seed, k=4):
    refs = em.noise_refs(train)
    fill = float(-meta["power_mean"][0] / meta["power_scale"][0])
    gen = torch.Generator().manual_seed(seed)
    S, M, ST = torch.from_numpy(train.slots), torch.from_numpy(train.mask), torch.from_numpy(train.static)
    Xs = [cb_features(train.slots, train.mask, train.static, meta, ext)]
    for _ in range(k):
        s, m, st = em.augment_batch(S, M, ST, fill, refs[0], refs[1], 0.3, 0.25, gen)
        Xs.append(cb_features(s.numpy(), m.numpy(), st.numpy(), meta, ext))
    return np.vstack(Xs), np.tile(train.labels, k + 1)


def cb_fit_predict(params, t, seed):
    d, meta = data(t)
    ext = params["feat"] == "ext"
    X, y = cb_train_set(d["train"], meta, ext, seed)
    cb = CatBoostClassifier(depth=params["depth"], learning_rate=params["lr"], iterations=params["iters"], l2_leaf_reg=params["l2"],
                            auto_class_weights="Balanced", random_seed=seed, verbose=0, thread_count=8)
    cb.fit(X, y)
    refs = em.noise_refs(d["train"])
    noisy = em.noisy_copies(d["val"], refs, seed_offset=seed)
    out = {"clean": cb.predict_proba(cb_features(d["val"].slots, d["val"].mask, d["val"].static, meta, ext))[:, 1]}
    for lv, ns in zip(em.NOISE_LEVELS, noisy):
        out[f"n{lv}"] = cb.predict_proba(cb_features(ns.slots, ns.mask, ns.static, meta, ext))[:, 1]
    return cb, out


def fuse_eval(p_cb, p_nn, w, y):
    f = {k: macro_f1_score(y, ((w * p_cb[k] + (1 - w) * p_nn[k]) >= 0.5).astype(int)) for k in p_cb}
    noisy = [f[f"n{lv}"] for lv in em.NOISE_LEVELS]
    pred = ((w * p_cb["clean"] + (1 - w) * p_nn["clean"]) >= 0.5).astype(int)
    rec = float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
    return dict(clean_f1=f["clean"], noisy_f1=noisy, noisy_mean=float(np.mean(noisy)), S_T=0.5 * f["clean"] + 0.5 * float(np.mean(noisy)), recall_clean=rec)


def e1_probs(t, s):
    z = np.load(OUT / "probs" / f"E1_scma{t}_seed{s}.npz")
    return {k: z[k] for k in z.files if k != "y"}


def optuna_e3():
    fp = OUT / "optuna_E3_best.json"
    if fp.exists():
        return json.loads(fp.read_text())
    y = data(1)[0]["val"].labels
    p_e1 = e1_probs(1, 0)

    def obj(trial):
        p = dict(depth=trial.suggest_int("depth", 3, 8), lr=trial.suggest_float("lr", 0.03, 0.3, log=True), iters=trial.suggest_int("iters", 100, 500),
                 l2=trial.suggest_float("l2", 1.0, 10.0), feat=trial.suggest_categorical("feat", ["base7", "ext"]))
        w = trial.suggest_float("w", 0.0, 1.0)
        _, pc = cb_fit_predict(p, 1, 0)
        r = fuse_eval(pc, p_e1, w, y)
        trial.set_user_attr("clean", r["clean_f1"]); trial.set_user_attr("noisy", r["noisy_mean"])
        print(f"[E3 trial {trial.number}] {p} w={w:.3f} clean={r['clean_f1']:.4f} noisy={r['noisy_mean']:.4f} S={r['S_T']:.4f}", flush=True)
        return r["S_T"]

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0),
                             storage=f"sqlite:///{(OUT / 'optuna.db').as_posix()}", study_name="E3", load_if_exists=True)
    todo = N_TRIALS - len([t for t in st.trials if t.state.name == "COMPLETE"])
    if todo > 0:
        st.optimize(obj, n_trials=todo)
    best = dict(best_params=st.best_params, best_value=st.best_value, trials=[dict(n=t.number, value=t.value, params=t.params, clean=t.user_attrs.get("clean"), noisy=t.user_attrs.get("noisy")) for t in st.trials])
    jdump(fp, best)
    return best


def seeds_e3():
    b = optuna_e3()
    bp = dict(b["best_params"]); w = bp.pop("w")
    cbp = dict(depth=bp["depth"], lr=bp["lr"], iters=bp["iters"], l2=bp["l2"], feat=bp["feat"])
    for t in TYPES:
        y = data(t)[0]["val"].labels
        for s in SEEDS:
            fp = OUT / f"E3_scma{t}_seed{s}.json"
            if fp.exists():
                continue
            t0 = time.time()
            cb, pc = cb_fit_predict(cbp, t, s)
            r = fuse_eval(pc, e1_probs(t, s), w, y)
            out = dict(cand="E3", scma=t, seed=s, w=w, cb=cbp, n_params=json.loads((OUT / f"E1_scma{t}_seed{s}.json").read_text())["n_params"] + cbp["iters"] * 2 ** cbp["depth"],
                       train_time_s=time.time() - t0, converged=True, **r)
            # latency (seed0, type1): E1 network + catboost row + features
            if t == 1 and s == 0:
                e1 = json.loads((OUT / "E1_scma1_seed0.json").read_text())
                d, meta = data(1)
                v = d["val"]; ext = cbp["feat"] == "ext"
                for _ in range(10): cb.predict_proba(cb_features(v.slots[:1], v.mask[:1], v.static[:1], meta, ext))
                ts = []
                for _ in range(200):
                    t1 = time.perf_counter(); cb.predict_proba(cb_features(v.slots[:1], v.mask[:1], v.static[:1], meta, ext)); ts.append(time.perf_counter() - t1)
                out["latency_ms_catboost_part"] = float(np.median(ts) * 1e3)
                out["latency_ms_cuda"] = e1.get("latency_ms_cuda", float("nan")) + out["latency_ms_catboost_part"]
                out["latency_ms_cpu"] = e1["latency_ms_cpu"] + out["latency_ms_catboost_part"]
            jdump(fp, out)
            print(f"[E3] SCMA-{t} seed{s}: clean={r['clean_f1']:.4f} noisy={r['noisy_f1']} S_T={r['S_T']:.4f} rec={r['recall_clean']:.3f}", flush=True)


def main():
    what = sys.argv[1:] or ["all"]
    for w in what:
        if w in ("ref", "all"): seeds_ref()
        if w in ("E1", "all"): seeds_nn("E1")
        if w in ("E3", "all"): seeds_e3()
        if w in ("E2", "all"): seeds_nn("E2")
        if w in ("E4", "all"): seeds_nn("E4")


if __name__ == "__main__":
    main()
