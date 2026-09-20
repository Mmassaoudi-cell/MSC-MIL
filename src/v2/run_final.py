"""3-seed runs of each tuned config on v2 SCMA-1/2/4 (train->val), clean + VAL-only noise protocols.
usage: python -m src.v2.run_final --models tag1,tag2   (tag = model or model_aug)
Appends to results/v2/baselines_tuning/final_runs.csv (skips finished model/type/seed).
"""
import argparse, json, time
import numpy as np, pandas as pd
from src.v2 import baselines_lib as L

TYPES, SEEDS = (1, 2, 4), (0, 1, 2)
CSV = L.OUT / "final_runs.csv"
KEYS = ["macro_f1", "recall", "false_alarm_rate", "precision", "accuracy", "roc_auc"]


def done_set():
    if not CSV.exists(): return set()
    df = pd.read_csv(CSV); return set(zip(df.model, df.scma, df.seed))


def main(tags):
    dev = L.get_device(); sr = L.sigma_r()
    protos = L.noise_protocols()
    for tag in tags:
        best = json.loads((L.OUT / f"best_{tag}.json").read_text())
        name, aug, p = best["base_model"], best["aug"], best["best_params"]
        for t in TYPES:
            data = L.load_trainval(t); va = data["val"]
            for s in SEEDS:
                if (tag, t, s) in done_set(): continue
                t0 = time.time()
                if name in L.TREE_MODELS:
                    X, y = L.tree_train_set(data, aug, s)
                    est = L.fit_tree(name, p, X, y, s)
                    score = lambda d: est.predict_proba(L.feats(d))[:, 1]
                    info = {}
                else:
                    model, info = L.train_deep(name, p, data, s, dev, aug=aug)
                    if model is None:
                        print("FAILED", tag, t, s, info); continue
                    score = lambda d: L.infer(model, d, dev)
                rows = []
                m = L.metrics_from_score(va.labels, score(va)); rows.append(("clean", m))
                for pname, kind, par in protos:
                    dn = L.make_noisy(va, data, kind, par, s, sr)
                    rows.append((pname, L.metrics_from_score(va.labels, score(dn))))
                out = [dict(model=tag, scma=t, seed=s, condition=c, train_time_s=time.time() - t0, **{k: mm.get(k, np.nan) for k in KEYS}) for c, mm in rows]
                pd.DataFrame(out).to_csv(CSV, mode="a", header=not CSV.exists(), index=False)
                print(f"[{tag}] SCMA-{t} seed {s} clean F1={rows[0][1]['macro_f1']:.4f} cal4x={rows[-1][1]['macro_f1']:.4f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--models"); a = ap.parse_args()
    main(a.models.split(","))
