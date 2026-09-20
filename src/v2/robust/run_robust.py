"""Part 1: robustness of the frozen models on test_full (SCMA-1, SCMA-4; seeds 0-4), 11 conditions. Resumable per (type, seed)."""
import json, sys, time, numpy as np
from src.v2 import baselines_lib as L
from src.v2.robust import lib

PROB_MODELS = ["pi_fusion", "e4_net", "xgboost_pf", "msc_mil"]

def main():
    device = lib.dev()
    for t in lib.TYPES:
        data, test = lib.load_all(t)
        y = test.labels
        np.save(lib.PROBS / f"labels_scma{t}.npy", y)
        for s in lib.SEEDS5:
            out = lib.RAW / f"robust_scma{t}_seed{s}.json"
            if out.exists():
                continue
            miss = lib.bundle_available(t, s)
            if miss:
                print(f"SKIP SCMA-{t} seed{s}: missing {len(miss)} ckpts (e.g. {miss[0]})", flush=True); continue
            t0 = time.time()
            b = lib.load_bundle(t, s, data["meta"], device)
            res = {}
            for cond in lib.CONDS:
                d = lib.make_condition(cond, test, data["train"], data["meta"], t, s)
                P = lib.predict_all(b, d, data["meta"], device)
                res[cond] = {m: lib.metrics(y, P[m]) for m in lib.MODELS}
                for m in PROB_MODELS:
                    np.save(lib.PROBS / f"{m}_scma{t}_seed{s}_{cond}.npy", P[m].astype(np.float32))
            out.write_text(json.dumps(dict(scma=t, seed=s, sigma_r=lib.SIGMA_R, results=res), default=float))
            print(f"SCMA-{t} seed{s} done {time.time()-t0:.0f}s; clean PI={res['clean']['pi_fusion']['macro_f1']:.4f} cal4x={res['cal_4x']['pi_fusion']['macro_f1']:.4f} leg15={res['legacy_15']['pi_fusion']['macro_f1']:.4f}", flush=True)

if __name__ == "__main__":
    main()
