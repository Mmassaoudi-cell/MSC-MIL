"""Batch-1 latency (median of 200 reps after 10 warm-ups), seed-0 checkpoints, measured after training finished.
Trees: CPU only (features from the raw tensors + predict_proba); GPU = n/a. Deep: forward pass on GPU (synchronised) and CPU.
pi_fusion: network (GPU or CPU) + tree features + XGBoost predict + fusion arithmetic (tree part is always CPU)."""
import json, pickle, time

import numpy as np
import torch

from src.v2 import baselines_lib as L
from src.v2 import enhanced_model as em
from src.v2.final import run_eval as R

REPS, WARM = 200, 10


def med(fn, sync=False):
    for _ in range(WARM):
        fn()
    ts = []
    for _ in range(REPS):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter(); fn()
        if sync:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3)


def build_net(model, t, meta):
    if model == "e4_net":
        c = em.FROZEN_NET_CFG
        return em.EnhancedMSCMIL(meta, physics=False, dropout=c["dropout"])
    return L.build_deep(model, R.FROZEN[model]["best_params"])


def main():
    out = {}
    cuda = torch.cuda.is_available()
    for t in R.__dict__.get("TYPES", (1, 2, 4)) if False else (1, 2, 4):
        data = L.load_trainval(t); meta = data["meta"]
        test = R.load_test(t)["test_full"]
        sl, mk, st = test.slots[:1], test.mask[:1], test.static[:1]
        d1 = L.D(sl, mk, st, test.labels[:1])
        for model in R.TREES:
            est = pickle.load(open(R.CKPT / f"{model}_scma{t}_seed0.pkl", "rb"))
            out[f"{model}_scma{t}"] = dict(cpu_ms=med(lambda: est.predict_proba(R.tree_feats(model, d1, meta))[:, 1]), gpu_ms=None)
        nets = ["mlp", "cnn1d", "plain_lstm_mil", "transformer_mil", "source_bilstm_attn", "msc_mil", "e4_net"]
        for model in nets:
            net = build_net(model, t, meta)
            ck = R.CKPT / (f"pi_fusion_scma{t}_seed0_net.pt" if model == "e4_net" else f"{model}_scma{t}_seed0.pt")
            net.load_state_dict(torch.load(ck, map_location="cpu")); net.eval()
            res = {}
            for dev in (["cuda"] if cuda else []) + ["cpu"]:
                n = net.to(dev)
                s_, m_, t_ = (torch.from_numpy(x).to(dev) for x in (sl, mk, st))
                with torch.no_grad():
                    res[f"{'gpu' if dev == 'cuda' else 'cpu'}_ms"] = med(lambda: n(s_, m_, t_), sync=(dev == "cuda"))
            out[f"{model}_scma{t}"] = res
        # fused: network + tree features + xgboost
        net = build_net("e4_net", t, meta)
        net.load_state_dict(torch.load(R.CKPT / f"pi_fusion_scma{t}_seed0_net.pt", map_location="cpu")); net.eval()
        xgb = pickle.load(open(R.CKPT / f"pi_fusion_scma{t}_seed0_xgb.pkl", "rb"))
        res = {}
        for dev in (["cuda"] if cuda else []) + ["cpu"]:
            n = net.to(dev)
            s_, m_, t_ = (torch.from_numpy(x).to(dev) for x in (sl, mk, st))

            def fused():
                with torch.no_grad():
                    pn = n(s_, m_, t_)[0].float().cpu().numpy()
                pt = xgb.predict_proba(em.tree_features(sl, mk, st, meta))[:, 1]
                return em.FROZEN_W_TREE * pt + (1 - em.FROZEN_W_TREE) * pn
            res[f"{'gpu' if dev == 'cuda' else 'cpu'}_ms"] = med(fused, sync=(dev == "cuda"))
        out[f"pi_fusion_scma{t}"] = res
        print(t, "done", flush=True)
    (R.FIN / "latency.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
