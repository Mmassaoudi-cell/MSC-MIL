"""Retrain the frozen noise-augmented baselines (no saved checkpoints existed) for SCMA-1/4, seeds 0-4, from their frozen configs
(results/v2/frozen_baseline_configs.json entries *_aug). Train/val only. Resumable."""
import json, pickle, time, torch
from src.v2 import baselines_lib as L
from src.v2.robust import lib

TAGS = ["random_forest_aug", "xgboost_aug", "catboost_aug", "msc_mil_aug", "source_bilstm_attn_aug"]

def main():
    device = L.get_device()
    for t in lib.TYPES:
        data = L.load_trainval(t)
        for tag in TAGS:
            ent = lib.FROZEN[tag]; base, p = ent["base_model"], ent["best_params"]
            for s in lib.SEEDS5:
                out = lib.RCK / f"{tag}_scma{t}_seed{s}.{'pkl' if base in L.TREE_MODELS else 'pt'}"
                if out.exists():
                    continue
                t0 = time.time()
                if base in L.TREE_MODELS:
                    X, y = L.tree_train_set(data, True, s)
                    est = L.fit_tree(base, p, X, y, s)
                    with open(out, "wb") as f: pickle.dump(est, f)
                    info = {}
                else:
                    net = None
                    for attempt in (0, 1):
                        net, info = L.train_deep(base, p, data, s, device, aug=True)
                        if net is not None: break
                    if net is None:
                        print("FAILED", tag, t, s, info, flush=True); continue
                    torch.save(net.state_dict(), out)
                (lib.RCK / f"{tag}_scma{t}_seed{s}.info.json").write_text(json.dumps(dict(tag=tag, scma=t, seed=s, config_hash=lib.sha_obj(ent), **{"train_time_s": time.time()-t0, **info}), default=float))
                print(f"[{tag}] SCMA-{t} seed{s} done {time.time()-t0:.0f}s {info.get('best_val_macro_f1','')}", flush=True)

if __name__ == "__main__":
    main()
