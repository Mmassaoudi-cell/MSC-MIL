"""FINAL one-shot TEST evaluation (pre-registered in results/external/EXTERNAL_PROTOCOL.md).

python -m src.external.run_final --dataset cic2017 --confirm-frozen

Requires (validation-only stage already done): tuned_configs_<ds>.json, tuned_configs_aug_<ds>.json, fusion_weights_<ds>.json.
Resumable: every (model, seed) test-probability block (7 conditions x n_test) is cached in results/external/final_cache/<ds>/.
Models: 9 frozen validation-tuned baselines + tree_aug + msc_aug_tuned + msc_e4 (+ PROPOSED = fusion of msc_e4 and tree_aug).
Conditions: clean; Gaussian noise 5/15/30% of train feature std (continuous features); removal of 10/25/50% of flows.
Outputs: final_probs_<ds>.npz, final_records_<ds>.csv (per seed), final_agg_<ds>.csv (mean/std over seeds),
         final_bootstrap_<ds>.csv (stratified bootstrap, paired PROPOSED - baseline).
"""
import argparse, json, time
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score
from src.external.data import load, perturb_noise, perturb_missing
from src.external.prep_common import RES
from src.external import zoo

NOISE = (0.05, 0.15, 0.30); MISS = (0.10, 0.25, 0.50); SEEDS = (0, 1, 2, 3, 4); B = 2000
CONDS = [("clean", 0.0)] + [("noise", s) for s in NOISE] + [("missing", s) for s in MISS]


def make_conds(te, fstd, cont, seed):
    rng = np.random.default_rng(1000 + seed)
    out = [(te["X"], None)]
    for s in NOISE: out.append((perturb_noise(te["X"], s, fstd, cont, rng), None))
    for f in MISS: out.append(perturb_missing(te["X"], f, rng))
    return out


def counts_metrics(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    f1p = 2 * tp / max(2 * tp + fp + fn, 1); f1n = 2 * tn / max(2 * tn + fn + fp, 1)
    return {"macro_f1": (f1p + f1n) / 2, "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
            "far": fp / max(fp + tn, 1), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def boot_macro(pred_seeds, ypos_idx, yneg_idx, ip, ineg):
    """pred_seeds (S,n) binary; ip (B,npos) indices into pos set; returns (B,) seed-averaged macro-F1."""
    acc = 0
    for p in pred_seeds:
        pp, pn = p[ypos_idx], p[yneg_idx]
        tp = pp[ip].sum(1).astype(np.float64); fn = len(ypos_idx) - tp
        fp = pn[ineg].sum(1).astype(np.float64); tn = len(yneg_idx) - fp
        acc = acc + (2 * tp / np.maximum(2 * tp + fp + fn, 1) + 2 * tn / np.maximum(2 * tn + fn + fp, 1)) / 2
    return acc / len(pred_seeds)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--confirm-frozen", action="store_true")
    a = ap.parse_args(); ds = a.dataset
    if not a.confirm_frozen:
        raise SystemExit("Refusing to read the TEST split: pass --confirm-frozen.")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = load(ds, ("train", "val", "test")); tr, va, te = D["train"], D["val"], D["test"]
    ctx = zoo.Ctx(ds, tr, va, D["meta"], dev)
    w = json.loads((RES / f"fusion_weights_{ds}.json").read_text())["w_tree"]
    cdir = RES / "final_cache" / ds; cdir.mkdir(parents=True, exist_ok=True)
    y = te["y"]; probs = {}
    for seed in SEEDS:
        conds = None
        for name in zoo.ALL:
            f = cdir / f"{name}_s{seed}.npy"
            if not f.exists():
                if conds is None: conds = make_conds(te, ctx.fstd, ctx.cont, seed)
                t0 = time.time(); fit = zoo.get(ctx, name, seed)
                np.save(f, np.stack([fit.predict_proba(X, m) for X, m in conds]).astype(np.float32))
                print(f"[{ds}] seed {seed} {name} done ({time.time()-t0:.0f}s)", flush=True)
            probs.setdefault(name, []).append(np.load(f))
    probs = {k: np.stack(v) for k, v in probs.items()}                       # (S, 7, n)
    probs["PROPOSED"] = w * probs["tree_aug"] + (1 - w) * probs["msc_e4"]
    np.savez_compressed(RES / f"final_probs_{ds}.npz", y=y, w_tree=w, **probs)
    # ---- per-seed records
    rows = []
    for name, P in probs.items():
        for si, seed in enumerate(SEEDS):
            for ci, (kind, sev) in enumerate(CONDS):
                p = P[si, ci]; m = counts_metrics(y, (p >= 0.5).astype(int))
                m["roc_auc"] = float(roc_auc_score(y, p)); m["pr_auc"] = float(average_precision_score(y, p))
                rows.append({"dataset": ds, "model": name, "seed": seed, "kind": kind, "severity": sev, **m})
    df = pd.DataFrame(rows); df.to_csv(RES / f"final_records_{ds}.csv", index=False)
    g = df.groupby(["model", "kind", "severity"])[["macro_f1", "recall", "precision", "far", "roc_auc", "pr_auc"]].agg(["mean", "std"])
    g.columns = ["_".join(c) for c in g.columns]; g = g.reset_index(); g.to_csv(RES / f"final_agg_{ds}.csv", index=False)
    # ---- stratified bootstrap over test windows (same resamples for every model), paired vs PROPOSED
    rng = np.random.default_rng(20260919)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    ip = rng.integers(0, len(pos), (B, len(pos)), dtype=np.int32); ineg = rng.integers(0, len(neg), (B, len(neg)), dtype=np.int32)
    bm = {}
    for name, P in probs.items():
        for ci in range(len(CONDS)):
            bm[(name, ci)] = boot_macro((P[:, ci] >= 0.5).astype(np.int8), pos, neg, ip, ineg)
    brow = []
    for name in probs:
        for ci, (kind, sev) in enumerate(CONDS):
            v = bm[(name, ci)]; rec = {"dataset": ds, "model": name, "kind": kind, "severity": sev,
                                       "macro_f1_boot_mean": float(v.mean()), "ci_lo": float(np.percentile(v, 2.5)), "ci_hi": float(np.percentile(v, 97.5))}
            if name != "PROPOSED":
                d = bm[("PROPOSED", ci)] - v
                rec.update({"diff_proposed_minus_this": float(d.mean()), "diff_ci_lo": float(np.percentile(d, 2.5)),
                            "diff_ci_hi": float(np.percentile(d, 97.5)), "p_boot_diff_le_0": float((d <= 0).mean())})
            brow.append(rec)
    pd.DataFrame(brow).to_csv(RES / f"final_bootstrap_{ds}.csv", index=False)
    print(g.pivot_table(index="model", columns=["kind", "severity"], values="macro_f1_mean").round(4).to_string())


if __name__ == "__main__":
    main()
