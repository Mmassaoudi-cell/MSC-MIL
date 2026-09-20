"""Aggregate Part 1 (robustness) -> CSVs + bootstrap + ROBUSTNESS_v2.md (tables from data; narrative from results/v2/robust/narrative/ROBUST.md)."""
import json
import numpy as np
import pandas as pd
from src.v2.robust import lib

COND_ORDER = list(lib.CONDS)
LABEL = {"clean": "clean", "cal_1x": "cal 1x (0.226%)", "cal_2x": "cal 2x (0.452%)", "cal_4x": "cal 4x (0.904%)", "legacy_5": "legacy 5%", "legacy_15": "legacy 15%",
         "legacy_30": "legacy 30%", "miss_real": "miss 0.15% (real)", "miss_10": "miss 10%", "miss_25": "miss 25%", "miss_50": "miss 50%"}
NARR = lib.ROB / "narrative" / "ROBUST.md"
B = 1000


def load():
    rows = []
    for f in sorted(lib.RAW.glob("robust_scma*_seed*.json")):
        r = json.loads(f.read_text())
        for cond, dm in r["results"].items():
            for m, mm in dm.items():
                rows.append(dict(scma=r["scma"], seed=r["seed"], condition=cond, model=m, **mm))
    return pd.DataFrame(rows)


def aggregate(df):
    clean = df[df.condition == "clean"].set_index(["scma", "seed", "model"])
    out = []
    for (t, c, m), g in df.groupby(["scma", "condition", "model"]):
        row = dict(scma=t, condition=c, model=m, display=lib.DISPLAY[m], n_seeds=len(g))
        for k in ("macro_f1", "recall", "far"):
            row[k + "_mean"] = g[k].mean(); row[k + "_std"] = g[k].std(ddof=1) if len(g) > 1 else np.nan
            dr = [g_.__getitem__(k) - clean.loc[(t, g_["seed"], m), k] for _, g_ in g.iterrows()]
            row[k + "_drop_mean"] = float(np.mean(dr)) if k != "far" else float(np.mean(dr))  # value - clean (negative macro-F1/recall = degradation; positive FAR = degradation)
        out.append(row)
    a = pd.DataFrame(out)
    a["_c"] = a.condition.map({c: i for i, c in enumerate(COND_ORDER)}); a["_m"] = a.model.map({m: i for i, m in enumerate(lib.MODELS)})
    a = a.sort_values(["scma", "_m", "_c"]).drop(columns=["_c", "_m"])
    a.to_csv(lib.ROB / "robustness_aggregate.csv", index=False)
    return a


def bootstrap():
    rows = []
    for t in lib.TYPES:
        yf = lib.PROBS / f"labels_scma{t}.npy"
        if not yf.exists():
            continue
        y = np.load(yf); pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        rng = np.random.default_rng(20260921 + t)
        ps = rng.choice(pos, size=(B, len(pos))); ns = rng.choice(neg, size=(B, len(neg)))

        def mf1(pred):
            tp = pred[ps].sum(1); fn = ps.shape[1] - tp; fp = pred[ns].sum(1); tn = ns.shape[1] - fp
            return (2 * tp / np.maximum(2 * tp + fp + fn, 1) + 2 * tn / np.maximum(2 * tn + fn + fp, 1)) / 2, tp / ps.shape[1], fp / ns.shape[1]

        for cond in ("clean", "cal_4x", "legacy_15"):
            avg = {}
            for m in ("pi_fusion", "xgboost_pf", "msc_mil"):
                fs = sorted(lib.PROBS.glob(f"{m}_scma{t}_seed*_{cond}.npy"))
                if len(fs) < 5:
                    avg = None; break
                avg[m] = np.mean([np.load(f) for f in fs], axis=0)
            if avg is None:
                continue
            pr = {m: (avg[m] >= 0.5).astype(np.int64) for m in avg}
            bs = {m: mf1(pr[m]) for m in pr}
            for other in ("xgboost_pf", "msc_mil"):
                d = bs["pi_fusion"][0] - bs[other][0]
                pt = lambda m: (lib.metrics(y, avg[m])["macro_f1"])
                lo, hi = np.percentile(d, [2.5, 97.5])
                dr = bs["pi_fusion"][1] - bs[other][1]; df_ = bs["pi_fusion"][2] - bs[other][2]
                rows.append(dict(scma=t, condition=cond, comparison=f"PI-Fusion - {lib.DISPLAY[other]}", n_seeds_averaged=5, point_diff_macro_f1=pt("pi_fusion") - pt(other),
                                 ci_lo=lo, ci_hi=hi, verdict=("PI-Fusion better" if lo > 0 else ("PI-Fusion worse" if hi < 0 else "no significant difference")),
                                 recall_diff_ci=f"[{np.percentile(dr, 2.5):+.4f}, {np.percentile(dr, 97.5):+.4f}]", far_diff_ci=f"[{np.percentile(df_, 2.5):+.4f}, {np.percentile(df_, 97.5):+.4f}]"))
    b = pd.DataFrame(rows); b.to_csv(lib.ROB / "robustness_bootstrap.csv", index=False)
    return b


def fmt(m, s, pct=False):
    return "n/a" if pd.isna(m) else (f"{m:.4f} ± {s:.4f}" if not pd.isna(s) else f"{m:.4f}")


def table(a, t, key, conds):
    g = a[a.scma == t]
    lines = ["| model | " + " | ".join(LABEL[c] for c in conds) + " |", "|---|" + "---|" * len(conds)]
    for m in lib.MODELS:
        r = []
        for c in conds:
            x = g[(g.model == m) & (g.condition == c)]
            r.append(fmt(x[key + "_mean"].iloc[0], x[key + "_std"].iloc[0]) if len(x) else "n/a")
        lines.append(f"| {lib.DISPLAY[m]} | " + " | ".join(r) + " |")
    return "\n".join(lines)


def drop_table(a, t, conds):
    g = a[a.scma == t]
    lines = ["| model | " + " | ".join(LABEL[c] for c in conds) + " |", "|---|" + "---|" * len(conds)]
    for m in lib.MODELS:
        r = []
        for c in conds:
            x = g[(g.model == m) & (g.condition == c)]
            r.append(f"{x['macro_f1_drop_mean'].iloc[0]:+.4f}" if len(x) else "n/a")
        lines.append(f"| {lib.DISPLAY[m]} | " + " | ".join(r) + " |")
    return "\n".join(lines)


def main():
    df = load()
    if df.empty:
        print("no raw results"); return
    a = aggregate(df); b = bootstrap()
    ns = sorted(df.groupby("scma").seed.nunique().to_dict().items())
    md = ["# ROBUSTNESS_v2: post-freeze robustness of the frozen models on the v2 artifact-free benchmark (test_full)", "",
          "Code: `src/v2/robust/{lib,train_aug,run_robust,report_robust}.py`. Raw per-run JSON: `results/v2/robust/raw/robust_scma{t}_seed{s}.json`; aggregates `robustness_aggregate.csv`; bootstrap `robustness_bootstrap.csv`.",
          "", "## Protocol", "",
          f"* Frozen models, no tuning, evaluated with the saved final-evaluation checkpoints (`results/v2/final/ckpt`), seeds 0-4, SCMA-1 and SCMA-4, `test_full` (7945 benign + 883 attack). Seeds available: {ns}. Threshold 0.5. This is a post-freeze analysis; nothing here changes any model.",
          f"* Calibrated sensor noise: multiplicative Gaussian x' = x(1 + r*eps) on active-slot power and on the static energy feature, r = 1x/2x/4x sigma_r = 0.226%/0.452%/0.904% (`noise_calibration.json`).",
          "* Legacy stress noise: `src.benchmarks.robustness.perturb_noise` (additive, std = 5/15/30% of the TRAIN std of scaled active-slot power and static energy).",
          "* Missing telemetry: stress levels 10/25/50% of the active slots of each session masked with the legacy function `perturb_missing` (fixed fraction, same fill value = mean inactive-slot value of TRAIN). Realistic level 0.15% of slots (NOISE_CALIBRATION.md): the legacy rule rounds `0.0015 * n_active` to zero for every session shorter than 334 slots (all of them), so it would mask nothing; the realistic level is therefore implemented as independent Bernoulli(0.0015) masking of each active slot (same fill convention).",
          "* Trees (XGBoost-PF, LightGBM-PF, CatBoost-PF, PI-Fusion's tree) recompute the 7 summary features and the 8 physics features from the perturbed bags; 7-feature trees recompute the 7 summary features. All models of one (condition, type, seed) see exactly the same perturbed copy (paired).",
          "* Noise-augmented baselines (`*_aug`, frozen configs from `frozen_baseline_configs.json`): no checkpoints had been saved in the baseline stage, so they were RETRAINED ONCE (SCMA-1 and SCMA-4, seeds 0-4) from their frozen configs with `src/v2/robust/train_aug.py` (train/val only; checkpoints in `results/v2/robust/ckpt/`).",
          f"* Bootstrap: 1000 stratified (by label) paired resamples of test sessions; the statistic is macro-F1 at 0.5 of the seed-averaged probability of the model (5 seeds). The interval is over test sessions, not over training seeds. Drop = value under perturbation minus value clean, averaged over seeds (negative = macro-F1/recall degradation, positive = FAR degradation).",
          ""]
    if NARR.exists():
        md += ["## Findings", "", NARR.read_text(), ""]
    for t in lib.TYPES:
        if t not in a.scma.values:
            continue
        md += [f"## SCMA-{t}", ""]
        for key, nm in (("macro_f1", "Macro-F1"), ("recall", "Recall"), ("far", "False-alarm rate (FAR)")):
            md += [f"### SCMA-{t}: {nm}, calibrated + legacy sensor noise (mean ± std over seeds)", "", table(a, t, key, ["clean", "cal_1x", "cal_2x", "cal_4x", "legacy_5", "legacy_15", "legacy_30"]), "",
                   f"### SCMA-{t}: {nm}, missing telemetry", "", table(a, t, key, ["clean", "miss_real", "miss_10", "miss_25", "miss_50"]), ""]
        md += [f"### SCMA-{t}: macro-F1 drop relative to clean (mean over seeds)", "", drop_table(a, t, COND_ORDER[1:]), ""]
    if len(b):
        md += ["## Paired bootstrap over test sessions (PI-Fusion minus comparator, macro-F1, seed-averaged probabilities)", "",
               "| SCMA | condition | comparison | point diff | 95% CI | verdict | recall diff 95% CI | FAR diff 95% CI |", "|---|---|---|---|---|---|---|---|"]
        for _, r in b.iterrows():
            md.append(f"| {r.scma} | {LABEL[r.condition]} | {r.comparison} | {r.point_diff_macro_f1:+.4f} | [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] | {r.verdict} | {r.recall_diff_ci} | {r.far_diff_ci} |")
        md.append("")
    (lib.ROB / "ROBUSTNESS_v2.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote ROBUSTNESS_v2.md", len(md))


if __name__ == "__main__":
    main()
