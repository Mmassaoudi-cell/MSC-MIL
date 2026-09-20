"""Aggregate Part 2 -> ablation_aggregate.csv + ABLATION_v2.md (narrative from results/v2/robust/narrative/ABLATION.md)."""
import json, pickle
import numpy as np
import pandas as pd
from src.v2 import baselines_lib as L
from src.v2 import enhanced_model as em
from src.v2.final import run_eval as R
from src.v2.robust import lib

NARR = lib.ROB / "narrative" / "ABLATION.md"
CL = {"clean": "clean", "cal_4x": "calibrated 4x (0.904%)", "legacy_15": "legacy 15%"}


def load():
    rows = []
    for f in sorted(lib.RAW.glob("ablation_scma*_seed*.json")):
        r = json.loads(f.read_text())
        for c, d in r["results"].items():
            for name, mm in d.items():
                rows.append(dict(scma=r["scma"], seed=r["seed"], condition=c, row=name, **mm))
    return pd.DataFrame(rows)


def params():
    """parameter counts: network params + tree nodes (xgboost booster node count, as in the final evaluation)."""
    out = {}
    net = em.EnhancedMSCMIL({"power_mean": [0.0], "power_scale": [1.0], "static_mean": [0.0] * 4, "static_scale": [1.0] * 4}, physics=False, dropout=0.2)
    out["E4 network"] = em.n_params(net)
    out["MSC-MIL orig"] = em.n_params(L.build_deep("msc_mil", lib.FROZEN["msc_mil"]["best_params"]))
    def nodes(model, t=1, s=0):
        return json.loads((R.RUNS / f"{model}_scma{t}_seed{s}.json").read_text())["size"]
    x = json.loads((R.RUNS / "pi_fusion_scma1_seed0.json").read_text())
    out["PI-Fusion tree component (xgb nodes, SCMA-1 seed0)"] = x["xgb_nodes"]
    out["XGBoost-PF (nodes, SCMA-1 seed0)"] = nodes("xgboost_pf")
    out["XGBoost 7-feat frozen baseline (nodes, SCMA-1 seed0)"] = nodes("xgboost")
    ck = lib.RCK / "xgb7aug_scma1_seed0.pkl"
    if ck.exists():
        out["7-feat aug XGBoost (nodes, SCMA-1 seed0)"] = R.tree_nodes(pickle.load(open(ck, "rb")), "xgboost")
    return out


def main():
    df = load()
    if df.empty:
        print("nothing"); return
    rows = []
    for (t, c, r), g in df.groupby(["scma", "condition", "row"], sort=False):
        d = dict(scma=t, condition=c, row=r, n_seeds=len(g))
        for k in ("macro_f1", "recall", "far"):
            d[k + "_mean"] = g[k].mean(); d[k + "_std"] = g[k].std(ddof=1)
        rows.append(d)
    a = pd.DataFrame(rows); a.to_csv(lib.ROB / "ablation_aggregate.csv", index=False)
    P = params()
    (lib.ROB / "ablation_param_counts.json").write_text(json.dumps(P, indent=1))
    order = list(dict.fromkeys(df.row))
    main_rows = [r for r in order if not r.startswith("W.")]
    sweep = [r for r in order if r.startswith("W.")]
    md = ["# ABLATION_v2: ablation of PI-Fusion on test_full (post-hoc, frozen models)", "",
          "Code: `src/v2/robust/run_ablation.py`, `report_ablation.py`. Raw: `results/v2/robust/raw/ablation_scma{t}_seed{s}.json`; aggregate `ablation_aggregate.csv`, `ablation_param_counts.json`.", "",
          f"SCMA-1 and SCMA-4, seeds 0-4 (n seeds per cell: {sorted(set(a.n_seeds))}), threshold 0.5, macro-F1 / recall / FAR mean ± std over seeds. Same seed index = same trained components as the final evaluation (saved checkpoints). "
          "Conditions: clean, calibrated 4x sensor noise (0.904%), legacy 15% noise (identical perturbed copy for every row of one type/seed).",
          "", "Component definitions. **E4 network** and **PI-Fusion tree component** (15 physics-augmented features, clean + 4 noise/dropout-augmented copies) are the two parts of the saved PI-Fusion run. "
          "**XGBoost-PF** is the separately trained baseline (same 15 features and hyper-parameters, but trained on clean data only, no augmentation). **7-feat aug XGBoost** is a NEW training (the only retraining in this part): same frozen XGBoost "
          "hyper-parameters and the same augmentation stream as the PI-Fusion tree, but the 7 summary features only, isolating the physics features. **Orig. MSC-MIL** is the frozen clean-trained original network (no augmentation, own tuned config).", "",
          "**The fusion-weight sweep is a post-hoc sensitivity analysis run on test data; it was not used to change the frozen model (w_tree = 0.2 remains the proposed model).**", ""]
    if NARR.exists():
        md += ["## Findings", "", NARR.read_text(), ""]
    for t in lib.TYPES:
        g = a[a.scma == t]
        if g.empty:
            continue
        md += [f"## SCMA-{t}: component ablation", ""]
        for c in ("clean", "cal_4x", "legacy_15"):
            md += [f"### SCMA-{t}, {CL[c]}", "", "| row | macro-F1 | recall | FAR |", "|---|---|---|---|"]
            for r in main_rows:
                x = g[(g.condition == c) & (g.row == r)]
                if len(x):
                    x = x.iloc[0]
                    md.append(f"| {r} | {x.macro_f1_mean:.4f} ± {x.macro_f1_std:.4f} | {x.recall_mean:.4f} ± {x.recall_std:.4f} | {x.far_mean:.4f} ± {x.far_std:.4f} |")
            md.append("")
        md += [f"### SCMA-{t}: fusion-weight sweep (post-hoc sensitivity; w_tree = weight of the aug-trained PI-Fusion tree component; 0 = E4 network only, 1 = tree only)", "",
               "| w_tree | clean macro-F1 | clean recall | clean FAR | cal 4x macro-F1 | legacy 15% macro-F1 | legacy 15% recall | legacy 15% FAR |", "|---|---|---|---|---|---|---|---|"]
        for r in sweep:
            def cell(c, k):
                x = g[(g.condition == c) & (g.row == r)]
                return f"{x[k + '_mean'].iloc[0]:.4f} ± {x[k + '_std'].iloc[0]:.4f}" if len(x) else "n/a"
            md.append(f"| {r.split('=')[1]} | {cell('clean','macro_f1')} | {cell('clean','recall')} | {cell('clean','far')} | {cell('cal_4x','macro_f1')} | {cell('legacy_15','macro_f1')} | {cell('legacy_15','recall')} | {cell('legacy_15','far')} |")
        md.append("")
    md += ["## Parameter counts", "", "| component | count |", "|---|---|"] + [f"| {k} | {v:,} |" for k, v in P.items()]
    md += ["", "PI-Fusion total = 205,057 network parameters + XGBoost node count of the tree component (final evaluation: `size` field of `results/v2/final/runs/pi_fusion_*.json`, node counts vary by seed)."]
    (lib.ROB / "ABLATION_v2.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote ABLATION_v2.md")


if __name__ == "__main__":
    main()
