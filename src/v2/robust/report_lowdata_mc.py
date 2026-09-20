"""Reports for Part 3 (LOWDATA_v2.md) and Part 4 (MULTICLASS_v2.md). Narrative read from results/v2/robust/narrative/{LOWDATA,MULTICLASS}.md."""
import json
import numpy as np
import pandas as pd
from src.v2.robust import lib

NAMES = ["benign"] + ["SCMA-%d" % i for i in range(1, 7)]


def lowdata():
    rows = []
    for f in sorted(lib.RAW.glob("lowdata_scma1_frac*_seed*.json")):
        r = json.loads(f.read_text())
        for m in ("pi_fusion", "e4_net", "pi_tree_component", "xgboost_pf", "msc_mil"):
            if r.get(m):
                rows.append(dict(frac=r["frac"], seed=r["seed"], model=m, n_train=r["n_train"], **{k: r[m][k] for k in ("macro_f1", "recall", "far")}))
    # 100 % = frozen final runs, same seeds 0-2
    for s in (0, 1, 2):
        for m, fm in (("pi_fusion", "pi_fusion"), ("e4_net", "e4_net"), ("xgboost_pf", "xgboost_pf"), ("msc_mil", "msc_mil")):
            j = json.loads((lib.FIN / "runs" / f"{fm}_scma1_seed{s}.json").read_text())["test_full"]
            rows.append(dict(frac=1.0, seed=s, model=m, n_train=7996, macro_f1=j["macro_f1"], recall=j["recall"], far=j["far"]))
    df = pd.DataFrame(rows)
    if df.empty:
        return
    g = df.groupby(["model", "frac"]).agg(n=("seed", "count"), n_train=("n_train", "mean"), f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", lambda x: x.std(ddof=1)),
                                           rec_mean=("recall", "mean"), rec_std=("recall", lambda x: x.std(ddof=1)), far_mean=("far", "mean"), far_std=("far", lambda x: x.std(ddof=1))).reset_index()
    g.to_csv(lib.ROB / "lowdata_aggregate.csv", index=False)
    df.to_csv(lib.ROB / "lowdata_runs.csv", index=False)
    disp = {"pi_fusion": "PI-Fusion (E5)", "xgboost_pf": "XGBoost-PF (clean-trained)", "msc_mil": "MSC-MIL (orig.)", "e4_net": "E4 network alone (component)", "pi_tree_component": "PI-Fusion tree component (aug)"}
    fr = sorted(g.frac.unique())
    md = ["# LOWDATA_v2: low-data curve on SCMA-1 (test_full, clean)", "",
          "Code: `src/v2/robust/run_lowdata.py`, `report_lowdata_mc.py`. Raw: `results/v2/robust/raw/lowdata_scma1_frac{10,25,50}_seed{s}.json`, `lowdata_runs.csv`, `lowdata_aggregate.csv`.", "",
          "Protocol: stratified (by label) subsample of the TRAIN pool (7996 bags: 4000 benign + 3996 attack) at 10/25/50%, with the VALIDATION split subsampled the same way (used only for each model's own frozen early stopping); "
          "seeds 0-2; every model uses its own frozen config (PI-Fusion: E4 recipe + frozen tree, w_tree 0.2; XGBoost-PF: frozen tuned params on the 15 features, clean-trained; MSC-MIL: original tuned config, clean-trained). "
          "Scalers are the frozen train-benign scalers of the full pool. Evaluation: the full `test_full` (7945 + 883). Mean ± std over 3 seeds. The 100% column is NOT retrained: it is the frozen final-evaluation run of the same seeds 0-2 (identical procedure, `results/v2/final/runs`).", ""]
    NARR = lib.ROB / "narrative" / "LOWDATA.md"
    if NARR.exists():
        md += ["## Findings", "", NARR.read_text(), ""]
    for key, nm in (("f1", "Macro-F1"), ("rec", "Recall"), ("far", "FAR")):
        md += [f"## {nm} (mean ± std over 3 seeds)", "", "| model | " + " | ".join(f"{int(f*100)}% (n={int(g[g.frac == f].n_train.iloc[0]) if (g.frac == f).any() else ''})" for f in fr) + " |", "|---|" + "---|" * len(fr)]
        for m in disp:
            c = []
            for f in fr:
                x = g[(g.model == m) & (g.frac == f)]
                c.append(f"{x[key + '_mean'].iloc[0]:.4f} ± {x[key + '_std'].iloc[0]:.4f}" if len(x) else "n/a")
            if any(v != "n/a" for v in c):
                md.append(f"| {disp[m]} | " + " | ".join(c) + " |")
        md.append("")
    (lib.ROB / "LOWDATA_v2.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote LOWDATA_v2.md")


def multiclass():
    OUT = lib.ROB / "multiclass"
    fs = sorted(OUT.glob("multiclass_seed*.json"))
    if not fs:
        return
    R = [json.loads(f.read_text()) for f in fs]
    audit = json.loads((lib.ROB / "multiclass_data_audit.json").read_text())
    disp = {"net": "E4 network, 7-way softmax", "tree": "XGBoost-PF multiclass", "fusion": "PI-Fusion multiclass (0.2 tree + 0.8 net)"}
    rows = []
    md = ["# MULTICLASS_v2: 7-way attribution (benign + SCMA-1..6) on the v2 benchmark, test_full", "",
          "Code: `src/v2/robust/run_multiclass.py`, `report_lowdata_mc.py`. Raw: `results/v2/robust/multiclass/multiclass_seed{s}.json` (+ probabilities), `multiclass_data_audit.json`, CSVs `multiclass_*.csv`.", "",
          f"Seeds: {[r['seed'] for r in R]} (n={len(R)}). Training data: the six cached v2 types; benign bags (identical in all six caches) + the attacked bags of each type as class k. Only train/val used for training/early stopping; evaluation on `test_full` "
          "(benign 7945 + 883 attacked bags per class). (a) E4 network with a 7-way softmax head (`MSCMILMulticlass`, frozen E4 hyper-parameters/augmentation/cosine schedule/patience, CE + L2 + SupCon on the 7 labels; validation score = 0.5 * clean macro-F1 + 0.5 * mean macro-F1 under legacy 5/15/30% noise). "
          "(b) XGBoost multi:softprob on the 15 summary+physics features, frozen params, clean + 4 augmented copies. (c) fusion 0.2 * p_tree + 0.8 * p_net, argmax.", "",
          "**Data audit (documented, not engineered around).** The same underlying charging sessions appear under several classes: every attack class re-uses the same source sessions "
          f"(train {audit['train']['n_distinct_attack_source_sessions']} sources, val {audit['val']['n_distinct_attack_source_sessions']}, test_full {audit['test_full']['n_distinct_attack_source_sessions']}; identical across types: {audit['test_full']['sources_identical_across_types']}), "
          f"and in val/test the attacked source sessions are drawn from the same sessions that appear as benign bags (test_full: {audit['test_full']['attack_sources_also_present_as_benign_bag_in_same_split']} of {audit['test_full']['n_distinct_attack_source_sessions']} sources also appear as a benign bag; "
          f"train: {audit['train']['attack_sources_also_present_as_benign_bag_in_same_split']} of {audit['train']['n_distinct_attack_source_sessions']}). Source sessions overlap between train and test: {audit['cross_split_source_overlap_train_vs_test']}. "
          "Class balance on test is 7945 benign vs 883 per attack class, so accuracy is dominated by the benign class; macro-F1 is the headline metric. Class counts per split: " + json.dumps({sp: audit[sp]['counts'] for sp in ('train', 'val', 'test_full')}), ""]
    NARR = lib.ROB / "narrative" / "MULTICLASS.md"
    if NARR.exists():
        md += ["## Findings", "", NARR.read_text(), ""]
    summ = []
    for m in ("net", "tree", "fusion"):
        mf = np.array([r[m]["macro_f1"] for r in R]); ac = np.array([r[m]["accuracy"] for r in R])
        bd = np.array([r[m]["binary_detect_recall"] for r in R]); bf = np.array([r[m]["binary_far"] for r in R])
        sd = lambda x: x.std(ddof=1) if len(x) > 1 else float("nan")
        summ.append(dict(model=m, macro_f1_mean=mf.mean(), macro_f1_std=sd(mf), accuracy_mean=ac.mean(), accuracy_std=sd(ac), benign_vs_any_recall=bd.mean(), benign_vs_any_far=bf.mean()))
    pd.DataFrame(summ).to_csv(lib.ROB / "multiclass_summary.csv", index=False)
    md += ["## Summary (mean ± std over seeds)", "", "| model | macro-F1 | accuracy | detection recall (any attack vs benign) | detection FAR |", "|---|---|---|---|---|"]
    for s_ in summ:
        md.append(f"| {disp[s_['model']]} | {s_['macro_f1_mean']:.4f} ± {s_['macro_f1_std']:.4f} | {s_['accuracy_mean']:.4f} ± {s_['accuracy_std']:.4f} | {s_['benign_vs_any_recall']:.4f} | {s_['benign_vs_any_far']:.4f} |")
    md.append("")
    pcs = []
    for m in ("net", "tree", "fusion"):
        md += [f"## {disp[m]}", "", "### Per-class precision / recall / F1 (mean ± std over seeds)", "", "| class | precision | recall | F1 | support |", "|---|---|---|---|---|"]
        for k in range(7):
            v = {q: np.array([r[m][q][k] for r in R]) for q in ("precision", "recall", "f1")}
            sd = lambda x: x.std(ddof=1) if len(x) > 1 else float("nan")
            md.append(f"| {NAMES[k]} | {v['precision'].mean():.3f} ± {sd(v['precision']):.3f} | {v['recall'].mean():.3f} ± {sd(v['recall']):.3f} | {v['f1'].mean():.3f} ± {sd(v['f1']):.3f} | {R[0][m]['support'][k]} |")
            pcs.append(dict(model=m, cls=NAMES[k], **{q + "_mean": v[q].mean() for q in v}, **{q + "_std": sd(v[q]) for q in v}))
        cm = np.mean([np.array(r[m]["confusion"], float) for r in R], axis=0)
        cmn = cm / cm.sum(1, keepdims=True)
        pd.DataFrame(cm, index=NAMES, columns=NAMES).to_csv(lib.ROB / f"multiclass_confusion_{m}_mean_counts.csv")
        pd.DataFrame(cmn, index=NAMES, columns=NAMES).to_csv(lib.ROB / f"multiclass_confusion_{m}_rownorm.csv")
        md += ["", "### 7x7 confusion matrix, mean count over seeds (rows = true class, columns = predicted)", "", "| true \\ pred | " + " | ".join(NAMES) + " |", "|---|" + "---|" * 7]
        for i in range(7):
            md.append(f"| {NAMES[i]} | " + " | ".join(f"{cm[i, j]:.1f}" for j in range(7)) + " |")
        md += ["", "Row-normalised (fraction of the true class):", "", "| true \\ pred | " + " | ".join(NAMES) + " |", "|---|" + "---|" * 7]
        for i in range(7):
            md.append(f"| {NAMES[i]} | " + " | ".join(f"{cmn[i, j]:.3f}" for j in range(7)) + " |")
        md.append("")
    pd.DataFrame(pcs).to_csv(lib.ROB / "multiclass_per_class.csv", index=False)
    (lib.ROB / "MULTICLASS_v2.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote MULTICLASS_v2.md")


if __name__ == "__main__":
    lowdata()
    multiclass()
