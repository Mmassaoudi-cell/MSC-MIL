"""POST-HOC DIAGNOSTIC report builder (tables from posthoc/*.csv + hand-written text in posthoc/narrative_*.md)."""
import pandas as pd
from src.v2.final import run_eval as R
PH = R.FIN / "posthoc"
DN = {"pi_fusion": "PI-Fusion", "e4_net": "E4 network", "msc_mil": "MSC-MIL", "source_bilstm_attn": "Source BiLSTM+attn", "xgboost_pf": "XGBoost-PF",
      "catboost_pf": "CatBoost-PF", "random_forest": "RandomForest", "xgboost": "XGBoost", "svm_rbf": "RBF-SVM"}
def md(df):
    df = df.reset_index()
    l = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        l.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(l)
def nar(n):
    f = PH / f"narrative_{n}.md"; return f.read_text(encoding="utf-8") if f.exists() else ""
o = ["# POST-HOC DIAGNOSTICS of the v2 final test results (post-hoc diagnostic, analysis only)\n",
     "**Every table and statement in this file is a post-hoc diagnostic.** It was produced after the final test numbers were seen, changes no model, config, seed or selection, and must not be used to tune anything. Code: `src/v2/final/posthoc.py`, `posthoc_report.py`; CSVs in `results/v2/final/posthoc/` (all named `posthoc_*`). Benign sessions are identical across SCMA types, so feature-shift analysis uses the SCMA-1 cache. Site = first path element of the session uid (attack bags inherit their source session's site). Model probabilities are seed-averaged, threshold 0.5. Validation predictions were recomputed from the saved checkpoints.\n", nar("head")]
sh = pd.read_csv(PH / "posthoc_site_share.csv")
o.append("## 1. Temporal-shift quantification (post-hoc diagnostic)\n\n### Benign share per site and split\n")
o.append(md(sh.pivot(index="split", columns="site", values="share")) + "\n\n" + md(sh.pivot(index="split", columns="site", values="n_benign")) + "\n")
s = pd.read_csv(PH / "posthoc_shift_benign.csv")
for site in ["all", "caltech", "jpl", "office_01"]:
    for cmpn in ("train->test_full", "train->val"):
        g = s[(s.site == site) & (s.comparison == cmpn)].sort_values("psi", ascending=False)
        if not len(g): continue
        n = 8 if cmpn == "train->test_full" else 4
        o.append(f"\n### Benign, site = {site}, {cmpn}: top {n} features by PSI (SMD = standardised mean difference, KS statistic, PSI with train-decile bins)\n")
        o.append(md(g.head(n)[["feature", "smd", "ks", "psi"]].set_index("feature")) + "\n")
o.append("\nFull table (all features, sites, comparisons): `posthoc_shift_benign.csv`. Rule of thumb for PSI: <0.1 negligible, 0.1-0.25 moderate, >0.25 large. office_01 has only 107 train and 207 test benign sessions, jpl 210 and 419, so their estimates are noisy.\n")
o.append(nar("shift"))
e = pd.read_csv(PH / "posthoc_error_decomposition.csv"); e["model"] = e.model.map(DN)
o.append("\n## 2. Error decomposition, validation vs test (post-hoc diagnostic)\n")
for t in (1, 2, 4):
    g = e[e.scma == t].pivot(index="model", columns="split", values=["recall", "far", "macro_f1"])
    g.columns = [f"{a} ({b})" for a, b in g.columns]
    g = g[[f"recall (val)", "recall (test_full)", "far (val)", "far (test_full)", "macro_f1 (val)", "macro_f1 (test_full)"]].reindex(list(DN.values()))
    o.append(f"\n### SCMA-{t}\n" + md(g) + "\n")
fs = pd.read_csv(PH / "posthoc_far_by_site_test.csv"); fs["model"] = fs.model.map(DN)
o.append("\n### False-alarm rate on test benign sessions, by site (n_benign: caltech 7319, jpl 419, office_01 207)\n")
for t in (1, 2, 4):
    o.append(f"\nSCMA-{t}\n\n" + md(fs[fs.scma == t].pivot(index="model", columns="site", values="far").reindex(list(DN.values()))) + "\n")
o.append(nar("errors"))
o.append("\n## 3. Why RBF-SVM holds up on SCMA-2/4 (post-hoc diagnostic)\n\nThe frozen RBF-SVM (C=1.54, gamma=3.69, TRAIN subsample capped at 4000) uses the **7 summary features only**; there is no physics-feature SVM in the frozen set, so a 7-feature-vs-frozen-set comparison does not exist for it (the frozen set *is* the 7 features). Its FAR/recall against the other models is in the tables of Section 2.\n")
pi = pd.read_csv(PH / "posthoc_permutation_importance_val.csv")
pi["model"] = pi.model.map({"svm_rbf": "RBF-SVM", "xgboost": "XGBoost", "random_forest": "RandomForest"})
o.append("\n### Permutation importance on VALIDATION (drop in macro-F1 when one feature is shuffled; 10 shuffles, seeds 0-2 averaged)\n")
for t in (1, 2, 4):
    o.append(f"\nSCMA-{t}\n\n" + md(pi[pi.scma == t].pivot(index="feature", columns="model", values="perm_drop")) + "\n")
d = pd.read_csv(PH / "posthoc_far_by_distance_from_train_benign.csv")
cnt = d[(d.model == "svm_rbf")&(d.scma==1)][["zbin", "n_benign"]].set_index("zbin").reindex(["<2", "2-3", "3-5", "5-10", ">10"])
o.append("\n### Test benign FAR by distance from the train-benign cloud (z = max over the 7 frozen features of |x - train-benign mean| / train-benign std)\n\nNumber of test benign sessions per bin:\n\n" + md(cnt) + "\n")
d2 = d[~d.model.str.contains("attacks")].copy(); d2["model"] = d2.model.map(DN)
for t in (1, 2, 4):
    o.append(f"\nSCMA-{t} FAR\n\n" + md(d2[d2.scma == t].pivot(index="zbin", columns="model", values="far").reindex(["<2", "2-3", "3-5", "5-10", ">10"])) + "\n")
d3 = d[d.model.str.contains("attacks")]
o.append("\nRecall of test attacks in the same z bins (no attacks lie beyond z=10):\n\n" + md(d3.pivot_table(index=["scma", "zbin"], columns="model", values="far").reindex(["<2", "2-3", "3-5", "5-10", ">10"], level=1)) + "\n")
o.append(nar("svm"))
r = pd.read_csv(PH / "posthoc_recentered_macro_f1.csv"); r["model"] = r.model.map(DN)
o.append("\n## 4. Per-site re-centering of test inputs (post-hoc diagnostic; descriptive, not a selection)\n")
o.append(nar("recenter_method"))
for t in (1, 2, 4):
    g = r[r.scma == t].pivot(index="model", columns="variant", values=["macro_f1", "recall", "far"])
    g.columns = [f"{a} ({b})" for a, b in g.columns]
    g = g[[c for c in g.columns if "none" in c] + [c for c in g.columns if "benign_stats" in c]].reindex(list(DN.values()))
    o.append(f"\n### SCMA-{t}\n" + md(g) + "\n")
o.append("\nThe `recentered_all_sessions_stats` variant (site statistics from all test sessions, strictly label-free) gives similar macro-F1 (differences up to about 0.03; see `posthoc_recentered_macro_f1.csv`).\n")
o.append(nar("recenter"))
(R.FIN / "POSTHOC_DIAGNOSTICS.md").write_text("\n".join(o), encoding="utf-8")
