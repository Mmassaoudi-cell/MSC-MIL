import pandas as pd, numpy as np
from src.external.prep_common import RES
ORDER=["PROPOSED","msc_e4","msc_aug_tuned","tree_aug","msc_mil","bilstm_attn_mil","transformer_mil","plain_lstm_mil","xgboost","lightgbm","catboost","random_forest","logreg"]
CONDS=[("clean",0.0),("noise",.05),("noise",.15),("noise",.30),("missing",.1),("missing",.25),("missing",.5)]
lab=lambda k,s:"clean" if k=="clean" else f"{'noise' if k=='noise' else 'miss'} {int(round(s*100))}%"
out=[]
for ds in ("cic2017","edge_iiot"):
    g=pd.read_csv(RES/f"final_agg_{ds}.csv"); b=pd.read_csv(RES/f"final_bootstrap_{ds}.csv")
    out.append(f"\n### {ds}: test macro-F1@0.5, mean +- std over 5 seeds\n")
    out.append("| model | "+" | ".join(lab(k,s) for k,s in CONDS)+" | drop clean->noise30 | drop clean->miss50 |\n|---|"+"---|"*9)
    for m in ORDER:
        r=g[g.model==m]; cells=[]; v={}
        for k,s in CONDS:
            x=r[(r.kind==k)&(np.isclose(r.severity,s))].iloc[0]; cells.append(f"{x.macro_f1_mean:.4f} +- {x.macro_f1_std:.4f}"); v[(k,s)]=x.macro_f1_mean
        out.append(f"| {m} | "+" | ".join(cells)+f" | {v[('clean',0.0)]-v[('noise',.30)]:+.4f} | {v[('clean',0.0)]-v[('missing',.5)]:+.4f} |")
    out.append(f"\n{ds}: clean recall / precision / FAR / ROC-AUC (mean over seeds)\n")
    out.append("| model | recall | precision | FAR | ROC-AUC | PR-AUC |\n|---|---|---|---|---|---|")
    for m in ORDER:
        x=g[(g.model==m)&(g.kind=="clean")].iloc[0]; out.append(f"| {m} | {x.recall_mean:.4f} | {x.precision_mean:.4f} | {x.far_mean:.4f} | {x.roc_auc_mean:.4f} | {x.pr_auc_mean:.4f} |")
    out.append(f"\n{ds}: paired bootstrap difference in macro-F1, PROPOSED minus baseline [95% CI] (2000 stratified resamples; * = CI excludes 0)\n")
    out.append("| baseline | "+" | ".join(lab(k,s) for k,s in CONDS)+" |\n|---|"+"---|"*7)
    for m in ORDER[1:]:
        cells=[]
        for k,s in CONDS:
            x=b[(b.model==m)&(b.kind==k)&(np.isclose(b.severity,s))].iloc[0]
            star="*" if (x.diff_ci_lo>0 or x.diff_ci_hi<0) else ""
            cells.append(f"{x.diff_proposed_minus_this:+.4f} [{x.diff_ci_lo:+.4f},{x.diff_ci_hi:+.4f}]{star}")
        out.append(f"| {m} | "+" | ".join(cells)+" |")
(RES/"_final_tables_autogen.md").write_text("\n".join(out)); print("\n".join(out))
