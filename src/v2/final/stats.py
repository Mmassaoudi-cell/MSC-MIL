"""Aggregation + statistics for the final test evaluation. Reads results/v2/final/{runs,preds,latency.json}; writes
final_benchmark_comparison.csv, wilcoxon_tests.csv, bootstrap_ci.csv, BENCHMARK_WTL_v2.csv."""
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.v2.final import run_eval as R

FIN = R.FIN
DISPLAY = {"pi_fusion": "PI-Fusion (E5)", "e4_net": "E4 network alone", "msc_mil": "MSC-MIL (orig. tuned)", "source_bilstm_attn": "Source BiLSTM+attn MIL",
           "mlp": "Flatten-MLP", "cnn1d": "1D-CNN", "plain_lstm_mil": "Plain-LSTM-MIL", "transformer_mil": "Transformer-MIL",
           "logreg": "LogReg", "knn": "k-NN", "svm_rbf": "RBF-SVM", "random_forest": "RandomForest", "xgboost": "XGBoost", "lightgbm": "LightGBM",
           "catboost": "CatBoost", "random_forest_pf": "RandomForest-PF", "xgboost_pf": "XGBoost-PF", "lightgbm_pf": "LightGBM-PF", "catboost_pf": "CatBoost-PF"}
MODELS = list(DISPLAY)
MET = ["macro_f1", "recall", "precision", "far", "pr_auc", "balanced_accuracy"]
B, BOOT_SEED = 2000, 20260919


def load_runs():
    rows = []
    for f in sorted(R.RUNS.glob("*.json")):
        if f.name.endswith("FAILED.json"):
            continue
        r = json.loads(f.read_text())
        for var in ("test_full", "test_n200"):
            rows.append(dict(model=r["model"], scma=r["scma"], seed=r["seed"], variant=var, **r[var], train_time_s=r.get("train_time_s"),
                             size=r.get("size"), size_kind=r.get("size_kind"), config_hash=r["config_hash"]))
    return pd.DataFrame(rows)


def aggregate(df):
    lat = json.loads((FIN / "latency.json").read_text()) if (FIN / "latency.json").exists() else {}
    out = []
    for (m, t, v), g in df.groupby(["model", "scma", "variant"]):
        row = dict(model=m, display=DISPLAY[m], scma=t, variant=v, n_seeds=len(g))
        for k in MET:
            row[f"{k}_mean"] = g[k].mean(); row[f"{k}_std"] = g[k].std(ddof=1) if len(g) > 1 else np.nan; row[f"{k}_median"] = g[k].median()
        row["train_time_s_mean"] = g.train_time_s.mean(); row["size_mean"] = g["size"].mean(); row["size_kind"] = g.size_kind.iloc[0]
        l = lat.get(f"{m}_scma{t}", {})
        row["latency_gpu_ms_b1"] = l.get("gpu_ms"); row["latency_cpu_ms_b1"] = l.get("cpu_ms")
        out.append(row)
    a = pd.DataFrame(out)
    a["_o"] = a.model.map({m: i for i, m in enumerate(MODELS)})
    a = a.sort_values(["scma", "variant", "_o"]).drop(columns="_o")
    a.to_csv(FIN / "final_benchmark_comparison.csv", index=False)
    return a


def holm(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p); adj = np.empty(n); run = 0.0
    for i, j in enumerate(order):
        run = max(run, (n - i) * p[j]); adj[j] = min(run, 1.0)
    return adj


def paired_tests(df):
    rows = []
    for t in sorted(df.scma.unique()):
        d = df[(df.scma == t) & (df.variant == "test_full")]
        ref = d[d.model == "pi_fusion"].set_index("seed").macro_f1
        fam = []
        for m in MODELS[1:]:
            o = d[d.model == m].set_index("seed").macro_f1
            common = sorted(set(ref.index) & set(o.index))
            if not common:
                continue
            diff = (ref[common] - o[common]).values
            sd = diff.std(ddof=1) if len(diff) > 1 else np.nan
            dz = diff.mean() / sd if sd and sd > 0 else np.nan
            r = dict(scma=t, model=m, n_pairs=len(common), mean_diff=diff.mean(), cohens_dz=dz, wilcoxon_p=np.nan, tested=False)
            if len(common) >= 8 and np.any(diff != 0):
                r["wilcoxon_p"] = float(wilcoxon(diff).pvalue); r["tested"] = True
            fam.append(r)
        idx = [i for i, r in enumerate(fam) if r["tested"]]
        if idx:
            adj = holm([fam[i]["wilcoxon_p"] for i in idx])
            for i, a in zip(idx, adj):
                fam[i]["holm_p"] = float(a)
        rows += fam
    w = pd.DataFrame(rows)
    w.to_csv(FIN / "wilcoxon_tests.csv", index=False)
    return w


def macro_f1_boot(pred, y_pos_idx, y_neg_idx, pos_s, neg_s):
    tp = pred[pos_s].sum(1); fn = pos_s.shape[1] - tp
    fp = pred[neg_s].sum(1); tn = neg_s.shape[1] - fp
    f1p = 2 * tp / np.maximum(2 * tp + fp + fn, 1); f1n = 2 * tn / np.maximum(2 * tn + fn + fp, 1)
    return (f1p + f1n) / 2


def bootstrap(df, variant="test_full"):
    rows, wtl = [], []
    for t in sorted(df.scma.unique()):
        y = np.load(R.PREDS / f"labels_scma{t}_test_full.npy")
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        rng = np.random.default_rng(BOOT_SEED + t)
        pos_s = rng.choice(pos, size=(B, len(pos))); neg_s = rng.choice(neg, size=(B, len(neg)))   # same resamples for every model
        bf, point = {}, {}
        for m in MODELS:
            fs = sorted((R.PREDS).glob(f"{m}_scma{t}_seed*.npy"))
            fs = [f for f in fs if not f.name.endswith("_n200.npy")]
            if not fs:
                continue
            p = np.mean([np.load(f) for f in fs], axis=0)
            pred = (p >= 0.5).astype(np.int64)
            bf[m] = macro_f1_boot(pred, pos, neg, pos_s, neg_s)
            tp = pred[pos].sum(); fp = pred[neg].sum(); fn = len(pos) - tp; tn = len(neg) - fp
            point[m] = ((2 * tp / (2 * tp + fp + fn)) + (2 * tn / (2 * tn + fn + fp))) / 2
            lo, hi = np.percentile(bf[m], [2.5, 97.5])
            rows.append(dict(scma=t, model=m, n_seeds_averaged=len(fs), macro_f1_seedavg_prob=point[m], ci_lo=lo, ci_hi=hi))
        for m in bf:
            if m == "pi_fusion":
                continue
            dd = bf["pi_fusion"] - bf[m]
            lo, hi = np.percentile(dd, [2.5, 97.5])
            verdict = "win" if lo > 0 else ("loss" if hi < 0 else "tie")
            wtl.append(dict(scma=t, comparator=m, display=DISPLAY[m], point_diff=point["pi_fusion"] - point[m], ci_lo=lo, ci_hi=hi, verdict=verdict))
    b = pd.DataFrame(rows); b.to_csv(FIN / "bootstrap_ci.csv", index=False)
    w = pd.DataFrame(wtl); w.to_csv(FIN / "BENCHMARK_WTL_v2.csv", index=False)
    return b, w


def main():
    df = load_runs()
    a = aggregate(df); w = paired_tests(df); b, wtl = bootstrap(df)
    for t, g in wtl.groupby("scma"):
        c = g.verdict.value_counts().to_dict()
        print(f"SCMA-{t}: wins {c.get('win', 0)} / ties {c.get('tie', 0)} / losses {c.get('loss', 0)}")
    print("done", len(df))


if __name__ == "__main__":
    main()
