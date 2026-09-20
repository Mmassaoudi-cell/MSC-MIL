"""Builds results/v2/final/FINAL_EVAL_REPORT.md from the CSVs. The hand-written 'reading' (reading.md) is inlined."""
import pandas as pd

from src.v2.final import run_eval as R
from src.v2.final.stats import DISPLAY, MODELS

FIN = R.FIN


def f(x, d=4):
    return "n/a" if pd.isna(x) else f"{x:.{d}f}"


def md(df, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    a = pd.read_csv(FIN / "final_benchmark_comparison.csv")
    b = pd.read_csv(FIN / "bootstrap_ci.csv")
    w = pd.read_csv(FIN / "BENCHMARK_WTL_v2.csv")
    wc = pd.read_csv(FIN / "wilcoxon_tests.csv")
    out = ["# FINAL_EVAL_REPORT (v2 benchmark, single test evaluation)\n"]
    out.append(open(FIN / "reading_head.md").read() if (FIN / "reading_head.md").exists() else "")
    out.append("## 1. Main table: test_full macro-F1 (mean +/- std over seeds; median in brackets)\n")
    rows = []
    for m in MODELS:
        r = {"model": DISPLAY[m]}
        for t in (1, 2, 4):
            g = a[(a.model == m) & (a.scma == t) & (a.variant == "test_full")]
            if len(g):
                g = g.iloc[0]
                r[f"SCMA-{t}"] = f"{g.macro_f1_mean:.4f} +/- {f(g.macro_f1_std)} [{g.macro_f1_median:.4f}] (n={g.n_seeds})"
            else:
                r[f"SCMA-{t}"] = "-"
        rows.append(r)
    out.append(md(pd.DataFrame(rows), ["model", "SCMA-1", "SCMA-2", "SCMA-4"]))
    for t in (1, 2, 4):
        out.append(f"\n### Full metric set, SCMA-{t}, test_full (mean over seeds)\n")
        g = a[(a.scma == t) & (a.variant == "test_full")].copy()
        g["model"] = g.model.map(DISPLAY)
        for k in ("macro_f1", "recall", "precision", "far", "pr_auc", "balanced_accuracy"):
            g[k] = g[f"{k}_mean"].map(lambda x: f"{x:.4f}")
        out.append(md(g, ["model", "n_seeds", "macro_f1", "recall", "precision", "far", "pr_auc", "balanced_accuracy"]))
    out.append("\n### test_n200 macro-F1 (200 sessions; information only, 20 attacks so very noisy)\n")
    rows = []
    for m in MODELS:
        r = {"model": DISPLAY[m]}
        for t in (1, 2, 4):
            g = a[(a.model == m) & (a.scma == t) & (a.variant == "test_n200")]
            r[f"SCMA-{t}"] = f"{g.iloc[0].macro_f1_mean:.4f} +/- {f(g.iloc[0].macro_f1_std)}" if len(g) else "-"
        rows.append(r)
    out.append(md(pd.DataFrame(rows), ["model", "SCMA-1", "SCMA-2", "SCMA-4"]))

    out.append("\n## 2. Bootstrap CIs over test sessions (seed-averaged probability, 2000 label-stratified resamples, 95%)\n")
    for t in (1, 2, 4):
        out.append(f"\n### SCMA-{t}\n")
        g = b[b.scma == t].copy()
        g = g.merge(w[w.scma == t][["comparator", "point_diff", "ci_lo", "ci_hi", "verdict"]].rename(columns={"comparator": "model", "ci_lo": "d_lo", "ci_hi": "d_hi"}), on="model", how="left")
        g["model"] = g.model.map(DISPLAY)
        g["macro-F1 [95% CI]"] = g.apply(lambda r: f"{r.macro_f1_seedavg_prob:.4f} [{r.ci_lo:.4f}, {r.ci_hi:.4f}]", axis=1)
        g["PI-Fusion minus model [95% CI]"] = g.apply(lambda r: "-" if pd.isna(r.point_diff) else f"{r.point_diff:+.4f} [{r.d_lo:+.4f}, {r.d_hi:+.4f}]", axis=1)
        g["verdict"] = g.verdict.fillna("-")
        out.append(md(g, ["model", "n_seeds_averaged", "macro-F1 [95% CI]", "PI-Fusion minus model [95% CI]", "verdict"]))

    out.append("\n## 3. Wins / ties / losses of PI-Fusion (paired bootstrap CI of the difference)\n")
    for t, g in w.groupby("scma"):
        c = g.verdict.value_counts().to_dict()
        out.append(f"* **SCMA-{t}: {c.get('win', 0)} wins / {c.get('tie', 0)} ties / {c.get('loss', 0)} losses** (of {len(g)} comparators). Wins: {', '.join(g[g.verdict == 'win'].display) or '-'}; ties: {', '.join(g[g.verdict == 'tie'].display) or '-'}; losses: {', '.join(g[g.verdict == 'loss'].display) or '-'}.")
    out.append("\nFile: `BENCHMARK_WTL_v2.csv`.\n")

    out.append("\n## 4. Paired seed tests (PI-Fusion minus model, macro-F1 on test_full)\n")
    out.append("Wilcoxon signed-rank only where both sides have >= 8 seeds (SCMA-1 only for the models with 10 seeds); Holm correction within the family of tested comparisons of that SCMA type. Elsewhere only the point difference and Cohen's dz are given (seed count limits testing; dz with 3-5 pairs is very unstable).\n")
    wc["model"] = wc.model.map(DISPLAY)
    for k in ("mean_diff", "cohens_dz", "wilcoxon_p", "holm_p"):
        if k not in wc:
            wc[k] = float("nan")
        wc[k] = wc[k].map(lambda x: "n/a" if pd.isna(x) else f"{x:.4g}")
    out.append(md(wc, ["scma", "model", "n_pairs", "mean_diff", "cohens_dz", "wilcoxon_p", "holm_p"]))

    out.append("\n## 5. Efficiency (SCMA-1; per-type values in the CSV)\n")
    g = a[(a.scma == 1) & (a.variant == "test_full")].copy()
    g["model"] = g.model.map(DISPLAY)
    g["train_s"] = g.train_time_s_mean.map(lambda x: f"{x:.1f}")
    g["size"] = g.apply(lambda r: f"{int(r['size_mean']):,} ({r.size_kind})", axis=1)
    g["lat_gpu"] = g.latency_gpu_ms_b1.map(lambda x: "n/a" if pd.isna(x) else f"{x:.2f}")
    g["lat_cpu"] = g.latency_cpu_ms_b1.map(lambda x: "n/a" if pd.isna(x) else f"{x:.2f}")
    out.append(md(g, ["model", "size", "train_s", "lat_gpu", "lat_cpu"]))
    out.append("\nLatency: median of 200 batch-1 predictions after 10 warm-ups on seed-0 checkpoints, measured after all training had finished (trees: features + predict_proba on CPU; PI-Fusion: network on GPU/CPU + tree features + XGBoost on CPU). Training time is wall clock; deep-model runs shared the machine with tree jobs, so times are indicative.\n")
    out.append(open(FIN / "reading.md").read() if (FIN / "reading.md").exists() else "")
    (FIN / "FINAL_EVAL_REPORT.md").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
