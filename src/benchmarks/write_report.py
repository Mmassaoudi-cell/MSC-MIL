"""Builds BENCHMARK_RESULTS.md (project root) from the CSVs final_eval.py and
stats_wtl.py already wrote. Pure formatting -- no numbers are computed here
that weren't already computed and saved by those two scripts."""
from __future__ import annotations

import json

import pandas as pd

from src.benchmarks.data_utils import PROJECT_ROOT

AGG_DIR = PROJECT_ROOT / "results" / "aggregate"
RAW_DIR = PROJECT_ROOT / "results" / "raw" / "final_eval"

METHOD_LABELS = {
    "MSC-MIL": "MSC-MIL (final, frozen)",
    "SOURCE_METHOD_REPRODUCTION": "SOURCE_METHOD_REPRODUCTION (MIL-BiLSTM+attn, reused)",
    "mlp": "MLP (flatten, no-sequence control)",
    "cnn1d": "1D-CNN",
    "plain_lstm_mil": "Plain-LSTM-MIL (mean-pool, attention ablation)",
    "transformer_mil": "Transformer-MIL",
    "logreg": "Logistic Regression",
    "svm_rbf": "SVM (RBF)",
    "knn": "k-NN",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}


def fmt(mean, std, n):
    if pd.isna(mean):
        return "N/A"
    if n and n > 1:
        return f"{mean:.4f} +/- {std:.4f} (n={int(n)})"
    return f"{mean:.4f} (n={int(n)}, std=0)"


def main():
    df = pd.read_csv(AGG_DIR / "final_benchmark_comparison.csv")
    wtl = pd.read_csv(AGG_DIR / "BENCHMARK_WTL.csv", comment="#")
    with open(AGG_DIR / "BENCHMARK_WTL.csv") as f:
        lines = f.readlines()
    summary_line = [l.strip("# \n") for l in lines if l.startswith("#")][0]

    sanity_path = RAW_DIR / "config_sanity_check.json"
    sanity = json.load(open(sanity_path)) if sanity_path.exists() else None

    methods = sorted(df["method"].unique())

    lines_out = []
    lines_out.append("# BENCHMARK_RESULTS.md\n")
    lines_out.append(
        "Central comparison of the frozen final MSC-MIL model against 11 newly "
        "implemented benchmarks and the already-reproduced SOURCE_METHOD_REPRODUCTION "
        "baseline, evaluated on SCMA-1 `test_n200` and `test_full`. This is the "
        "**single, final TEST evaluation** for the new model-selection work in this "
        "project -- TEST was never read during Part A (validation-only tuning) or "
        "during any of Stage 1-3's candidate screening.\n"
    )

    lines_out.append("## Config-file sanity check (FINAL_MODEL_CONFIG.yaml)\n")
    if sanity:
        lines_out.append(
            f"Retraining MSC-MIL seed=0 from the literal YAML file gave val macro-F1 = "
            f"**{sanity['final_retrained_seed0_val_macro_f1']:.4f}**, vs. Stage 3's reported "
            f"single-seed value of **{sanity['stage3_reported_seed0_val_macro_f1']:.4f}** "
            f"(diff = {sanity['diff']:+.4f}). "
            + ("This is a close match; the config file reproduces Stage 3's result." if abs(sanity['diff']) < 0.02
               else "This is NOT a close match -- flagged as a genuine reproduction discrepancy, not silently patched.")
            + "\n"
        )
    else:
        lines_out.append("Sanity check file not found.\n")

    lines_out.append("## Main comparison table (test_full, SCMA-1)\n")
    lines_out.append("| Method | Macro-F1 | FAR | Recall | Precision | Latency (ms) | Params |\n")
    lines_out.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for m in methods:
        sub = df[(df.method == m) & (df.test_variant == "test_full")].set_index("metric")
        def g(metric):
            if metric in sub.index:
                row = sub.loc[metric]
                return fmt(row["mean"], row["std"], row["n_seeds"])
            return "N/A"
        label = METHOD_LABELS.get(m, m)
        lines_out.append(f"| {label} | {g('macro_f1')} | {g('false_alarm_rate')} | {g('recall')} | {g('precision')} | {g('latency_ms')} | {g('n_params')} |\n")

    lines_out.append("\n## Main comparison table (test_n200, SCMA-1, paper-matched N=200)\n")
    lines_out.append("| Method | Macro-F1 | FAR | Recall | Precision |\n")
    lines_out.append("|---|---:|---:|---:|---:|\n")
    for m in methods:
        sub = df[(df.method == m) & (df.test_variant == "test_n200")].set_index("metric")
        def g(metric):
            if metric in sub.index:
                row = sub.loc[metric]
                return fmt(row["mean"], row["std"], row["n_seeds"])
            return "N/A"
        label = METHOD_LABELS.get(m, m)
        lines_out.append(f"| {label} | {g('macro_f1')} | {g('false_alarm_rate')} | {g('recall')} | {g('precision')} |\n")

    lines_out.append("\n## Win / Tie / Loss vs. MSC-MIL (test_full macro-F1, paired across seeds)\n")
    lines_out.append(f"**{summary_line}**\n\n")
    lines_out.append(wtl.to_markdown(index=False) + "\n")

    out_path = PROJECT_ROOT / "BENCHMARK_RESULTS.md"
    with open(out_path, "w") as f:
        f.writelines(lines_out)
    print(f"[write_report] wrote {out_path}")


if __name__ == "__main__":
    main()
