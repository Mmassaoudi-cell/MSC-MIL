"""Part C: MSC-MIL vs. each of the 12 other methods (11 new benchmarks +
SOURCE_METHOD_REPRODUCTION), paired on test_full macro-F1 across seeds.

Test choice: Wilcoxon signed-rank test (paired, non-parametric) rather than a
paired t-test -- chosen because n=5 seeds is too small to safely assume
normality of the per-seed macro-F1 differences, and Wilcoxon is the standard
choice for small-n paired comparisons in this kind of seed-level ML
benchmarking (consistent choice applied uniformly to every testable
comparison, not cherry-picked per comparison).

Multiple-comparison correction: Holm-Bonferroni across the TESTABLE
comparisons only (both sides have >=2 seeds so a paired test is meaningful).
Deterministic single-seed benchmarks (LogReg, kNN: std=0, no random_state
affects their fit given fixed data) are excluded from the correction family
and flagged "no variance to test significance against" -- a point-difference
and paired effect size (Cohen's dz, informally computed against MSC-MIL's own
seed spread) are still reported, never a fabricated p-value.

Writes results/aggregate/BENCHMARK_WTL.csv (project root copy: BENCHMARK_WTL.csv).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.benchmarks.data_utils import PROJECT_ROOT

RAW_DIR = PROJECT_ROOT / "results" / "raw" / "final_eval"
ALPHA = 0.05


def load_macro_f1_by_seed(records, method):
    d = {}
    for r in records:
        if r["method"] == method and r["test_variant"] == "test_full":
            d[r["seed"]] = r["metrics"]["macro_f1"]
    return d


def holm_bonferroni(pvals: list[float], alpha: float = ALPHA):
    """Returns list of (reject: bool, adjusted_alpha_threshold) in the ORIGINAL
    input order. Standard Holm step-down procedure."""
    m = len(pvals)
    order = np.argsort(pvals)
    thresholds = np.empty(m)
    reject = np.empty(m, dtype=bool)
    still_rejecting = True
    for rank, idx in enumerate(order):
        thr = alpha / (m - rank)
        thresholds[idx] = thr
        if still_rejecting and pvals[idx] <= thr:
            reject[idx] = True
        else:
            still_rejecting = False
            reject[idx] = False
    return reject.tolist(), thresholds.tolist()


def classify(msc_mean, other_mean, p, reject_h0):
    if p is None:
        if msc_mean > other_mean:
            return "practically superior (no variance to test significance against)"
        elif msc_mean < other_mean:
            return "inferior (no variance to test significance against, point-difference only)"
        else:
            return "tie (deterministic, identical point estimate)"
    if reject_h0:
        return "statistically superior" if msc_mean > other_mean else "statistically inferior"
    else:
        if msc_mean > other_mean:
            return "practically superior (nominal win, not significant)"
        elif msc_mean < other_mean:
            return "statistically indistinguishable (nominal loss, not significant)"
        else:
            return "statistically indistinguishable (exact tie)"


def main():
    with open(RAW_DIR / "all_methods_per_seed_metrics.json") as f:
        records = json.load(f)

    methods = sorted(set(r["method"] for r in records if r["method"] != "MSC-MIL"))
    msc_by_seed = load_macro_f1_by_seed(records, "MSC-MIL")
    print(f"[stats_wtl] MSC-MIL test_full macro_f1 by seed: {msc_by_seed}")

    comparisons = []
    for method in methods:
        other_by_seed = load_macro_f1_by_seed(records, method)
        common_seeds = sorted(set(msc_by_seed) & set(other_by_seed))
        msc_vals = np.array([msc_by_seed[s] for s in common_seeds])
        other_vals = np.array([other_by_seed[s] for s in common_seeds])
        msc_mean, other_mean = float(msc_vals.mean()), float(other_vals.mean())
        diff = msc_vals - other_vals
        point_diff = msc_mean - other_mean

        n_other_seeds = len(other_by_seed)
        n_common = len(common_seeds)
        testable = n_other_seeds >= 2 and n_common >= 2 and not np.allclose(diff, diff[0])

        p = None
        dz = None
        if n_common >= 2:
            std_diff = diff.std(ddof=1) if n_common > 1 else 0.0
            dz = float(diff.mean() / std_diff) if std_diff > 1e-12 else (np.inf if diff.mean() > 0 else (-np.inf if diff.mean() < 0 else 0.0))
        if testable:
            try:
                stat, p = wilcoxon(msc_vals, other_vals, zero_method="wilcox", alternative="two-sided")
                p = float(p)
            except ValueError as e:
                p = None  # e.g. all differences zero after dropping ties

        comparisons.append({
            "method": method, "msc_mean_macro_f1": msc_mean, "other_mean_macro_f1": other_mean,
            "point_diff": point_diff, "effect_size_dz": dz, "n_msc_seeds": len(common_seeds),
            "n_other_seeds": n_other_seeds, "p_value_raw": p, "testable": p is not None,
        })

    testable_idx = [i for i, c in enumerate(comparisons) if c["p_value_raw"] is not None]
    pvals = [comparisons[i]["p_value_raw"] for i in testable_idx]
    if pvals:
        reject, thresholds = holm_bonferroni(pvals, ALPHA)
    else:
        reject, thresholds = [], []
    for j, i in enumerate(testable_idx):
        comparisons[i]["holm_alpha_threshold"] = thresholds[j]
        comparisons[i]["reject_h0_after_holm"] = reject[j]
    for c in comparisons:
        if not c["testable"]:
            c["holm_alpha_threshold"] = None
            c["reject_h0_after_holm"] = None

    rows = []
    wins = ties = losses = 0
    for c in comparisons:
        cls = classify(c["msc_mean_macro_f1"], c["other_mean_macro_f1"], c["p_value_raw"], c["reject_h0_after_holm"])
        if "superior" in cls:
            wins += 1
        elif "inferior" in cls:
            losses += 1
        else:
            ties += 1
        rows.append({
            "method": c["method"],
            "msc_mean_macro_f1_test_full": round(c["msc_mean_macro_f1"], 4),
            "other_mean_macro_f1_test_full": round(c["other_mean_macro_f1"], 4),
            "point_diff": round(c["point_diff"], 4),
            "effect_size_cohens_dz": round(c["effect_size_dz"], 4) if c["effect_size_dz"] not in (None, np.inf, -np.inf) else c["effect_size_dz"],
            "n_msc_seeds": c["n_msc_seeds"],
            "n_other_seeds": c["n_other_seeds"],
            "wilcoxon_p_raw": round(c["p_value_raw"], 5) if c["p_value_raw"] is not None else "N/A (no variance to test against)",
            "holm_alpha_threshold": round(c["holm_alpha_threshold"], 5) if c["holm_alpha_threshold"] is not None else "N/A",
            "reject_h0_after_holm": c["reject_h0_after_holm"] if c["reject_h0_after_holm"] is not None else "N/A",
            "classification": cls,
        })

    df = pd.DataFrame(rows).sort_values("method").reset_index(drop=True)
    summary_line = f"MSC-MIL: {wins} wins / {ties} ties / {losses} losses (of {len(comparisons)} comparisons, test_full macro-F1, Wilcoxon signed-rank + Holm-Bonferroni, alpha={ALPHA})"
    print(f"[stats_wtl] {summary_line}")

    out_path_agg = PROJECT_ROOT / "results" / "aggregate" / "BENCHMARK_WTL.csv"
    out_path_root = PROJECT_ROOT / "BENCHMARK_WTL.csv"
    for out_path in (out_path_agg, out_path_root):
        with open(out_path, "w", newline="") as f:
            df.to_csv(f, index=False)
            f.write(f"# {summary_line}\n")
    print(f"[stats_wtl] wrote {out_path_agg} and {out_path_root}")
    return df, summary_line


if __name__ == "__main__":
    main()
