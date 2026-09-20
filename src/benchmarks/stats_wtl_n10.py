"""Part 1 step 2: re-run the Wilcoxon signed-rank + Holm-Bonferroni test at
n=10 seeds for MSC-MIL vs. the 4 tree/boosting benchmarks (CatBoost, XGBoost,
LightGBM, RandomForest) ONLY -- reading
results/raw/final_eval/n10_power_analysis_test_full.json (written by
extend_seeds_n10.py). The other 8 comparisons in BENCHMARK_WTL.csv are left
untouched at n=5 (not silently mixed with these n=10 rows) -- this script
only overwrites the 4 tree-method rows and appends a documentation note.

Holm-Bonferroni here is applied across the SAME family of 4 tests being
re-run at n=10 (not re-pooled with the other 8 n=5 comparisons, which were
already independently corrected within their own original n=5 family in
BENCHMARK_RESULTS.md / stats_wtl.py) -- documented choice: mixing n=5 and
n=10 p-values into one Holm family would not be a meaningful correction
(different power under the null per comparison), so the n=10 subset is
corrected as its own family of 4, consistent with how it will be reported
(a distinctly-flagged, adequately-powered re-run of this specific 4-comparison
subset).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.benchmarks.data_utils import PROJECT_ROOT
from src.benchmarks.stats_wtl import classify, holm_bonferroni

RAW_DIR = PROJECT_ROOT / "results" / "raw" / "final_eval"
ALPHA = 0.05
TREE_METHODS = ["catboost", "xgboost", "lightgbm", "random_forest"]


def load_macro_f1_by_seed(records, method):
    d = {}
    for r in records:
        if r["method"] == method and r["test_variant"] == "test_full":
            d[r["seed"]] = r["metrics"]["macro_f1"]
    return d


def main():
    with open(RAW_DIR / "n10_power_analysis_test_full.json") as f:
        records = json.load(f)

    msc_by_seed = load_macro_f1_by_seed(records, "MSC-MIL")
    print(f"[stats_wtl_n10] MSC-MIL test_full macro_f1 by seed (n={len(msc_by_seed)}): {msc_by_seed}")
    assert len(msc_by_seed) == 10, f"expected 10 MSC-MIL seeds, got {len(msc_by_seed)}"

    comparisons = []
    for method in TREE_METHODS:
        other_by_seed = load_macro_f1_by_seed(records, method)
        common_seeds = sorted(set(msc_by_seed) & set(other_by_seed))
        assert len(common_seeds) == 10, f"{method}: expected 10 common seeds, got {len(common_seeds)}"
        msc_vals = np.array([msc_by_seed[s] for s in common_seeds])
        other_vals = np.array([other_by_seed[s] for s in common_seeds])
        msc_mean, other_mean = float(msc_vals.mean()), float(other_vals.mean())
        diff = msc_vals - other_vals
        point_diff = msc_mean - other_mean
        std_diff = diff.std(ddof=1)
        dz = float(diff.mean() / std_diff) if std_diff > 1e-12 else (np.inf if diff.mean() > 0 else (-np.inf if diff.mean() < 0 else 0.0))
        stat, p = wilcoxon(msc_vals, other_vals, zero_method="wilcox", alternative="two-sided")
        p = float(p)
        comparisons.append({
            "method": method, "msc_mean_macro_f1": msc_mean, "other_mean_macro_f1": other_mean,
            "point_diff": point_diff, "effect_size_dz": dz, "n_msc_seeds": 10, "n_other_seeds": 10,
            "p_value_raw": p,
        })
        print(f"[stats_wtl_n10] {method}: msc={msc_mean:.4f} other={other_mean:.4f} diff={point_diff:.4f} "
              f"dz={dz:.4f} wilcoxon_p={p:.5f}")

    pvals = [c["p_value_raw"] for c in comparisons]
    reject, thresholds = holm_bonferroni(pvals, ALPHA)
    for c, r, t in zip(comparisons, reject, thresholds):
        c["reject_h0_after_holm"] = r
        c["holm_alpha_threshold"] = t

    rows = []
    for c in comparisons:
        cls = classify(c["msc_mean_macro_f1"], c["other_mean_macro_f1"], c["p_value_raw"], c["reject_h0_after_holm"])
        rows.append({
            "method": c["method"],
            "msc_mean_macro_f1_test_full": round(c["msc_mean_macro_f1"], 4),
            "other_mean_macro_f1_test_full": round(c["other_mean_macro_f1"], 4),
            "point_diff": round(c["point_diff"], 4),
            "effect_size_cohens_dz": round(c["effect_size_dz"], 4) if c["effect_size_dz"] not in (np.inf, -np.inf) else c["effect_size_dz"],
            "n_msc_seeds": c["n_msc_seeds"], "n_other_seeds": c["n_other_seeds"],
            "wilcoxon_p_raw": round(c["p_value_raw"], 5),
            "holm_alpha_threshold": round(c["holm_alpha_threshold"], 5),
            "reject_h0_after_holm": c["reject_h0_after_holm"],
            "classification": cls,
            "note": "RE-RUN AT n=10 SEEDS (Part 1 power analysis, seeds 0-9) -- distinct Holm family of 4, NOT pooled with the n=5 comparisons below",
        })
    df_n10 = pd.DataFrame(rows)

    # ---- Merge into BENCHMARK_WTL.csv: replace the 4 tree rows with the n=10
    # version, keep the other 8 rows at n=5 untouched, add a documentation note. ----
    wtl_path_agg = PROJECT_ROOT / "results" / "aggregate" / "BENCHMARK_WTL.csv"
    wtl_path_root = PROJECT_ROOT / "BENCHMARK_WTL.csv"
    old_df = pd.read_csv(wtl_path_agg, comment="#")
    old_df["note"] = "original n=5 comparison, unchanged (see BENCHMARK_RESULTS.md)"
    kept = old_df[~old_df["method"].isin(TREE_METHODS)].copy()
    merged = pd.concat([kept, df_n10], ignore_index=True).sort_values("method").reset_index(drop=True)

    n_reject = int(sum(reject))
    summary_line = (
        f"Part 1 power-analysis update: MSC-MIL vs. {{catboost,xgboost,lightgbm,random_forest}} "
        f"re-run at n=10 seeds (0-9), Wilcoxon signed-rank + Holm-Bonferroni (family of 4, alpha={ALPHA}): "
        f"{n_reject}/4 comparisons reach significance after correction. "
        f"All other 8 comparisons remain at the original n=5 (documented in the 'note' column, not silently mixed)."
    )
    print(f"[stats_wtl_n10] {summary_line}")

    for out_path in (wtl_path_agg, wtl_path_root):
        with open(out_path, "w", newline="") as f:
            merged.to_csv(f, index=False)
            f.write(f"# {summary_line}\n")
            f.write("# Original n=5 WTL summary (BENCHMARK_RESULTS.md): MSC-MIL: 2 wins / 9 ties / 1 losses (of 12 comparisons)\n")
    print(f"[stats_wtl_n10] wrote {wtl_path_agg} and {wtl_path_root}")

    out_json = RAW_DIR / "n10_power_analysis_stats.json"
    with open(out_json, "w") as f:
        json.dump({"comparisons": comparisons, "summary": summary_line}, f, indent=2, default=str)
    print(f"[stats_wtl_n10] wrote {out_json}")
    return merged, summary_line


if __name__ == "__main__":
    main()
