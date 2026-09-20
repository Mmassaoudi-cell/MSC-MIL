"""Render results/v2/V2_DATA_REPORT.md from results/v2/v2_sanity.json."""
import json
from src.v2.common import RESULTS_V2
from src.v2 import dataset_v2 as D2

s = json.load(open(RESULTS_V2 / "v2_sanity.json"))
L = []
A = L.append
A("# V2_DATA_REPORT: curve-preserving attack realization")
A("")
A("All numbers below come from `src/v2/v2_sanity.py` (raw: `results/v2/v2_sanity.json`) run on the cached arrays in `data/processed_v2/`. "
  "Only TRAIN and VAL arrays were used for any check or decision; test arrays were constructed (Part 3) but never inspected.")
A("")
A("## Construction record")
A("- Aggregate attack logic: unchanged `scma_env.generate_scma`, frozen PPO attacker `results/checkpoints/ppo_attacker.pt`, `predict_c_attack` (policy mean) and, for train variants, samples from its Gaussian policy N(mu_i, exp(log_std)^2) (policy std = 0.576).")
A("- Realization (`src/v2/attack_realize.py`): window slots from the attacked aggregate (shift = round((t_start'-t_start)/5min), cut = round((t_end-t_end')/5min) relative to the real active run); real curve shifted/truncated into the new window; inactive slots = 0 and mask False; curve multiplied by (E'/E) * E_full_curve / E_retained; clipped to [0, 20] kW (`MAX_EVSE_KW` of preprocessing). Static features come from the attacked aggregate exactly as in v1.")
A(f"- Train: same 4000 benign + same 444 attack-source sessions as v1; each source gets **{D2.TRAIN_VARIANTS} variants** (variant 0 = policy mean, variants 1-8 = stochastic C_attack, RNG seed {D2.TRAIN_C_SEED}; the same C draws are reused for all six SCMA types) -> 3996 positives, 7996 bags, 49.98% positive (v1: 8028 bags, ~50.2%, with ADASYN). No ADASYN.")
A(f"- Val: the same 800 benign as v1 + **new** attack sources (seed {D2.VAL_ATTACK_SEED}): 89 attacks (10%). test_full: all 7945 benign + 883 attacks (seed {D2.FULL_ATTACK_SEED}); test_n200: same 180 benign as v1 + 20 attacks (seed {D2.N200_ATTACK_SEED}). Val/test use the policy-mean C_attack (as v1 deployment). Attack sources are shared across the six SCMA types.")
A("- Scalers: fit on TRAIN benign bags only (reusing `dataset.fit_scalers`/`bag_to_tensors`). Split = unchanged `DATA_SPLIT_MANIFEST.csv`.")
A("- Tree features: `summary_features_v2` = v1 `flatten_features` semantics (train-scaled; the 'start' slot is the connection-hour proxy, exactly as in v1). Raw-unit 7-dim summaries (true t_start_h, kWh, kW) are in the cache (`load_extras(t)[split]['raw_summary']`).")
A("")
a = s["a_auroc_scma1_val"]
A("## (a) Single-feature AUROC, SCMA-1, VAL, v1 vs v2")
A("")
A("Oriented AUROC = max(a, 1-a). Raw AUROC (attack=positive) in brackets.")
A("")
A("| feature | v1 | v2 |")
A("|---|---|---|")
for k, v in a.items():
    A(f"| {k} | {v['v1_oriented']:.4f} ({v['v1_raw']:.4f}) | {v['v2_oriented']:.4f} ({v['v2_raw']:.4f}) |")
z = s["a_std_zero_frac_v2_val"]
A("")
A(f"v2 SCMA-1 val: attacked bags with active std < 1e-6 = {z['attack_std_lt_1e-6']*100:.1f}% (v1: 100.0%; benign {z['benign_std_lt_1e-6']*100:.2f}%). "
  "These remaining 11% are attacks whose attacked energy is 0 (the frozen policy's C_attack gives E' = max(., 0) = 0 for short sessions): a zero curve is the faithful realization of a zero-demand aggregate. "
  "The shortcut on std/max is therefore largely, not entirely, removed: std AUROC 0.9969 -> 0.6171, max 0.8916 -> 0.7273.")
A("")
A("Extra (val only): tuned-v1 CatBoost config trained on v2 SCMA-1 train, VAL Gaussian noise (identical `perturb_noise` protocol, 5 seeds, mean +/- std macro-F1):")
A("")
A("| feature set | clean | 5% | 15% | 30% |")
A("|---|---|---|---|---|")
for k, r in s["extra_v2_scma1_catboost_noise_val"].items():
    A(f"| {k} | " + " | ".join(f"{r[l]['mean']:.4f} +/- {r[l]['std']:.4f}" for l in ("0.0", "0.05", "0.15", "0.3")) + " |")
A("")
A("(v1 for comparison, from DIAGNOSIS_v1.md: full 0.9988 / 0.4737 / 0.4737 / 0.4737.) In v2 the tree no longer falls off a cliff at 5%, but it is also much less accurate on clean data; without std/max it is near chance.")
A("")
A("## (b) Shape preservation and energy consistency (attacked bags; val, with train in the JSON)")
A("")
A("`corr` = Pearson correlation between the original real active curve (over the retained slots) and the attacked active curve; only defined when the retained window has >= 3 slots and non-constant values. It is 1.0 by construction wherever no clipping occurs (the transform is a positive rescale of a slice), so it mainly certifies the implementation; the informative parts are the defined fraction and the clipped fraction. In v1 the corresponding correlation is undefined/0 by construction (the attacked curve is constant).")
A("")
A("| SCMA | corr mean (defined) | corr defined frac | frac windows empty | window length / original (p5, p50, p95) | frac clipped at 20 kW | energy within 5% of target | median rel. energy err vs attacked aggregate | scale (p5, p50, p95) |")
A("|---|---|---|---|---|---|---|---|---|")
for t, v in s["b_shape_energy"].items():
    x = v["val"]
    f3 = lambda l: "(" + ", ".join(f"{q:.3f}" for q in l) + ")"
    A(f"| {t} | {x['corr_mean_all']:.4f} | {x['corr_defined_frac']:.3f} | {x['frac_empty_window']:.3f} | {f3(x['n_new_frac_of_orig_pcts_5_50_95'])} | {x['frac_clipped']:.3f} | {x['energy_within5pct_of_target']:.3f} | {x['energy_rel_err_vs_attacked_agg_median']:.3f} | {f3(x['scale_pcts_5_50_95'])} |")
A("")
A("Notes. (i) 'target' = (E'/E) x energy of the full real curve; 'attacked aggregate' = E' in the static features. They differ slightly because the real measured curve's energy is within [0.993, 1.087] (p5-p95) of the recorded kWh (`e_full_over_kwh`). "
  "(ii) Unclipped bags hit the target within 1% in 100% of cases (see JSON `unclipped_energy_within1pct_of_target`). "
  "(iii) For SCMA-2..6 the reconstructed aggregate logic shrinks the window to a median of ~7% of the original (SCMA-6: mostly to zero), so the same energy must be delivered in a few slots; the 20 kW cap then bites in 33-44% of SCMA-2..5 attacks and the delivered energy falls short of the aggregate (median shortfall among clipped bags 0.35-0.48). This is the 'as closely as the truncated curve allows' clause, not a bug, but it reintroduces a smaller flat tell (a 20 kW plateau) for those types; see residual tells below.")
A("")
A("Residual flat-curve tells in v2 (val, raw units): fraction of attacked bags with std < 1e-6 / max at the 20 kW cap; benign with the same:")
A("")
A("| SCMA | attack std<1e-6 | benign std<1e-6 | attack max = 20 kW | benign max >= 20 kW | attack empty window |")
A("|---|---|---|---|---|---|")
for t, v in s["b_shape_energy"].items():
    x = v["val"]
    A(f"| {t} | {x['frac_attack_raw_std_lt_1e-6']:.3f} | {x['frac_benign_raw_std_lt_1e-6']:.3f} | {x['frac_attack_raw_max_eq_20kW']:.3f} | {x['frac_benign_raw_max_ge_20kW']:.3f} | {x['frac_empty_window']:.3f} |")
A("")
A("## (c) Inactive-mask correctness (all attacked bags in train+val, per type)")
A("")
A("Checks per attacked bag: inactive slots hold exactly the scaled zero; active slots form one contiguous run (or are empty); the active mask is a subset of the real session's active mask; mask size equals the realized window length; active-slot count matches the number of 5-min slots in [t_start', t_end') of the attacked aggregate (+/-1); "
  "type-specific edges (types without 'start' keep the real first slot, types without 'end' keep the real last slot, SCMA-1 mask == real mask); and, for unclipped bags, active power equals scale x real curve (rtol 1e-3).")
A("")
A("| SCMA | n bags | inactive==0 | single run | subset of real mask | count = realized window | count matches aggregate window | edge rules | shape exact (unclipped) | empty masks |")
A("|---|---|---|---|---|---|---|---|---|---|")
for t, x in s["c_mask"].items():
    n = x["n"]
    A(f"| {t} | {n} | {x['inactive_zero']}/{n} | {x['single_run']}/{n} | {x['subset_of_real_mask']}/{n} | {x['mask_count_eq_extras_n_new']}/{n} | {x['window_slot_count_match']}/{n} | {x['type_specific_edge_ok']}/{n} | {x['shape_exact_match_unclipped']}/{x['n_unclipped_checked']} | {x['empty']} |")
A("")
A("Empty masks (window of zero slots) occur when the reconstructed Algorithm-1 aggregate collapses the window (5-7% for SCMA-2..5, ~50% for SCMA-6). v1 produces exactly the same empty windows (same aggregates); models must handle an all-False mask.")
A("")
A("## (d) Separability probe per SCMA type (VAL macro-F1 at threshold 0.5, trained on the type's TRAIN)")
A("")
A("LogReg = StandardScaler + LogisticRegression(class_weight=balanced). CatBoost = library defaults with auto_class_weights=Balanced, seed 0. Features = the 7-dim summary. AUROC in brackets. v1 columns are computed the same way on the v1 data (train/val only) for reference.")
A("")
A("| SCMA | v2 LogReg | v2 CatBoost | v2 CatBoost no std/max | v1 LogReg | v1 CatBoost |")
A("|---|---|---|---|---|---|")
for t, x in s["d_probe"].items():
    f = lambda p, k: f"{p[k+'_macroF1']:.4f} ({p[k+'_auroc']:.3f})"
    A(f"| {t} | {f(x['v2'],'logreg')} | {f(x['v2'],'catboost')} | {f(x['v2_no_std_no_max'],'catboost')} | {f(x['v1'],'logreg')} | {f(x['v1'],'catboost')} |")
A("")
A("Oriented single-feature AUROC per type on v2 val (t_avail, std, max, mean):")
A("")
A("| SCMA | t_avail | act std | act max | act mean | e_demand |")
A("|---|---|---|---|---|---|")
for t, x in s["d_probe"].items():
    q = x["v2_feature_auroc_oriented"]
    A(f"| {t} | {q['t_avail']:.3f} | {q['act_pow_std']:.3f} | {q['act_pow_max']:.3f} | {q['act_pow_mean']:.3f} | {q['e_demand']:.3f} |")
A("")
A("Reading. Non-trivial in v2 for a summary-feature detector: SCMA-1 (CatBoost 0.81, LogReg 0.47), SCMA-2 (0.90) and SCMA-4 (0.89). SCMA-3, 5, 6 remain essentially perfectly separable (>= 0.99) because the shifted/truncated window makes t_avail collapse (AUROC 0.95-1.0) and, for SCMA-6, half the windows are empty. "
  "For SCMA-2/4 the separability is driven by t_avail (0.95) and by residual flat/clipped short windows (std AUROC 0.92/0.91), not by a whole-curve tell. All six types are still perfectly or near-perfectly separable in v1 (CatBoost >= 0.996), so v2 lowers separability for 1, 2, 4 and leaves 3, 5, 6 unchanged.")
A("")
A("## Known limitations of v2 (do not smooth over)")
A("- The frozen attacker's aggregate outputs are what they are: SCMA-2..6 collapse the window to ~7% of its length, so v2 SCMA-2..6 bags are short, often 20 kW-clipped bursts. v2 fixes the *curve* artifact, not the aggregate-level detectability.")
A("- The 20 kW cap (as specified) is below the peak of ~0.4% of benign val sessions (measured power up to ~49 kW exists in the raw data); the cap is a small residual difference between benign and attacked curves.")
A("- Train augmentation variants of one source are near-duplicates because the policy's stochastic spread (std 0.58 in C) moves E' by only a few percent for most sessions.")
A("- The `corr` metric is 1.0 by construction and is a check of the implementation, not evidence of realism.")
A("- Only tree/LogReg summary probes were run here; no sequence model was trained on v2.")
open(RESULTS_V2 / "V2_DATA_REPORT.md", "w").write("\n".join(L) + "\n")
