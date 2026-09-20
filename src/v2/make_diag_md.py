"""Render results/v2/DIAGNOSIS_v1.md from results/v2/diagnosis_v1.json (numbers only come from the run)."""
import json
from src.v2.common import RESULTS_V2

d = json.load(open(RESULTS_V2 / "diagnosis_v1.json"))
L = []
A = L.append
A("# DIAGNOSIS_v1: is the flat-rate attack artifact the cause of the tree models' results?")
A("")
A("Scope: SCMA-1, v1 data, TRAIN fit / VAL evaluation only (the test split was never built or read; `src/v2/common.py::build_v1_train_val`). "
  f"CatBoost config = tuned config from `results/aggregate/benchmark_val_tuning_classical_tree.json`: `{d['catboost_params']}` (auto_class_weights=Balanced). "
  f"Train n={d['n_train']} (pos rate {d['train_pos_rate']:.3f}); val n={d['n_val']} (pos rate {d['val_pos_rate']:.3f}). "
  "Features are exactly v1's 7-dim summary vector (`data_utils.flatten_features` semantics, in train-scaled units; note its 'start' slot is the connection-hour proxy, as in v1). "
  "Code: `src/v2/diagnose_v1.py`; raw numbers: `results/v2/diagnosis_v1.json`. Model-level numbers are mean over 5 CatBoost seeds unless stated.")
A("")
A("## 0. What the flat curve looks like on val")
p = d["val_feature_percentiles(0,5,50,95,100)"]
A(f"- Attacked val bags with active-power std < 1e-6: **{p['attack_frac_std_lt_1e-6']*100:.1f}%**; benign val bags with std < 1e-6: {p['benign_frac_std_lt_1e-6']*100:.2f}% "
  f"(std < 1e-3: attack {p['attack_frac_std_lt_1e-3']*100:.1f}%, benign {p['benign_frac_std_lt_1e-3']*100:.2f}%).")
A("- For attacked bags active max == active mean exactly (flat curve); for benign bags max is typically far above the mean.")
A("")
A("## 1. Feature importance (full 7-feature CatBoost)")
A("")
A("| feature | native PredictionValuesChange (%, mean of 5 seeds) | permutation importance on val (macro-F1 drop, seed 0, 10 repeats, +/- std) |")
A("|---|---|---|")
im = d["importance"]
for k in im["native_PredictionValuesChange_mean_over_5seeds"]:
    A(f"| {k} | {im['native_PredictionValuesChange_mean_over_5seeds'][k]:.2f} | {im['permutation_macroF1_drop_val_seed0'][k]:+.4f} +/- {im['permutation_std'][k]:.4f} |")
A("")
A(f"(seed-0 full-model val macro-F1 = {im['seed0_val_macroF1']:.4f})")
A("")
A("## 2. Single-feature separability (AUROC on val)")
A("")
A("| feature | AUROC (attack=positive) | oriented AUROC (max(a,1-a)) | attack values are |")
A("|---|---|---|---|")
for k, v in d["single_feature_auroc_val"].items():
    A(f"| {k} | {v['auroc_raw']:.4f} | {v['auroc_oriented']:.4f} | {'higher' if v['attack_higher'] else 'lower'} |")
A("")
A("## 3. Retraining without the shortcut features (clean VAL macro-F1, mean +/- std over 5 seeds)")
A("")
A("| feature set | clean val macro-F1 | recall | false-alarm rate |")
A("|---|---|---|---|")
for v, r in d["noise_val"].items():
    x = r["0.0"]
    A(f"| {v} | {x['macro_f1_mean']:.4f} +/- {x['macro_f1_std']:.4f} | {x['recall_mean']:.3f} | {x['far_mean']:.4f} |")
A("")
A("## 4. Gaussian noise on VAL (same protocol as `robustness.py::perturb_noise`; imported, not re-implemented)")
A(f"Noise sigma = level x TRAIN std (power {d['noise_refs']['power_std']:.4f}, energy {d['noise_refs']['energy_std']:.4f}, both from scaled train tensors), added to active-slot power and to the energy static feature; features recomputed from the noisy tensors; RNG seed = `int(level*1000)*100 + seed` as in v1 with model seed = noise seed. Models are trained on clean train data. Cells: macro-F1 mean +/- std (recall / FAR in brackets).")
A("")
lv = ["0.0", "0.05", "0.15", "0.3"]
A("| feature set | clean | 5% | 15% | 30% |")
A("|---|---|---|---|---|")
for v, r in d["noise_val"].items():
    cells = [f"{r[l]['macro_f1_mean']:.4f} +/- {r[l]['macro_f1_std']:.4f} ({r[l]['recall_mean']:.2f} / {r[l]['far_mean']:.3f})" for l in lv]
    A(f"| {v} | " + " | ".join(cells) + " |")
A("")
A("Macro-F1 = 0.4737 is the value of an all-benign predictor at 10% prevalence (recall 0).")
A("")
A("## Conclusion")
A("@@CONCLUSION@@")
open(RESULTS_V2 / "DIAGNOSIS_v1.md", "w").write("\n".join(L) + "\n")
