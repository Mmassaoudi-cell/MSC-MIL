import json, glob
import numpy as np, pandas as pd
from src.v2 import baselines_lib as L

OUT = L.OUT
ROOT = L.PROJECT_ROOT / "results" / "v2"
SPACES = {
 "logreg": {"C": "log[1e-3,1e2]"},
 "knn": {"n_neighbors": "int[1,50]", "weights": "{uniform,distance}", "p": "{1,2}"},
 "svm_rbf": {"C": "log[1e-2,1e3]", "gamma": "log[1e-3,10]", "note": f"stratified TRAIN subsample capped at {L.SVM_TRAIN_CAP} for fit time; probability=True, threshold 0.5 on predict_proba"},
 "random_forest": {"n_estimators": "int[100,500] step 50", "max_depth": "int[3,30]", "min_samples_leaf": "int[1,20]", "max_features": "{sqrt,log2,None}"},
 "xgboost": {"n_estimators": "int[50,500] step 50", "max_depth": "int[2,10]", "learning_rate": "log[0.01,0.3]", "subsample": "[0.5,1]", "colsample_bytree": "[0.5,1]", "reg_lambda": "log[0.01,10]", "min_child_weight": "int[1,10]"},
 "lightgbm": {"n_estimators": "int[50,500] step 50", "num_leaves": "log int[4,128]", "learning_rate": "log[0.01,0.3]", "min_child_samples": "log int[5,100]", "subsample": "[0.5,1] (freq 1)", "colsample_bytree": "[0.5,1]", "reg_lambda": "log[0.01,10]"},
 "catboost": {"iterations": "int[100,600] step 50", "depth": "int[3,8]", "learning_rate": "log[0.01,0.3]", "l2_leaf_reg": "log[1,20]"},
}
DEEP_SPACE = {"lr": "log[3e-4,3e-3]", "dropout": "[0.1,0.4]", "hidden": "{32,64,128}", "l2": "log[1e-6,1e-3]"}
for m in ["mlp", "cnn1d", "plain_lstm_mil", "transformer_mil", "source_bilstm_attn"]:
    SPACES[m] = dict(DEEP_SPACE)
SPACES["msc_mil"] = {"beta": "log[0.05,2]", "tau": "[0.05,0.5]", "dropout": "[0.1,0.4]", "lr": "log[3e-4,3e-3]",
                     "fixed": "hidden 64 (frozen MSC-MIL architecture), l2 1e-4, pool scales (1,3,12)"}
HID = {"mlp": "hidden layers (4h,h)", "cnn1d": "conv channels (h/2,h,h)", "plain_lstm_mil": "encoder hidden=out=h, BiLSTM h/dir",
       "transformer_mil": "d_model=encoder=h, 2 layers, 4 heads, ff=2h", "source_bilstm_attn": "encoder hidden=out=h, BiLSTM h/dir"}
NAMES = {"logreg": "Logistic Regression", "knn": "k-NN", "svm_rbf": "RBF-SVM", "random_forest": "Random Forest", "xgboost": "XGBoost",
         "lightgbm": "LightGBM", "catboost": "CatBoost", "mlp": "Flatten-MLP", "cnn1d": "1D-CNN", "plain_lstm_mil": "Plain-LSTM-MIL",
         "transformer_mil": "Transformer-MIL", "source_bilstm_attn": "Source BiLSTM+attn MIL", "msc_mil": "MSC-MIL"}
tags = [t for t in L.ALL_MODELS] + [t + "_aug" for t in ("random_forest", "xgboost", "catboost", "msc_mil", "source_bilstm_attn")]
nm = lambda t: NAMES[t.replace("_aug", "")] + (" +noise-aug" if t.endswith("_aug") else "")

bests = {t: json.loads((OUT / f"best_{t}.json").read_text()) for t in tags if (OUT / f"best_{t}.json").exists()}
df = pd.read_csv(OUT / "final_runs.csv")
ms = lambda s: f"{s.mean():.4f} +/- {s.std(ddof=1):.4f}" if len(s) > 1 else f"{s.mean():.4f}"

frozen = {"protocol": {"sampler": "Optuna TPE seed 0", "n_trials": 15, "objective": "val macro-F1@0.5 (SCMA-1, model seed 0)", "tuning_type": "SCMA-1 (v2)",
                       "deep": "max 30 epochs, patience 8, batch 32, Adam, BCE + manual L2, best-val-epoch checkpoint",
                       "tree_features": "summary_features_v2 (7-dim)", "reuse": "SCMA-1 tuned config reused for SCMA-2 and SCMA-4",
                       "seeds_for_validation_runs": [0, 1, 2], "threshold": 0.5,
                       "noise_calibration": "results/v2/noise_calibration.json (sigma_r)",
                       "noise_aug": "rel std ~ U(0,4 sigma_r) per bag; deep: redrawn each batch; trees: clean + 3 noisy copies"},
          "models": {}}
for t, b in bests.items():
    base = b["base_model"]
    frozen["models"][t] = dict(display=nm(t), base_model=base, noise_augmented=b["aug"], search_space=SPACES[base] | ({"hidden_meaning": HID[base]} if base in HID else {}),
                               best_params=b["best_params"], best_tuning_val_macro_f1_seed0=b["best_value"], n_trials=b["n_trials"],
                               code="src/v2/baselines_lib.py (build_tree / build_deep / train_deep / tree_train_set)")
(ROOT / "frozen_baseline_configs.json").write_text(json.dumps(frozen, indent=1))

L_ = []
w = L_.append
w("# BASELINES_TUNING_REPORT (v2, equal budget)\n")
w("All numbers: v2 TRAIN -> VAL only (test splits never loaded; `load_trainval` raises on any other split). Code: `src/v2/baselines_lib.py`, `run_tuning.py`, `run_final.py`, `make_baselines_report.py`. Raw: `results/v2/baselines_tuning/` (per-model `optuna_*.csv`, `best_*.json`, `final_runs.csv`, `logs/`). Frozen configs: `results/v2/frozen_baseline_configs.json`.\n")
w("## Protocol\n- Every model: Optuna TPE (seed 0), **15 trials**, objective = val macro-F1@0.5 on v2 SCMA-1, model seed 0. Trees/classical use `summary_features_v2`; deep models use the 288-slot bags, max 30 epochs, patience 8, batch 32.\n- Deep search space: lr log[3e-4,3e-3], dropout [0.1,0.4], hidden {32,64,128}, l2 log[1e-6,1e-3]. MSC-MIL: beta, temperature, dropout, lr as in Stage 3 (hidden 64 and l2 1e-4 fixed at frozen values, i.e. MSC-MIL had 4 searched parameters like the others, but not the same ones).\n- Then 3 seeds (0,1,2) on SCMA-1, SCMA-2, SCMA-4; SCMA-1's tuned config is reused unchanged for SCMA-2 and SCMA-4 (not re-tuned there).\n- Caveat: deep models restore the best-val-epoch checkpoint (early stopping on val macro-F1, as in all earlier stages), so their val scores carry a small optimistic selection bias that trees (fixed n_estimators) do not have. Tuning score is also a val score (selection on val); the final test evaluation is the unbiased one.\n- SVM: TRAIN subsample capped at 4000 (stratified) for fit time. Transformer-MIL uses a safe wrapper for bags with an all-False mask (v1 class would give NaN there).\n- Noise (VAL only): legacy additive 5/15/30% of train std; calibrated multiplicative 1x/2x/4x sigma_r with sigma_r = 0.226% (see NOISE_CALIBRATION.md). Noise seed = model seed.\n")
w("## Tuned parameters (SCMA-1)\n\n| model | tuning best val macro-F1 (seed 0) | best params |\n|---|---|---|")
for t, b in bests.items():
    p = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in b["best_params"].items()}
    w(f"| {nm(t)} | {b['best_value']:.4f} | {p} |")
w("\n## Clean validation metrics (mean +/- std over 3 seeds)\n")
for t_ in (1, 2, 4):
    w(f"\n### SCMA-{t_}{' (tuned here)' if t_ == 1 else ' (SCMA-1 config reused)'}\n\n| model | macro-F1 | recall | FAR |\n|---|---|---|---|")
    d = df[(df.scma == t_) & (df.condition == "clean")]
    rows = []
    for t in tags:
        x = d[d.model == t]
        if len(x): rows.append((x.macro_f1.mean(), f"| {nm(t)} | {ms(x.macro_f1)} | {ms(x.recall)} | {ms(x.false_alarm_rate)} |"))
    for _, r in sorted(rows, key=lambda z: -z[0]): w(r)
conds = ["clean", "legacy_5", "legacy_15", "legacy_30", "cal_1x", "cal_2x", "cal_4x"]
for t_ in (1, 2, 4):
    w(f"\n## Validation macro-F1 under noise, SCMA-{t_} (mean over 3 seeds; delta = cal_4x minus clean)\n\n| model | " + " | ".join(conds) + " | delta cal_4x |\n|" + "---|" * (len(conds) + 2))
    d = df[df.scma == t_]
    for t in tags:
        x = d[d.model == t]
        if not len(x): continue
        v = [x[x.condition == c].macro_f1.mean() for c in conds]
        w(f"| {nm(t)} | " + " | ".join(f"{a:.4f}" for a in v) + f" | {v[-1]-v[0]:+.4f} |")
w("\n## Noise-augmented vs clean-trained (SCMA-1, macro-F1 mean over 3 seeds)\n\n| model | variant | clean | legacy_15 | legacy_30 | cal_1x | cal_4x |\n|---|---|---|---|---|---|---|")
d = df[df.scma == 1]
for b in ("random_forest", "xgboost", "catboost", "msc_mil", "source_bilstm_attn"):
    for t in (b, b + "_aug"):
        x = d[d.model == t]
        if len(x): w(f"| {NAMES[b]} | {'noise-aug' if t.endswith('_aug') else 'clean-trained'} | " + " | ".join(f"{x[x.condition == c].macro_f1.mean():.4f}" for c in ("clean", "legacy_15", "legacy_30", "cal_1x", "cal_4x")) + " |")
(ROOT / "BASELINES_TUNING_REPORT.md").write_text("\n".join(L_) + "\n\n<!-- interpretation appended below -->\n")
print("\n".join(L_))
