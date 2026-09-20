"""Benchmarks 1-7 (Part A): classical (LogReg, SVM-RBF, kNN) + tree/boosting
(RandomForest, XGBoost, LightGBM, CatBoost) on the flattened 7-dim per-session
summary-vector features (data_utils.flatten_features), SCMA-1 only.

Light validation-only tuning: a small, documented grid (3-5 sensible configs
per model, not a full Optuna study -- these are reference points, not a
research contribution in themselves, per the task brief). Selection metric:
val macro-F1 @ threshold 0.5 (src.reproduction.metrics.macro_f1_score),
fit on TRAIN, scored on VAL. TEST is never touched here.

Output: results/aggregate/benchmark_val_tuning_classical_tree.json
  {model_name: {"best_params": {...}, "best_val_macro_f1": float, "grid_results": [...]}}
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.benchmarks.data_utils import PROJECT_ROOT, flatten_features, load_scma_data
from src.reproduction.metrics import macro_f1_score

AGG_DIR = PROJECT_ROOT / "results" / "aggregate"

GRIDS = {
    "logreg": [{"C": c} for c in [0.01, 0.1, 1.0, 10.0]],
    "svm_rbf": [
        {"C": 1.0, "gamma": "scale"}, {"C": 10.0, "gamma": "scale"},
        {"C": 1.0, "gamma": 0.1}, {"C": 10.0, "gamma": 0.01},
    ],
    "knn": [{"n_neighbors": k} for k in [3, 5, 7, 11, 15]],
    "random_forest": [
        {"n_estimators": 100, "max_depth": None}, {"n_estimators": 200, "max_depth": 10},
        {"n_estimators": 300, "max_depth": None}, {"n_estimators": 200, "max_depth": 20},
    ],
    "xgboost": [
        {"max_depth": 3, "n_estimators": 100, "learning_rate": 0.1},
        {"max_depth": 5, "n_estimators": 200, "learning_rate": 0.1},
        {"max_depth": 3, "n_estimators": 200, "learning_rate": 0.05},
        {"max_depth": 5, "n_estimators": 100, "learning_rate": 0.2},
    ],
    "lightgbm": [
        {"max_depth": 3, "n_estimators": 100, "learning_rate": 0.1},
        {"max_depth": 5, "n_estimators": 200, "learning_rate": 0.1},
        {"max_depth": -1, "n_estimators": 200, "learning_rate": 0.05},
        {"max_depth": 5, "n_estimators": 100, "learning_rate": 0.2},
    ],
    "catboost": [
        {"depth": 4, "iterations": 200, "learning_rate": 0.1},
        {"depth": 6, "iterations": 300, "learning_rate": 0.1},
        {"depth": 4, "iterations": 300, "learning_rate": 0.05},
        {"depth": 6, "iterations": 200, "learning_rate": 0.2},
    ],
}

RANDOM_MODELS = {"svm_rbf", "random_forest", "xgboost", "lightgbm", "catboost"}  # has a seed-sensitive fit


def build_estimator(name: str, params: dict, seed: int = 0):
    if name == "logreg":
        return LogisticRegression(class_weight="balanced", max_iter=5000, **params)
    if name == "svm_rbf":
        return SVC(probability=True, class_weight="balanced", random_state=seed, **params)
    if name == "knn":
        return KNeighborsClassifier(**params)
    if name == "random_forest":
        return RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1, **params)
    if name == "xgboost":
        return XGBClassifier(
            eval_metric="logloss", random_state=seed, n_jobs=-1,
            tree_method="hist", **params,
        )
    if name == "lightgbm":
        return LGBMClassifier(random_state=seed, n_jobs=-1, class_weight="balanced", verbosity=-1, **params)
    if name == "catboost":
        return CatBoostClassifier(random_seed=seed, verbose=0, auto_class_weights="Balanced", **params)
    raise ValueError(name)


def _scale_pos_weight(y):
    pos = y.sum()
    neg = len(y) - pos
    return float(neg / max(pos, 1))


def fit_and_predict(name, params, Xtr, ytr, Xte, seed=0):
    est = build_estimator(name, params, seed=seed)
    if name == "xgboost":
        est.set_params(scale_pos_weight=_scale_pos_weight(ytr))
    est.fit(Xtr, ytr)
    proba = est.predict_proba(Xte)[:, 1]
    return est, proba


def tune_model(name, Xtr, ytr, Xval, yval):
    results = []
    for params in GRIDS[name]:
        t0 = time.time()
        _, proba = fit_and_predict(name, params, Xtr, ytr, Xval, seed=0)
        pred = (proba >= 0.5).astype(int)
        f1m = macro_f1_score(yval, pred)
        results.append({"params": params, "val_macro_f1": f1m, "fit_time_s": time.time() - t0})
    results.sort(key=lambda r: r["val_macro_f1"], reverse=True)
    return results[0]["params"], results[0]["val_macro_f1"], results


def main():
    scma_type = 1
    data = load_scma_data(scma_type)
    Xtr = flatten_features(data["train"]); ytr = data["train"].labels
    Xval = flatten_features(data["val"]); yval = data["val"].labels
    print(f"[classical_tree] SCMA-{scma_type} train={Xtr.shape} val={Xval.shape} pos_rate(train)={ytr.mean():.3f}")

    out = {}
    for name in GRIDS:
        t0 = time.time()
        best_params, best_f1, grid_results = tune_model(name, Xtr, ytr, Xval, yval)
        dt = time.time() - t0
        out[name] = {
            "best_params": best_params, "best_val_macro_f1": best_f1,
            "grid_results": grid_results, "has_randomness": name in RANDOM_MODELS,
            "tuning_wall_time_s": dt,
        }
        print(f"[classical_tree] {name}: best_val_macro_f1={best_f1:.4f} params={best_params} ({dt:.1f}s)")

    AGG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = AGG_DIR / "benchmark_val_tuning_classical_tree.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[classical_tree] wrote {out_path}")

    # Sanity check: confirm SCMA-3/6 triviality still holds for 2-3 of these
    # models (task brief) -- use the 3 cheapest (logreg, knn, random_forest).
    sanity = {}
    for scma_type_chk in (3, 6):
        d2 = load_scma_data(scma_type_chk)
        Xtr2 = flatten_features(d2["train"]); ytr2 = d2["train"].labels
        Xval2 = flatten_features(d2["val"]); yval2 = d2["val"].labels
        sanity[scma_type_chk] = {}
        for name in ("logreg", "knn", "random_forest"):
            params = out[name]["best_params"]
            _, proba = fit_and_predict(name, params, Xtr2, ytr2, Xval2, seed=0)
            pred = (proba >= 0.5).astype(int)
            f1m = macro_f1_score(yval2, pred)
            sanity[scma_type_chk][name] = f1m
            print(f"[classical_tree] SANITY SCMA-{scma_type_chk} {name}: val_macro_f1={f1m:.4f}")
    with open(AGG_DIR / "benchmark_scma36_sanity_classical.json", "w") as f:
        json.dump(sanity, f, indent=2)


if __name__ == "__main__":
    main()
