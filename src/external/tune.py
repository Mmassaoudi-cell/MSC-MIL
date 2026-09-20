"""Equal-budget Optuna tuning (TPE, seed 0, 12 trials/model) on VALIDATION macro-F1@0.5. TEST is never loaded.
python -m src.external.tune --dataset cic2017 [--models a,b]
Outputs: results/external/tuning_<ds>/<model>.json (trials + 3-seed val), results/external/tuned_configs_<ds>.json"""
import argparse, json, time
import numpy as np, optuna, torch
from src.external.data import load
from src.external.models import ALL_MODELS, TREE, fit_model, macro_f1
from src.external.prep_common import RES

N_TRIALS, SEEDS = 12, (0, 1, 2)


def space(name, t):
    if name == "logreg": return {"C": t.suggest_float("C", 1e-3, 1e2, log=True)}
    if name == "random_forest":
        return {"n_estimators": t.suggest_int("n_estimators", 100, 400, step=100),
                "max_depth": t.suggest_categorical("max_depth", [6, 12, 24, None]),
                "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 10)}
    if name == "xgboost":
        return {"n_estimators": t.suggest_int("n_estimators", 100, 500, step=100), "max_depth": t.suggest_int("max_depth", 3, 10),
                "learning_rate": t.suggest_float("learning_rate", 0.02, 0.3, log=True),
                "subsample": t.suggest_float("subsample", 0.6, 1.0), "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0)}
    if name == "lightgbm":
        return {"n_estimators": t.suggest_int("n_estimators", 100, 500, step=100), "num_leaves": t.suggest_int("num_leaves", 8, 128, log=True),
                "learning_rate": t.suggest_float("learning_rate", 0.02, 0.3, log=True),
                "subsample": t.suggest_float("subsample", 0.6, 1.0), "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0)}
    if name == "catboost":
        return {"iterations": t.suggest_int("iterations", 100, 500, step=100), "depth": t.suggest_int("depth", 4, 8),
                "learning_rate": t.suggest_float("learning_rate", 0.02, 0.3, log=True)}
    p = {"lr": t.suggest_float("lr", 3e-4, 3e-3, log=True), "dropout": t.suggest_float("dropout", 0.1, 0.4),
         "hidden": t.suggest_categorical("hidden", [32, 64, 128]), "l2": t.suggest_float("l2", 1e-6, 1e-3, log=True)}
    if name == "transformer_mil": p["n_layers"] = t.suggest_int("n_layers", 1, 3)
    if name == "msc_mil":
        p["beta"] = t.suggest_float("beta", 0.05, 2.0, log=True); p["temperature"] = t.suggest_float("temperature", 0.05, 0.5)
    return p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--models", default=",".join(ALL_MODELS))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = load(a.dataset, ("train", "val")); tr, va = D["train"], D["val"]
    from src.external.data import summary_features
    Str = summary_features(tr["X"])
    out_dir = RES / f"tuning_{a.dataset}"; out_dir.mkdir(parents=True, exist_ok=True)
    for name in a.models.split(","):
        f = out_dir / f"{name}.json"
        if f.exists(): print("skip", name); continue
        t0 = time.time()
        def obj(trial):
            p = space(name, trial); t1 = time.time()
            fitted, info = fit_model(name, p, tr, va, 0, dev, Str)
            v = info.get("val_macro_f1", 0.0) if fitted is not None else 0.0
            trial.set_user_attr("secs", time.time() - t1); trial.set_user_attr("epochs", info.get("epochs"))
            print(f"[{a.dataset}/{name}] trial {trial.number} val_macroF1={v:.4f} {p} ({time.time()-t1:.0f}s)", flush=True)
            return v
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        st.optimize(obj, n_trials=N_TRIALS)
        best = {k: (None if v is None else v) for k, v in st.best_params.items()}
        # optuna returns categorical None fine; reload real types
        seeds = []
        for s in SEEDS:
            fitted, info = fit_model(name, best, tr, va, s, dev, Str)
            seeds.append(info.get("val_macro_f1", float("nan")))
        rec = {"model": name, "dataset": a.dataset, "best_params": best, "tuning_best_val_macro_f1_seed0": st.best_value,
               "val_macro_f1_seeds": seeds, "val_macro_f1_mean": float(np.mean(seeds)), "val_macro_f1_std": float(np.std(seeds)),
               "trials": [{"n": t.number, "params": t.params, "val": t.value, "secs": t.user_attrs.get("secs")} for t in st.trials],
               "total_secs": time.time() - t0}
        f.write_text(json.dumps(rec, indent=1))
        print(f"[{a.dataset}/{name}] DONE best={st.best_value:.4f} 3-seed mean={rec['val_macro_f1_mean']:.4f} ({rec['total_secs']:.0f}s)", flush=True)
    cfg = {n: json.loads((out_dir / f"{n}.json").read_text()) for n in ALL_MODELS if (out_dir / f"{n}.json").exists()}
    (RES / f"tuned_configs_{a.dataset}.json").write_text(json.dumps(
        {n: {"params": c["best_params"], "val_macro_f1_mean_3seeds": c["val_macro_f1_mean"], "val_macro_f1_std": c["val_macro_f1_std"]}
         for n, c in cfg.items()}, indent=1))


if __name__ == "__main__":
    main()
