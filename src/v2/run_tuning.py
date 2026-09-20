"""Equal-budget tuning: Optuna TPE, 15 trials, seed 0, objective = val macro-F1@0.5, SCMA-1, v2 train/val only.
usage: python -m src.v2.run_tuning --models a,b,c [--aug]
"""
import argparse, json, sys, time
import numpy as np, optuna
from src.v2 import baselines_lib as L
from src.reproduction.metrics import macro_f1_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
N_TRIALS, SEED, SCMA = 15, 0, 1


def run(name, aug):
    tag = name + ("_aug" if aug else "")
    best_json = L.OUT / f"best_{tag}.json"
    if best_json.exists():
        print("skip", tag); return
    data = L.load_trainval(SCMA)
    dev = L.get_device()
    if name in L.TREE_MODELS:
        Xtr, ytr = L.tree_train_set(data, aug, SEED); Xva = L.feats(data["val"]); yva = data["val"].labels
    rows = []

    def objective(trial):
        t0 = time.time()
        if name in L.TREE_MODELS:
            p = L.suggest_tree(name, trial)
            est = L.fit_tree(name, p, Xtr, ytr, SEED)
            f1 = macro_f1_score(yva, (est.predict_proba(Xva)[:, 1] >= 0.5).astype(int)); info = {}
        else:
            p = L.suggest_deep(name, trial)
            model, info = L.train_deep(name, p, data, SEED, dev, aug=aug)
            f1 = info["best_val_macro_f1"] if model is not None else 0.0
        dt = time.time() - t0
        trial.set_user_attr("time_s", dt)
        print(f"[{tag}] trial {trial.number} f1={f1:.4f} {dt:.0f}s {p} {info.get('epochs','')}", flush=True)
        return f1

    def cb(study, trial):
        study.trials_dataframe().to_csv(L.OUT / f"optuna_{tag}.csv", index=False)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[cb])
    best = dict(model=tag, base_model=name, aug=aug, scma_type=SCMA, n_trials=N_TRIALS, sampler="TPE(seed=0)",
                objective="val macro-F1@0.5 (seed 0)", best_value=study.best_value, best_params=study.best_params,
                best_trial=study.best_trial.number, tuning_time_s=time.time() - t0)
    if name == "svm_rbf": best["svm_train_cap"] = L.SVM_TRAIN_CAP
    best_json.write_text(json.dumps(best, indent=1))
    print("DONE", tag, best["best_value"], best["best_params"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--models"); ap.add_argument("--aug", action="store_true")
    a = ap.parse_args()
    for m in a.models.split(","):
        run(m, a.aug)
