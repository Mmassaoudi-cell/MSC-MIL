"""Master Prompt Sec.17-18: 7-way multiclass (benign + SCMA-1..6) attack-TYPE
attribution using the frozen MSC-MIL architecture/hyperparameters
(FINAL_MODEL_CONFIG.yaml), just a 7-way softmax classifier head + CE loss
instead of the frozen model's binary sigmoid/BCE. SupCon term (beta*L_con)
is kept, reusing common.supcon_loss unmodified (it is already label-equality
based and hence multiclass-general -- see msc_mil.MSCMILMulticlass docstring).

Trains 5 seeds (0-4, matching FINAL_MODEL_CONFIG.yaml's
seeds.final_evaluation_seeds) on {train, val}, evaluates ONCE on the held-out
multiclass test_full built by multiclass_dataset.build_multiclass_dataset.
Writes:
  results/raw/multiclass/msc_mil_multiclass_seed{seed}.json  (per-seed val history + test metrics)
  results/aggregate/multiclass_results.csv                    (aggregated across seeds)
  results/raw/multiclass/confusion_matrix_seed{seed}.json     (raw 7x7 confusion matrix per seed)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

from src.candidates.common import supcon_loss
from src.candidates.msc_mil import MSCMILMulticlass
from src.candidates.multiclass_dataset import SCMA_TYPES_ALL, build_multiclass_dataset
from src.reproduction import dataset as dsmod
from src.reproduction.config import ReproductionConfig
from src.reproduction.ppo_attacker import ActorCritic

PROJECT_ROOT = dsmod.PROJECT_ROOT
CONFIG_PATH = PROJECT_ROOT / "FINAL_MODEL_CONFIG.yaml"
RAW_DIR = PROJECT_ROOT / "results" / "raw" / "multiclass"
CKPT_DIR = PROJECT_ROOT / "results" / "checkpoints" / "multiclass"
AGG_DIR = PROJECT_ROOT / "results" / "aggregate"

SEEDS = (0, 1, 2, 3, 4)
CLASS_NAMES = ["benign", "SCMA-1", "SCMA-2", "SCMA-3", "SCMA-4", "SCMA-5", "SCMA-6"]
N_CLASSES = len(CLASS_NAMES)


def load_final_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def infer_probs(model, slots, mask, static, device, batch_size=256):
    model.eval()
    out = []
    n = len(slots)
    for start in range(0, n, batch_size):
        sl = slice(start, start + batch_size)
        s = torch.from_numpy(slots[sl]).to(device)
        m = torch.from_numpy(mask[sl]).to(device)
        st = torch.from_numpy(static[sl]).to(device)
        probs, _ = model(s, m, st)
        out.append(probs.cpu().numpy())
    return np.concatenate(out)


def macro_f1_multiclass(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=list(range(N_CLASSES))))


def train_one_seed(model, ids_cfg, data, device, seed, beta, tau, max_epochs, patience):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = TensorDataset(
        torch.from_numpy(data["train"].slots), torch.from_numpy(data["train"].mask),
        torch.from_numpy(data["train"].static), torch.from_numpy(data["train"].labels).long(),
    )
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=ids_cfg["batch_size"], shuffle=True, generator=gen)
    optimizer = torch.optim.Adam(model.parameters(), lr=ids_cfg["lr"])

    val = data["val"]
    best_val_f1 = -1.0
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0
    history = []
    converged = True
    fail_reason = None

    for epoch in range(max_epochs):
        model.train()
        progress = epoch / max(max_epochs - 1, 1)
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            slots, mask, static, labels = [t.to(device) for t in batch]
            logits, _, H = model.logits_with_embedding(slots, mask, static)
            det = F.cross_entropy(logits, labels) + ids_cfg["l2_reg"] * sum((p ** 2).sum() for p in model.parameters())
            con = supcon_loss(H, labels, temperature=tau)
            loss = det + beta * con
            if not torch.isfinite(loss):
                converged = False
                fail_reason = f"non-finite loss at epoch {epoch}, batch {n_batches}"
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        if not converged:
            break

        val_probs = infer_probs(model, val.slots, val.mask, val.static, device)
        if not np.isfinite(val_probs).all():
            converged = False
            fail_reason = f"non-finite val probs at epoch {epoch}"
            break
        val_pred = val_probs.argmax(axis=1)
        val_f1 = macro_f1_multiclass(val.labels, val_pred)
        history.append({"epoch": epoch, "train_loss": epoch_loss / max(n_batches, 1), "val_macro_f1": val_f1})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break

    return history, best_state, best_epoch, best_val_f1, converged, fail_reason


def main():
    cfg_yaml = load_final_config()
    arch = cfg_yaml["architecture"]
    hp = cfg_yaml["hyperparameters"]
    train_cfg = cfg_yaml["training"]
    ids_cfg = {
        "lr": hp["learning_rate"], "l2_reg": hp["l2_reg"], "batch_size": hp["batch_size"],
        "max_epochs": train_cfg["max_epochs"], "patience": train_cfg["early_stopping_patience"],
    }
    beta = hp["supcon_beta"]
    tau = hp["supcon_temperature"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[multiclass_train] device={device}")

    repro_cfg = ReproductionConfig()
    manifest = dsmod.load_manifest()
    store = dsmod.SlotStore.get()
    ckpt = torch.load(CKPT_DIR.parent / "ppo_attacker.pt", map_location=device)
    attacker = ActorCritic(repro_cfg.attacker).to(device)
    attacker.load_state_dict(ckpt["model_state"])
    attacker.eval()

    pools = dsmod.build_split_pools(manifest, seed=0)
    test_variants = dsmod.build_test_variants(manifest, seed=0)

    t0 = time.time()
    data = build_multiclass_dataset(manifest, store, pools, test_variants, attacker, device, repro_cfg, dataset_seed=0, scma_types=SCMA_TYPES_ALL)
    print(f"[multiclass_train] dataset built in {time.time()-t0:.1f}s: "
          f"train={len(data['train'].labels)} val={len(data['val'].labels)} test_full={len(data['test_full'].labels)}")
    print(f"[multiclass_train] train label counts: {np.bincount(data['train'].labels, minlength=N_CLASSES)}")
    print(f"[multiclass_train] test_full label counts: {np.bincount(data['test_full'].labels, minlength=N_CLASSES)}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    all_seed_results = []
    for seed in SEEDS:
        out_path = RAW_DIR / f"msc_mil_multiclass_seed{seed}.json"
        if out_path.exists():
            print(f"[multiclass_train] SKIP (exists) {out_path}")
            with open(out_path) as f:
                all_seed_results.append(json.load(f))
            continue
        print(f"[multiclass_train] === seed={seed} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = MSCMILMulticlass(
            input_dim=arch["input_dim"], encoder_hidden=arch["encoder_hidden"], encoder_out=arch["encoder_out"],
            lstm_hidden_per_dir=arch["lstm_hidden_per_dir"], static_dim=arch["static_dim"],
            static_hidden=arch["static_hidden"], n_classes=N_CLASSES, dropout=arch["dropout"],
        ).to(device)
        t0 = time.time()
        history, best_state, best_epoch, best_val_f1, converged, fail_reason = train_one_seed(
            model, ids_cfg, data, device, seed, beta, tau, ids_cfg["max_epochs"], ids_cfg["patience"],
        )
        train_time_s = time.time() - t0

        result = {
            "seed": seed, "converged": converged, "fail_reason": fail_reason,
            "n_params": sum(p.numel() for p in model.parameters()), "train_time_s": train_time_s,
            "epochs_trained": len(history), "best_epoch": best_epoch, "best_val_macro_f1": best_val_f1,
        }

        if converged and best_state is not None:
            model.load_state_dict(best_state)
            torch.save({"model_state": model.state_dict(), "seed": seed}, CKPT_DIR / f"msc_mil_multiclass_seed{seed}.pt")

            test = data["test_full"]
            test_probs = infer_probs(model, test.slots, test.mask, test.static, device)
            test_pred = test_probs.argmax(axis=1)
            test_macro_f1 = macro_f1_multiclass(test.labels, test_pred)
            precision, recall, f1c, support = precision_recall_fscore_support(
                test.labels, test_pred, labels=list(range(N_CLASSES)), zero_division=0
            )
            cm = confusion_matrix(test.labels, test_pred, labels=list(range(N_CLASSES)))

            result["test_macro_f1"] = test_macro_f1
            result["test_accuracy"] = float((test_pred == test.labels).mean())
            result["per_class"] = {
                CLASS_NAMES[i]: {
                    "precision": float(precision[i]), "recall": float(recall[i]),
                    "f1": float(f1c[i]), "support": int(support[i]),
                }
                for i in range(N_CLASSES)
            }
            result["confusion_matrix"] = cm.tolist()
            result["confusion_matrix_labels"] = CLASS_NAMES
            print(f"[multiclass_train] seed={seed}: OK val_macro_f1={best_val_f1:.4f} "
                  f"test_macro_f1={test_macro_f1:.4f} epochs={len(history)} best_epoch={best_epoch} time={train_time_s:.1f}s")
        else:
            print(f"[multiclass_train] seed={seed}: FAIL ({fail_reason})")

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        all_seed_results.append(result)

    # ---------------- Aggregate ----------------
    ok_results = [r for r in all_seed_results if r.get("test_macro_f1") is not None]
    rows = []
    if ok_results:
        macro_f1_vals = np.array([r["test_macro_f1"] for r in ok_results])
        acc_vals = np.array([r["test_accuracy"] for r in ok_results])
        rows.append({"metric": "test_macro_f1", "class": "ALL", "mean": macro_f1_vals.mean(),
                     "std": macro_f1_vals.std(ddof=1) if len(macro_f1_vals) > 1 else 0.0, "n_seeds": len(macro_f1_vals)})
        rows.append({"metric": "test_accuracy", "class": "ALL", "mean": acc_vals.mean(),
                     "std": acc_vals.std(ddof=1) if len(acc_vals) > 1 else 0.0, "n_seeds": len(acc_vals)})
        for cname in CLASS_NAMES:
            for metric in ("precision", "recall", "f1"):
                vals = np.array([r["per_class"][cname][metric] for r in ok_results])
                rows.append({"metric": metric, "class": cname, "mean": vals.mean(),
                             "std": vals.std(ddof=1) if len(vals) > 1 else 0.0, "n_seeds": len(vals)})
            support_vals = np.array([r["per_class"][cname]["support"] for r in ok_results])
            rows.append({"metric": "support", "class": cname, "mean": support_vals.mean(), "std": 0.0, "n_seeds": len(support_vals)})

        # mean confusion matrix across converged seeds (rounded to nearest int for readability)
        cm_stack = np.stack([np.array(r["confusion_matrix"]) for r in ok_results])
        cm_mean = cm_stack.mean(axis=0)
        np.save(RAW_DIR / "confusion_matrix_mean.npy", cm_mean)
        with open(RAW_DIR / "confusion_matrix_mean.json", "w") as f:
            json.dump({"labels": CLASS_NAMES, "mean_confusion_matrix": cm_mean.tolist(),
                       "n_seeds": len(ok_results)}, f, indent=2)

    import pandas as pd
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_path = AGG_DIR / "multiclass_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[multiclass_train] wrote {out_path} ({len(out_df)} rows)")
    print(f"[multiclass_train] {len(ok_results)}/{len(SEEDS)} seeds converged.")
    if ok_results:
        print(f"[multiclass_train] mean test_macro_f1 = {np.mean([r['test_macro_f1'] for r in ok_results]):.4f} "
              f"+/- {np.std([r['test_macro_f1'] for r in ok_results], ddof=1) if len(ok_results) > 1 else 0.0:.4f}")
    print("[multiclass_train] done.")


if __name__ == "__main__":
    main()
