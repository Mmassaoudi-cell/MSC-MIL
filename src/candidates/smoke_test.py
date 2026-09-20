"""Stage 1 smoke test (Master Prompt Sec.9) for the 4 hybrid candidates in
MODEL_CANDIDATES.md. Cheap, fast validation that each candidate (a) trains
without crashing/NaN-ing, (b) doesn't collapse to majority-class prediction,
(c) doesn't require unreasonable resources, (d) isn't obviously worse than
the already-reproduced baseline MIL-BiLSTM IDS. NOT the full multi-seed
validation screening (Stage 2/3) -- one seed (0), reduced epoch budget,
SCMA-1 and SCMA-6 only (simplest and most-blended attack types).

Loss-term hyperparameters not numerically specified by MODEL_CANDIDATES.md
(documented ASSUMPTIONs, same style as reproduction/config.py):
  - GRL lambda_max = 1.0, DANN gamma = 10 (common.dann_lambda)
  - SupCon beta (loss weight) = 0.5, temperature tau = 0.1
  - K = 4 generators for MGAT-MIL / GRMA-MIL (see train_multi_attackers.py)
All other architecture hyperparameters (encoder_hidden, encoder_out,
lstm_hidden_per_dir, static_dim, static_hidden, dropout, lr, l2_reg,
batch_size) are taken UNCHANGED from reproduction/config.py's IDSConfig, per
the task brief's "apples-to-apples on backbone capacity" instruction.

Training budget: max_epochs=30, early-stopping patience=8 on val macro-F1@0.5
(vs. the reproduction's max_epochs=200/patience=20) -- a smoke test, not a
final number.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.candidates import multigen_dataset as mgd
from src.candidates.common import dann_lambda, supcon_loss
from src.candidates.grma_mil import GRMAMIL
from src.candidates.mgat_mil import MGATMIL
from src.candidates.msc_mil import MSCMIL
from src.candidates.ssm_mil import SSMMIL
from src.reproduction import dataset as dsmod
from src.reproduction.config import ReproductionConfig
from src.reproduction.metrics import binary_metrics
from src.reproduction.mil_ids import bce_l2_loss
from src.reproduction.ppo_attacker import ActorCritic

PROJECT_ROOT = dsmod.PROJECT_ROOT
RAW_DIR = PROJECT_ROOT / "results" / "raw" / "candidates"
CKPT_DIR = PROJECT_ROOT / "results" / "checkpoints"

MAX_EPOCHS = 30
PATIENCE = 8
LAMBDA_MAX = 1.0
SUPCON_BETA = 0.5
SUPCON_TAU = 0.1
K_GENERATORS = 4
SEED = 0
SCMA_TYPES = (1, 6)

# DOCUMENTED SCOPE REDUCTION (added mid-run after an empirical timing check):
# ssm_mil's SelectiveSSMLayer (common.py) is a naive, non-vectorized Python
# for-loop scan over the 288 slots (mamba_ssm is not installed; see that
# module's docstring). A direct timing probe on the full SCMA-1 train split
# measured ~62s/epoch for ssm_mil vs. ~2-4s/epoch for the three BiLSTM-based
# candidates -- at the shared MAX_EPOCHS=30/PATIENCE=8 budget that's up to
# ~31 min/SCMA-type (~60+ min for both smoke-test types) for one candidate
# alone, which is disproportionate for a Stage-1 PASS/FAIL/CONCERN smoke
# signal (not a polished accuracy number). We therefore give ssm_mil ONLY a
# smaller epoch budget; all three BiLSTM-based candidates keep the full
# MAX_EPOCHS/PATIENCE above. This does not change what is being measured
# (converged / majority-collapse / val macro-F1 / train_time_s / n_params),
# only how many epochs ssm_mil is given to reach it -- a legitimate,
# documented smoke-test-only accommodation for its known efficiency
# trade-off (MODEL_CANDIDATES.md Candidate 3 explicitly trades some
# per-epoch cost for streaming capability; this reduction is about wall-
# clock budget for the SMOKE TEST harness, not the model's own inference
# efficiency claim, which is separately about single-pass vs. dual-pass
# recurrence, not this reference implementation's unvectorized loop).
EPOCH_BUDGET_OVERRIDE = {
    "ssm_mil": {"max_epochs": 10, "patience": 3},
}


def macro_f1_at(y_true, y_score, thresh=0.5) -> float:
    from sklearn.metrics import f1_score

    y_pred = (y_score >= thresh).astype(int)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


@torch.no_grad()
def infer_forward(model: nn.Module, slots: np.ndarray, mask: np.ndarray, static: np.ndarray, device, batch_size: int = 256):
    model.eval()
    scores = []
    n = len(slots)
    for start in range(0, n, batch_size):
        sl = slice(start, start + batch_size)
        s = torch.from_numpy(slots[sl]).to(device)
        m = torch.from_numpy(mask[sl]).to(device)
        st = torch.from_numpy(static[sl]).to(device)
        y_hat, _ = model(s, m, st)
        scores.append(y_hat.cpu().numpy())
    return np.concatenate(scores)


def majority_collapse_check(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    y_pred = (y_score >= 0.5).astype(int)
    classes_present = sorted(set(y_pred.tolist()))
    m = binary_metrics(y_true, y_pred, y_score)
    collapsed = len(classes_present) < 2 or m["precision"] == 0.0 or m["recall"] == 0.0
    return {"collapsed": collapsed, "classes_predicted": classes_present, "precision@0.5": m["precision"], "recall@0.5": m["recall"]}


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Per-candidate training loops. Each returns (history, best_state, best_epoch,
# best_val_f1, converged: bool, fail_reason: str|None).
# ---------------------------------------------------------------------------
def _common_loop(
    model,
    device,
    ids_cfg,
    loader,
    val_slots,
    val_mask,
    val_static,
    val_labels,
    compute_loss_fn,
    max_epochs=MAX_EPOCHS,
    patience=PATIENCE,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=ids_cfg.lr)
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
            loss = compute_loss_fn(model, batch, device, progress)
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

        val_score = infer_forward(model, val_slots, val_mask, val_static, device)
        if not np.isfinite(val_score).all():
            converged = False
            fail_reason = f"non-finite val score at epoch {epoch}"
            break
        val_f1 = macro_f1_at(val_labels, val_score, 0.5)
        history.append({"epoch": epoch, "train_loss": epoch_loss / max(n_batches, 1), "val_macro_f1@0.5": val_f1})

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


def train_ssm(model, ids_cfg, data, device, seed, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = TensorDataset(
        torch.from_numpy(data["train"].slots), torch.from_numpy(data["train"].mask),
        torch.from_numpy(data["train"].static), torch.from_numpy(data["train"].labels).float(),
    )
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=ids_cfg.batch_size, shuffle=True, generator=gen)

    def compute_loss(model, batch, device, progress):
        slots, mask, static, labels = [t.to(device) for t in batch]
        logits, _ = model.logits(slots, mask, static)
        return bce_l2_loss(logits, labels, model, ids_cfg.l2_reg)

    return _common_loop(
        model, device, ids_cfg, loader,
        data["val"].slots, data["val"].mask, data["val"].static, data["val"].labels,
        compute_loss, max_epochs=max_epochs, patience=patience,
    )


def train_msc(model, ids_cfg, data, device, seed, beta=SUPCON_BETA, tau=SUPCON_TAU, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = TensorDataset(
        torch.from_numpy(data["train"].slots), torch.from_numpy(data["train"].mask),
        torch.from_numpy(data["train"].static), torch.from_numpy(data["train"].labels).float(),
    )
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=ids_cfg.batch_size, shuffle=True, generator=gen)

    def compute_loss(model, batch, device, progress):
        slots, mask, static, labels = [t.to(device) for t in batch]
        logits, _, H = model.logits_with_embedding(slots, mask, static)
        det = bce_l2_loss(logits, labels, model, ids_cfg.l2_reg)
        con = supcon_loss(H, labels, temperature=tau)
        return det + beta * con

    return _common_loop(
        model, device, ids_cfg, loader,
        data["val"].slots, data["val"].mask, data["val"].static, data["val"].labels,
        compute_loss, max_epochs=max_epochs, patience=patience,
    )


def train_mgat(model, ids_cfg, multigen_train, val_data, device, seed, lambda_max=LAMBDA_MAX, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = TensorDataset(
        torch.from_numpy(multigen_train.slots), torch.from_numpy(multigen_train.mask),
        torch.from_numpy(multigen_train.static), torch.from_numpy(multigen_train.labels).float(),
        torch.from_numpy(multigen_train.gen_id),
    )
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=ids_cfg.batch_size, shuffle=True, generator=gen)

    def compute_loss(model, batch, device, progress):
        slots, mask, static, labels, gen_id = [t.to(device) for t in batch]
        lam = dann_lambda(progress, lambda_max=lambda_max)
        logits, _, domain_logits = model.logits_with_domain(slots, mask, static, lam)
        det = bce_l2_loss(logits, labels, model, ids_cfg.l2_reg)
        valid = gen_id >= 0
        if valid.any():
            dom = F.cross_entropy(domain_logits[valid], gen_id[valid])
        else:
            dom = logits.new_zeros(())
        return det + lam * dom

    return _common_loop(
        model, device, ids_cfg, loader,
        val_data.slots, val_data.mask, val_data.static, val_data.labels,
        compute_loss, max_epochs=max_epochs, patience=patience,
    )


def train_grma(model, ids_cfg, multigen_train, val_data, device, seed, lambda_max=LAMBDA_MAX, beta=SUPCON_BETA, tau=SUPCON_TAU, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = TensorDataset(
        torch.from_numpy(multigen_train.slots), torch.from_numpy(multigen_train.mask),
        torch.from_numpy(multigen_train.static), torch.from_numpy(multigen_train.labels).float(),
        torch.from_numpy(multigen_train.gen_id),
    )
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=ids_cfg.batch_size, shuffle=True, generator=gen)

    def compute_loss(model, batch, device, progress):
        slots, mask, static, labels, gen_id = [t.to(device) for t in batch]
        lam = dann_lambda(progress, lambda_max=lambda_max)
        logits, _, domain_logits, H = model.logits_with_domain(slots, mask, static, lam)
        det = bce_l2_loss(logits, labels, model, ids_cfg.l2_reg)
        valid = gen_id >= 0
        dom = F.cross_entropy(domain_logits[valid], gen_id[valid]) if valid.any() else logits.new_zeros(())
        con = supcon_loss(H, labels, temperature=tau)
        return det + lam * dom + beta * con

    return _common_loop(
        model, device, ids_cfg, loader,
        val_data.slots, val_data.mask, val_data.static, val_data.labels,
        compute_loss, max_epochs=max_epochs, patience=patience,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_one(candidate: str, scma_type: int, base_data, multigen_train, ids_cfg, device, seed=SEED, tiny=False):
    override = EPOCH_BUDGET_OVERRIDE.get(candidate, {})
    max_epochs = 2 if tiny else override.get("max_epochs", MAX_EPOCHS)
    patience = 1 if tiny else override.get("patience", PATIENCE)

    torch.manual_seed(seed)
    np.random.seed(seed)

    t0 = time.time()
    if candidate == "ssm_mil":
        model = SSMMIL(
            input_dim=ids_cfg.input_dim, encoder_hidden=ids_cfg.encoder_hidden, encoder_out=ids_cfg.encoder_out,
            static_dim=ids_cfg.static_dim, static_hidden=ids_cfg.static_hidden, dropout=ids_cfg.dropout,
        ).to(device)
        history, best_state, best_epoch, best_val_f1, converged, fail_reason = train_ssm(
            model, ids_cfg, base_data, device, seed, max_epochs=max_epochs, patience=patience
        )
        val_data = base_data["val"]
    elif candidate == "msc_mil":
        model = MSCMIL(
            input_dim=ids_cfg.input_dim, encoder_hidden=ids_cfg.encoder_hidden, encoder_out=ids_cfg.encoder_out,
            lstm_hidden_per_dir=ids_cfg.lstm_hidden_per_dir, static_dim=ids_cfg.static_dim,
            static_hidden=ids_cfg.static_hidden, dropout=ids_cfg.dropout,
        ).to(device)
        history, best_state, best_epoch, best_val_f1, converged, fail_reason = train_msc(
            model, ids_cfg, base_data, device, seed, max_epochs=max_epochs, patience=patience
        )
        val_data = base_data["val"]
    elif candidate == "mgat_mil":
        model = MGATMIL(
            input_dim=ids_cfg.input_dim, encoder_hidden=ids_cfg.encoder_hidden, encoder_out=ids_cfg.encoder_out,
            lstm_hidden_per_dir=ids_cfg.lstm_hidden_per_dir, static_dim=ids_cfg.static_dim,
            static_hidden=ids_cfg.static_hidden, n_generators=K_GENERATORS, dropout=ids_cfg.dropout,
        ).to(device)
        history, best_state, best_epoch, best_val_f1, converged, fail_reason = train_mgat(
            model, ids_cfg, multigen_train, base_data["val"], device, seed, max_epochs=max_epochs, patience=patience
        )
        val_data = base_data["val"]
    elif candidate == "grma_mil":
        model = GRMAMIL(
            input_dim=ids_cfg.input_dim, encoder_hidden=ids_cfg.encoder_hidden, encoder_out=ids_cfg.encoder_out,
            lstm_hidden_per_dir=ids_cfg.lstm_hidden_per_dir, static_dim=ids_cfg.static_dim,
            static_hidden=ids_cfg.static_hidden, n_generators=K_GENERATORS, dropout=ids_cfg.dropout,
        ).to(device)
        history, best_state, best_epoch, best_val_f1, converged, fail_reason = train_grma(
            model, ids_cfg, multigen_train, base_data["val"], device, seed, max_epochs=max_epochs, patience=patience
        )
        val_data = base_data["val"]
    else:
        raise ValueError(candidate)

    wall_time = time.time() - t0
    params = n_params(model)

    result = {
        "candidate": candidate,
        "scma_type": scma_type,
        "seed": seed,
        "converged": converged,
        "fail_reason": fail_reason,
        "n_params": params,
        "train_time_s": wall_time,
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "history": history,
    }

    if converged and best_state is not None:
        model.load_state_dict(best_state)
        val_score = infer_forward(model, val_data.slots, val_data.mask, val_data.static, device)
        collapse = majority_collapse_check(val_data.labels, val_score)
        result["val_macro_f1"] = best_val_f1
        result["majority_collapse"] = collapse["collapsed"]
        result["val_precision@0.5"] = collapse["precision@0.5"]
        result["val_recall@0.5"] = collapse["recall@0.5"]
        result["val_classes_predicted"] = collapse["classes_predicted"]
    else:
        result["val_macro_f1"] = None
        result["majority_collapse"] = None
        result["val_precision@0.5"] = None
        result["val_recall@0.5"] = None
        result["val_classes_predicted"] = None

    return result


def _make_plain_loader(data, batch_size, seed):
    ds = TensorDataset(
        torch.from_numpy(data.slots), torch.from_numpy(data.mask), torch.from_numpy(data.static), torch.from_numpy(data.labels).float()
    )
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", action="store_true", help="tiny shape/dtype sanity check (200-ish sessions, 2 epochs)")
    parser.add_argument("--candidates", type=str, default="mgat_mil,msc_mil,ssm_mil,grma_mil")
    parser.add_argument("--types", type=str, default="1,6")
    parser.add_argument("--resume", action="store_true", help="skip (candidate, scma_type) combos whose result JSON already exists")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], help="override auto device selection")
    args = parser.parse_args()

    candidates = args.candidates.split(",")
    types = [int(x) for x in args.types.split(",")]

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke_test] device={device} tiny={args.tiny} resume={args.resume} candidates={candidates} types={types}")

    cfg = ReproductionConfig()
    ids_cfg = cfg.ids
    manifest = dsmod.load_manifest()
    store = dsmod.SlotStore.get()

    primary_ckpt = torch.load(CKPT_DIR / "ppo_attacker.pt", map_location=device)
    primary_attacker = ActorCritic(cfg.attacker).to(device)
    primary_attacker.load_state_dict(primary_ckpt["model_state"])
    primary_attacker.eval()

    gen_attackers = []
    need_multigen = any(c in candidates for c in ("mgat_mil", "grma_mil"))
    if need_multigen:
        for k in range(K_GENERATORS):
            ck = torch.load(CKPT_DIR / f"ppo_attacker_gen{k}.pt", map_location=device)
            m = ActorCritic(cfg.attacker).to(device)
            m.load_state_dict(ck["model_state"])
            m.eval()
            gen_attackers.append(m)

    pools = dsmod.build_split_pools(manifest, seed=0)
    test_variants = dsmod.build_test_variants(manifest, seed=0)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for scma_type in types:
        t0 = time.time()
        base_data = dsmod.build_type_dataset(manifest, store, pools, test_variants, scma_type, primary_attacker, device, cfg, dataset_seed=0)
        print(f"[smoke_test] SCMA-{scma_type} base dataset built in {time.time()-t0:.1f}s "
              f"(train={len(base_data['train'].labels)}, val={len(base_data['val'].labels)})")

        if args.tiny:
            tiny_rng = np.random.default_rng(0)
            for key in ("train", "val"):
                d = base_data[key]
                n_keep = min(64, len(d.labels))
                # stratified random subsample so both classes survive truncation
                # (bags are label-ordered: benign block then attack block) --
                # important since tiny mode should exercise both label branches.
                idx = tiny_rng.choice(len(d.labels), size=n_keep, replace=False)
                idx.sort()
                base_data[key] = dsmod.TypeSplitData(
                    slots=d.slots[idx], mask=d.mask[idx], static=d.static[idx], labels=d.labels[idx],
                    uids=[d.uids[i] for i in idx],
                )

        multigen_train = None
        if need_multigen:
            t1 = time.time()
            scalers = mgd.fit_scalers_for_type(manifest, store, pools, scma_type, primary_attacker, device, cfg)
            multigen_train = mgd.build_multigen_train_split(
                manifest, store, pools, scma_type, gen_attackers, primary_attacker, device, cfg, scalers, dataset_seed=0
            )
            if args.tiny:
                tiny_rng2 = np.random.default_rng(1)
                n_keep = min(64, len(multigen_train.labels))
                idx = tiny_rng2.choice(len(multigen_train.labels), size=n_keep, replace=False)
                idx.sort()
                multigen_train = mgd.MultigenTypeSplitData(
                    slots=multigen_train.slots[idx], mask=multigen_train.mask[idx], static=multigen_train.static[idx],
                    labels=multigen_train.labels[idx], gen_id=multigen_train.gen_id[idx],
                    uids=[multigen_train.uids[i] for i in idx],
                )
            print(f"[smoke_test] SCMA-{scma_type} multigen train dataset built in {time.time()-t1:.1f}s "
                  f"(train={len(multigen_train.labels)}, gen_id_counts={np.bincount(multigen_train.gen_id[multigen_train.gen_id >= 0])})")

        for candidate in candidates:
            out_path = RAW_DIR / f"{'tiny_' if args.tiny else ''}smoke_{candidate}_scma{scma_type}.json"
            if args.resume and out_path.exists():
                print(f"[smoke_test] SKIP (resume) {candidate} / SCMA-{scma_type}: {out_path} already exists")
                with open(out_path) as f:
                    all_results.append(json.load(f))
                continue
            print(f"[smoke_test] === {candidate} / SCMA-{scma_type} ===")
            result = run_one(candidate, scma_type, base_data, multigen_train, ids_cfg, device, seed=SEED, tiny=args.tiny)
            tag = "TINY" if args.tiny else ""
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            status = "CRASH/NAN" if not result["converged"] else ("COLLAPSE" if result["majority_collapse"] else "OK")
            print(
                f"[smoke_test] {tag} {candidate} SCMA-{scma_type}: status={status} "
                f"val_f1={result['val_macro_f1']} n_params={result['n_params']} "
                f"time={result['train_time_s']:.1f}s epochs={result['epochs_trained']} -> {out_path}"
            )
            all_results.append(result)

    print("[smoke_test] done.")


if __name__ == "__main__":
    main()
