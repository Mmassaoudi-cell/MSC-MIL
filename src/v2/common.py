"""Shared helpers for the v2 (curve-preserving attack) study. Nothing here
modifies v1 code; v1 modules are imported read-only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reproduction import dataset as dsmod  # noqa: E402
from src.reproduction.config import ReproductionConfig  # noqa: E402
from src.reproduction.ppo_attacker import ActorCritic  # noqa: E402

CKPT = PROJECT_ROOT / "results" / "checkpoints" / "ppo_attacker.pt"
RESULTS_V2 = PROJECT_ROOT / "results" / "v2"
DATA_V2 = PROJECT_ROOT / "data" / "processed_v2"

FEATURE_NAMES = ["e_demand", "t_avail", "t_start(conn_hod proxy)", "t_end", "act_pow_mean", "act_pow_std", "act_pow_max"]


def load_attacker(device="cpu"):
    cfg = ReproductionConfig()
    ck = torch.load(CKPT, map_location=device)
    m = ActorCritic(cfg.attacker).to(device)
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m, cfg


def summary_features(d) -> np.ndarray:
    """(M,7) tree-model summary features from a TypeSplitData. Numerically the
    same as src.benchmarks.data_utils.flatten_features (v1) but vectorised:
    [e_demand, t_avail, conn-hour proxy for t_start, t_end, active-slot power
    mean/std/max] in the TRAIN-scaled units the tensors carry."""
    x = d.slots[:, :, 0].astype(np.float64)
    m = d.mask
    n = m.sum(1)
    safe = np.maximum(n, 1)
    mean = (x * m).sum(1) / safe
    var = (((x - mean[:, None]) ** 2) * m).sum(1) / safe
    std = np.sqrt(var)
    mx = np.where(m, x, -np.inf).max(1)
    empty = n == 0
    mean[empty] = 0.0
    std[empty] = 0.0
    mx[empty] = 0.0
    out = np.zeros((len(d.labels), 7), dtype=np.float64)
    out[:, 0] = d.static[:, 2]
    out[:, 1] = d.static[:, 3]
    out[:, 2] = d.static[:, 0]
    out[:, 3] = d.static[:, 1]
    out[:, 4] = mean
    out[:, 5] = std
    out[:, 6] = mx
    return out


def build_v1_train_val(scma_type: int = 1, device="cpu"):
    """v1 TRAIN and VAL only (identical construction to
    dataset.build_type_dataset, minus the test splits, which are never built)."""
    attacker, cfg = load_attacker(device)
    manifest = dsmod.load_manifest()
    store = dsmod.SlotStore.get()
    pools = dsmod.build_split_pools(manifest, seed=0)
    mi = manifest.set_index("session_uid")
    c_adj = cfg.attacker.c_adjust
    tb, ta = dsmod.generate_split_bags(pools["train"], mi, store, scma_type, attacker, device, c_adj)
    vb, va = dsmod.generate_split_bags(pools["val"], mi, store, scma_type, attacker, device, c_adj)
    scalers = dsmod.fit_scalers(tb)
    synth = dsmod.adasyn_augment(tb, ta, cfg.ids, scma_type, c_adj, seed=0)
    train = dsmod._bags_to_arrays(tb + ta + synth, scalers, seed=0)
    val = dsmod._bags_to_arrays(vb + va, scalers, seed=0)
    return train, val
