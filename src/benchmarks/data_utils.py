"""Shared data-loading + feature-flattening utilities for the benchmark suite
(Master Prompt-style benchmark comparison task). Reuses
src/reproduction/dataset.py's build_type_dataset(...) verbatim -- NOT
reimplemented -- so every benchmark and MSC-MIL see byte-identical
train/val/test_n200/test_full tensors for a given SCMA type.

Flattened 7-dim feature vector for the classical/tree/boosting benchmarks:
dataset.py's adasyn_augment() operates on a 7-dim per-session summary vector
[e_demand_kwh, t_avail_h, t_start_h, t_end_h, mean/std/max active-slot power]
computed from the raw Bag/Session objects. Those raw objects are not retained
in the TypeSplitData the shared dataset builder returns (by design -- it only
returns the final tensors: slots (M,288,3), mask (M,288), static (M,4),
labels (M,)). We therefore reconstruct an equivalent 7-dim vector directly
from TypeSplitData's own tensors (documented, not silently improvised):

  static[:, 0] = connection hour-of-day (scaled)   -> used as t_start_h proxy
  static[:, 1] = t_end_h (scaled)
  static[:, 2] = e_demand_kwh (scaled)
  static[:, 3] = t_avail_h (scaled)
  slots[:, :, 0] = per-slot scaled power; masked-mean/std/max over active
    slots (mask==True) gives the power-shape features.

This uses the SAME scaled quantities MSC-MIL/benchmarks 8-11 see (StandardScaler
fit on TRAIN, as documented in dataset.py) rather than re-deriving raw kWh
units -- irrelevant to scale-invariant tree models and harmless (monotonic
rescale) to LogReg/SVM/kNN, which internally standardize/are distance-based
anyway. This is a deliberate simplification to avoid threading raw Session
objects through a second parallel code path; it does not change what
information is available to the flattened-feature benchmarks vs. the
sequence-aware ones.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.reproduction import dataset as dsmod
from src.reproduction.config import ReproductionConfig
from src.reproduction.ppo_attacker import ActorCritic

PROJECT_ROOT = dsmod.PROJECT_ROOT
CKPT_DIR = PROJECT_ROOT / "results" / "checkpoints"

_CACHE: dict[int, dict] = {}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_scma_data(scma_type: int, device: torch.device | None = None) -> dict:
    """Returns dict(train/val/test_n200/test_full -> TypeSplitData) for one
    SCMA type, built exactly as MSC-MIL/candidates stages built it (same
    pools/attacker/seed=0), cached per-process."""
    if scma_type in _CACHE:
        return _CACHE[scma_type]
    device = device or get_device()
    cfg = ReproductionConfig()
    manifest = dsmod.load_manifest()
    store = dsmod.SlotStore.get()

    ckpt = torch.load(CKPT_DIR / "ppo_attacker.pt", map_location=device)
    attacker = ActorCritic(cfg.attacker).to(device)
    attacker.load_state_dict(ckpt["model_state"])
    attacker.eval()

    pools = dsmod.build_split_pools(manifest, seed=0)
    test_variants = dsmod.build_test_variants(manifest, seed=0)

    data = dsmod.build_type_dataset(manifest, store, pools, test_variants, scma_type, attacker, device, cfg, dataset_seed=0)
    _CACHE[scma_type] = data
    return data


def flatten_features(d: "dsmod.TypeSplitData") -> np.ndarray:
    """(M, 7) flattened summary-vector features, see module docstring."""
    slots = d.slots[:, :, 0]  # (M, 288) scaled power
    mask = d.mask  # (M, 288) bool
    out = np.zeros((len(d.labels), 7), dtype=np.float64)
    for i in range(len(d.labels)):
        active = slots[i, mask[i]]
        if active.size == 0:
            active = np.array([0.0])
        out[i, 0] = d.static[i, 2]  # e_demand_kwh (scaled)
        out[i, 1] = d.static[i, 3]  # t_avail_h (scaled)
        out[i, 2] = d.static[i, 0]  # t_start_h proxy (connection hour-of-day, scaled)
        out[i, 3] = d.static[i, 1]  # t_end_h (scaled)
        out[i, 4] = active.mean()
        out[i, 5] = active.std()
        out[i, 6] = active.max()
    return out
