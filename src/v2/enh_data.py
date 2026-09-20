"""Validation-only data access for the v2 enhancement study.

HARD RULE: only 'train' and 'val' are ever read. Any other split key raises.
np.load on an .npz is lazy, so test arrays in the cache file are never
deserialized.
"""
from __future__ import annotations

import numpy as np

from src.v2.common import DATA_V2, dsmod

ALLOWED = ("train", "val")


def load_v2_trainval(scma_type: int, with_extras: bool = False):
    """-> (dict split -> TypeSplitData for train/val, meta dict[, extras])."""
    path = DATA_V2 / f"scma{scma_type}_v2.npz"
    z = np.load(path, allow_pickle=False)
    out, extras = {}, {}
    for sp in ALLOWED:
        out[sp] = dsmod.TypeSplitData(slots=z[f"{sp}__slots"], mask=z[f"{sp}__mask"], static=z[f"{sp}__static"],
                                      labels=z[f"{sp}__labels"], uids=[str(u) for u in z[f"{sp}__uids"]])
        if with_extras:
            extras[sp] = {k[len(sp) + 4:]: z[k] for k in z.files if k.startswith(f"{sp}__x_")}
    meta = {k[6:]: z[k].astype(np.float64) for k in z.files if k.startswith("meta__")}
    return (out, meta, extras) if with_extras else (out, meta)


def get_split(data: dict, name: str):
    if name not in ALLOWED:
        raise PermissionError(f"split '{name}' is forbidden during development (train/val only)")
    return data[name]
