"""v2 attack realization: transform the session's REAL per-slot curve so it is
consistent with the attacked aggregate (Algorithm 1 output, unchanged), instead
of replacing it by a flat rate (v1, session_to_bag.build_attacked_bag).

Aggregate logic is untouched: `scma_env.generate_scma` decides the attacked
(t_start, t_end, e_demand) exactly as in v1. This module only decides what the
slot-level curve looks like.

Realization (works uniformly for SCMA-1..6 because every type is expressed by
the attacked aggregate relative to the original):
  1. window: shift_slots = round((att.t_start - orig.t_start)/dt),
             cut_slots   = round((orig.t_end   - att.t_end)/dt);
     new active slots = [a0+shift, a0+n_orig-cut) where [a0, a0+n_orig) is the
     real active run. Slots outside are inactive: power 0 and mask False.
  2. shape: new_power[ns+j] = real_curve[j] for j < n_new (the real curve is
     shifted later by shift_slots and truncated at the new end; SCMA-2 = pure
     truncation, SCMA-3 = pure shift, SCMA-4/5/6 = both).
  3. energy: multiply by scale = (att.e/orig.e) * E_full_curve / E_retained,
     i.e. the retained curve is rescaled so that its delivered energy equals
     the attacked energy ratio applied to the full real curve's energy. For
     SCMA-1 this is exactly att.e/orig.e (pure shape-preserving rescale).
  4. clip to [0, MAX_EVSE_KW=20] (src/data/build_session_dataset.py::MAX_EVSE_KW).
     When clipping bites, delivered energy is below the attacked target
     ("as closely as the truncated curve allows"); this is recorded.
"""
from __future__ import annotations

import numpy as np

from src.reproduction.scma_env import Session

N_SLOTS = 288
DT_H = 5.0 / 60.0
MAX_EVSE_KW = 20.0


def realize_attacked_curve(real_power: np.ndarray, real_active: np.ndarray, orig: Session, att: Session):
    """Returns (power[288] float32, active[288] bool, info dict)."""
    idx_act = np.flatnonzero(real_active)
    power = np.zeros(N_SLOTS, dtype=np.float32)
    active = np.zeros(N_SLOTS, dtype=bool)
    info = dict(scale=np.nan, e_target=np.nan, e_new=np.nan, clipped=False, n_orig=0, n_new=0,
                corr=np.nan, e_full=np.nan)
    if idx_act.size == 0:
        return power, active, info
    a0 = int(idx_act[0])
    n_orig = int(idx_act[-1] - a0 + 1)  # contiguous run assumption (checked in V2_DATA_REPORT)
    curve = real_power[a0:a0 + n_orig].astype(np.float64)
    shift = int(round((att.t_start_h - orig.t_start_h) / DT_H))
    cut = int(round((orig.t_end_h - att.t_end_h) / DT_H))
    shift = max(shift, 0)
    cut = max(cut, 0)
    ns = a0 + shift
    ne = a0 + n_orig - cut
    n_new = max(ne - ns, 0)
    info.update(n_orig=n_orig, n_new=n_new)
    if n_new <= 0:
        return power, active, info
    seg = curve[:n_new]
    e_full = curve.sum() * DT_H
    e_ret = seg.sum() * DT_H
    ratio = att.e_demand_kwh / max(orig.e_demand_kwh, 1e-9)
    e_target = ratio * e_full
    scale = (e_target / e_ret) if e_ret > 1e-9 else 0.0
    scaled = seg * scale
    clipped = bool((scaled > MAX_EVSE_KW).any())
    scaled = np.clip(scaled, 0.0, MAX_EVSE_KW)
    power[ns:ns + n_new] = scaled.astype(np.float32)
    active[ns:ns + n_new] = True
    if n_new >= 3 and seg.std() > 1e-9 and scaled.std() > 1e-9:
        info["corr"] = float(np.corrcoef(seg, scaled)[0, 1])
    info.update(scale=float(scale), e_target=float(e_target), e_new=float(scaled.sum() * DT_H),
                clipped=clipped, e_full=float(e_full))
    return power, active, info
