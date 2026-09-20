"""Dataset-construction layer bridging DATA_SPLIT_MANIFEST.csv + session_slots.parquet
(real ACN-Data telemetry) -> per-SCMA-type MIL bags consumable by MILBiLSTMIDS.

Design decisions documented here (none silently assumed, per project policy):

1. **Sample-size caps (train/val).** The eligible pool is large (37,068 train /
   7,942 val / 7,945 test sessions -- see DATA_AUDIT.md sec. 9), but training
   6 SCMA types x 5 seeds of a BiLSTM-attention model on the full 37k-session
   train pool per type is not tractable within this reproduction's compute
   budget. We cap TRAIN_BENIGN_N=4000 / VAL_BENIGN_N=800 (with attack counts
   derived from the paper's own 90/10 ratio), sampled once with a fixed seed
   from the eligible chronological pool, and reused identically across all 6
   SCMA types and all 5 IDS seeds for comparability. This is a *documented*
   resource-constrained scope reduction, distinct from the TEST split, which
   uses (a) the paper-matched N=200 subsample AND (b) the FULL eligible test
   pool (7,945 benign-eligible sessions) -- see REPRODUCTION_REPORT.md.

2. **90/10 benign:attack ratio** applied to train/val/test pools alike (the
   paper only states this ratio for its N=200 test set; SOURCE_PAPER_AUDIT.md
   documents extending it to train/val as the only stated ratio available).

3. **Attack-source sessions are drawn from the same eligible pool as benign
   sessions**, independently sampled (may overlap in underlying session
   identity with the benign subsample). This matches the paper's own "hybrid"
   dataset construction (DATA_AUDIT.md sec 3: "malicious sessions = same real
   sessions with DRL-perturbed energy/timing fields") -- session identity is
   not a model input, so overlap carries no label-leakage risk; only the
   ATTACK TRANSFORM differs between the two roles.

4. **The same underlying attack-source session subset is reused across all 6
   SCMA types** (fixed seed per split) -- consistent with Algorithm 1's
   structure, where C_attack is predicted once per session (line 15)
   independent of which manipulation branch (lines 16-36) is then applied.

5. **Per-slot raw feature** = [power_kw normalized by a StandardScaler fit on
   TRAIN active-slot power values ONLY, sin(2*pi*hour_of_day/24),
   cos(2*pi*hour_of_day/24)]. hour_of_day is computed from the session's
   REAL wall-clock connection_time + slot_idx*5min, mod 24 -- this anchor is
   kept fixed even for attacked bags whose active WINDOW may shift (SCMA-2/3),
   since Algorithm 1 never relocates a session to a different wall-clock day,
   only shifts its start/end within its own timeline.

6. **Static features** (session_to_bag.slot_static_features: [connection
   hour-of-day, t_end_h, e_demand_kwh, t_avail_h]) are standardized with a
   scaler fit on TRAIN BENIGN sessions only (documented choice: avoids fitting
   the "normal" reference distribution on synthetic/attacked data).

7. **ADASYN reconstruction (train only).** ADASYN operates on a 7-dim
   per-session summary vector [e_demand_kwh, duration_hours(=t_avail_h),
   t_start_h, t_end_h, mean/std/max active-slot power]. For each SYNTHETIC
   minority sample ADASYN returns, we reconstruct its slot-level bag with the
   same flat-rate-over-active-window logic as session_to_bag.build_attacked_bag
   (rate = interpolated e_demand / interpolated t_avail, active window =
   [interpolated t_start, t_end)). ADASYN's summary vector has no temporal
   (wall-clock) feature, so each synthetic sample's hour-of-day anchor is
   drawn uniformly at random in [0,24) -- documented nuisance-context choice,
   since hour-of-day is cyclical context, not part of the attack signal ADASYN
   is balancing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import StandardScaler

from .config import IDSConfig, ReproductionConfig
from .scma_env import Session
from .session_to_bag import (
    N_SLOTS,
    DT_MIN,
    build_attacked_bag,
    session_from_summary,
    slot_static_features,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "DATA_SPLIT_MANIFEST.csv"
SLOTS_PARQUET = PROJECT_ROOT / "data" / "processed" / "session_slots.parquet"

# --- documented sample-size caps (see module docstring point 1) ---
TRAIN_BENIGN_N = 4000
VAL_BENIGN_N = 800
TEST_N200_BENIGN = 180
TEST_N200_ATTACK = 20
ATTACK_RATIO = 0.10  # 90/10, Sec. V-C.3

SUMMARY_DIM = 7  # e_demand, t_avail(duration), t_start, t_end, mean/std/max active power


def _attack_count(benign_n: int) -> int:
    """90/10 ratio: attack = benign * (10/90)."""
    return max(1, round(benign_n * ATTACK_RATIO / (1 - ATTACK_RATIO)))


# ---------------------------------------------------------------------------
# Slot store: loads session_slots.parquet ONCE (24.7M rows), reshapes to
# (n_sessions, 288) arrays using the confirmed contiguous-288-row-per-session
# block structure (verified empirically: every session occupies exactly 288
# contiguous rows in original preprocessing order).
# ---------------------------------------------------------------------------
class SlotStore:
    _instance: "SlotStore | None" = None

    def __init__(self):
        df = pd.read_parquet(SLOTS_PARQUET, columns=["session_uid", "power_kw", "active"])
        uid = df["session_uid"].to_numpy()
        n = len(uid) // N_SLOTS
        assert len(uid) == n * N_SLOTS, "session_slots.parquet is not a clean 288-row-per-session block layout"
        self.uid_order = uid[::N_SLOTS].copy()
        self.power = df["power_kw"].to_numpy(dtype=np.float32).reshape(n, N_SLOTS)
        self.active = df["active"].to_numpy(dtype=bool).reshape(n, N_SLOTS)
        self.index = {u: i for i, u in enumerate(self.uid_order)}
        del df, uid

    @classmethod
    def get(cls) -> "SlotStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_slots(self, session_uid: str):
        i = self.index[session_uid]
        return self.power[i], self.active[i]


def load_manifest() -> pd.DataFrame:
    m = pd.read_csv(MANIFEST_PATH)
    # explicit to_datetime (not read_csv's parse_dates) -- the manifest's
    # timestamps carry mixed/UTC offsets that parse_dates silently leaves as
    # object dtype in this pandas version; to_datetime(utc=True) is robust.
    m["connection_time"] = pd.to_datetime(m["connection_time"], utc=True, format="ISO8601")
    m["disconnection_time"] = pd.to_datetime(m["disconnection_time"], utc=True, format="ISO8601")
    return m


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------
def hour_of_day_sincos(connection_time: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    base_hour = connection_time.hour + connection_time.minute / 60.0 + connection_time.second / 3600.0
    slot_offsets = np.arange(N_SLOTS) * (DT_MIN / 60.0)
    hod = (base_hour + slot_offsets) % 24.0
    ang = 2 * np.pi * hod / 24.0
    return np.sin(ang).astype(np.float32), np.cos(ang).astype(np.float32)


def random_hour_of_day_sincos(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """For ADASYN-synthetic samples with no real connection_time -- see docstring point 7."""
    base_hour = rng.uniform(0.0, 24.0)
    slot_offsets = np.arange(N_SLOTS) * (DT_MIN / 60.0)
    hod = (base_hour + slot_offsets) % 24.0
    ang = 2 * np.pi * hod / 24.0
    return np.sin(ang).astype(np.float32), np.cos(ang).astype(np.float32)


def summary_vec(sess: Session, power: np.ndarray, active: np.ndarray) -> np.ndarray:
    active_power = power[active]
    if active_power.size == 0:
        active_power = np.array([0.0], dtype=np.float32)
    return np.array(
        [
            sess.e_demand_kwh,
            sess.t_avail_h,
            sess.t_start_h,
            sess.t_end_h,
            float(active_power.mean()),
            float(active_power.std()),
            float(active_power.max()),
        ],
        dtype=np.float32,
    )


@dataclass
class Bag:
    slots: np.ndarray  # (288, 3) float32, RAW (unnormalized) power in channel 0
    active: np.ndarray  # (288,) bool
    sess: Session
    connection_time: pd.Timestamp | None  # None for ADASYN-synthetic bags
    label: int
    session_uid: str


def make_benign_bag(uid: str, row: pd.Series, store: SlotStore) -> Bag:
    power, active = store.get_slots(uid)
    sess = session_from_summary(row.connection_time, row.disconnection_time, row.kwh_delivered)
    return Bag(slots=power.copy(), active=active.copy(), sess=sess, connection_time=row.connection_time, label=0, session_uid=uid)


def make_attacked_bag(uid: str, row: pd.Series, scma_type: int, c_attack: float, c_adjust: float) -> Bag:
    sess = session_from_summary(row.connection_time, row.disconnection_time, row.kwh_delivered)
    power, active, attacked_sess = build_attacked_bag(sess, scma_type, c_attack, c_adjust)
    return Bag(slots=power, active=active, sess=attacked_sess, connection_time=row.connection_time, label=1, session_uid=uid + f"::scma{scma_type}")


# ---------------------------------------------------------------------------
# Scalers (fit on TRAIN only)
# ---------------------------------------------------------------------------
@dataclass
class Scalers:
    power: StandardScaler
    static: StandardScaler


def fit_scalers(benign_train_bags: list[Bag]) -> Scalers:
    """Power scaler: TRAIN active-slot power values only (module docstring pt.5).
    Static scaler: TRAIN BENIGN sessions only (pt.6)."""
    active_powers = []
    static_feats = []
    for b in benign_train_bags:
        active_powers.append(b.slots[b.active])
        conn_hod = b.connection_time.hour + b.connection_time.minute / 60.0
        static_feats.append(slot_static_features(b.sess, conn_hod))
    power_vals = np.concatenate(active_powers).reshape(-1, 1) if active_powers else np.zeros((1, 1))
    power_scaler = StandardScaler().fit(power_vals)
    static_scaler = StandardScaler().fit(np.stack(static_feats))
    return Scalers(power=power_scaler, static=static_scaler)


def bag_to_tensors(bag: Bag, scalers: Scalers, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (slots[288,3] normalized, mask[288] bool, static[4] normalized)."""
    p_norm = scalers.power.transform(bag.slots.reshape(-1, 1)).reshape(-1).astype(np.float32)
    if bag.connection_time is not None:
        sin_h, cos_h = hour_of_day_sincos(bag.connection_time)
    else:
        assert rng is not None, "ADASYN-synthetic bags need an rng for random hour-of-day anchor"
        sin_h, cos_h = random_hour_of_day_sincos(rng)
    slots = np.stack([p_norm, sin_h, cos_h], axis=-1)
    conn_hod = (bag.connection_time.hour + bag.connection_time.minute / 60.0) if bag.connection_time is not None else float(rng.uniform(0, 24))
    static_raw = slot_static_features(bag.sess, conn_hod)
    static = scalers.static.transform(static_raw.reshape(1, -1)).reshape(-1).astype(np.float32)
    return slots, bag.active.astype(bool), static


# ---------------------------------------------------------------------------
# PPO attacker inference (frozen policy -> C_attack per session)
# ---------------------------------------------------------------------------
def predict_c_attack(sessions: list[Session], model, device) -> np.ndarray:
    """Deterministic (policy-mean) C_attack per session, processed as a single
    temporal sequence through the LSTM policy (consistent with how it was
    trained -- Fig.2's LSTM-based actor). We use the policy MEAN (not a
    stochastic sample) at dataset-generation time: Algorithm 1 predicts one
    C_attack per session for deployment, not an exploratory action -- documented
    choice since the paper doesn't distinguish train-time vs. deploy-time
    action selection for the frozen attacker."""
    model.eval()
    with torch.no_grad():
        state_seq = torch.tensor(
            np.array([[s.t_start_h, s.t_end_h, s.e_demand_kwh] for s in sessions], dtype=np.float32),
            device=device,
        ).unsqueeze(0)
        mu, std, v, _ = model(state_seq)
        c_attack = mu.squeeze(0).squeeze(-1).cpu().numpy()
    return c_attack


# ---------------------------------------------------------------------------
# Split pool construction
# ---------------------------------------------------------------------------
@dataclass
class SplitPool:
    benign_uids: list[str]
    attack_uids: list[str]  # shared attack-source session subset across all 6 SCMA types


def build_split_pools(manifest: pd.DataFrame, seed: int = 0) -> dict[str, SplitPool]:
    rng = np.random.default_rng(seed)
    pools = {}
    sizes = {
        "train": (TRAIN_BENIGN_N, _attack_count(TRAIN_BENIGN_N)),
        "val": (VAL_BENIGN_N, _attack_count(VAL_BENIGN_N)),
    }
    for split, (n_benign, n_attack) in sizes.items():
        sub = manifest[manifest.split == split]
        n_benign = min(n_benign, len(sub))
        benign_idx = rng.choice(sub.index.to_numpy(), size=n_benign, replace=False)
        attack_idx = rng.choice(sub.index.to_numpy(), size=n_attack, replace=False)
        pools[split] = SplitPool(
            benign_uids=manifest.loc[benign_idx, "session_uid"].tolist(),
            attack_uids=manifest.loc[attack_idx, "session_uid"].tolist(),
        )
    # test: two variants built separately in build_test_variants()
    return pools


def build_test_variants(manifest: pd.DataFrame, seed: int = 0) -> dict[str, SplitPool]:
    rng = np.random.default_rng(seed + 1000)
    sub = manifest[manifest.split == "test"]
    all_idx = sub.index.to_numpy()

    # (a) paper-matched N=200: 180 benign + 20 attack, SAME 180 benign & 20
    # attack-source sessions reused across all 6 SCMA types (only the SCMA
    # transform differs), for direct Table-II comparability.
    n200_benign_idx = rng.choice(all_idx, size=min(TEST_N200_BENIGN, len(all_idx)), replace=False)
    remaining = np.setdiff1d(all_idx, n200_benign_idx)
    n200_attack_idx = rng.choice(remaining, size=min(TEST_N200_ATTACK, len(remaining)), replace=False)

    # (b) full eligible test pool: ALL benign-eligible test sessions + 10%
    # more (independently sampled) as attack sources.
    full_benign_idx = all_idx
    n_full_attack = _attack_count(len(full_benign_idx))
    full_attack_idx = rng.choice(all_idx, size=min(n_full_attack, len(all_idx)), replace=False)

    return {
        "test_n200": SplitPool(
            benign_uids=manifest.loc[n200_benign_idx, "session_uid"].tolist(),
            attack_uids=manifest.loc[n200_attack_idx, "session_uid"].tolist(),
        ),
        "test_full": SplitPool(
            benign_uids=manifest.loc[full_benign_idx, "session_uid"].tolist(),
            attack_uids=manifest.loc[full_attack_idx, "session_uid"].tolist(),
        ),
    }


# ---------------------------------------------------------------------------
# ADASYN augmentation (train only)
# ---------------------------------------------------------------------------
def adasyn_augment(
    benign_bags: list[Bag],
    attack_bags: list[Bag],
    cfg: IDSConfig,
    scma_type: int,
    c_adjust: float,
    seed: int,
) -> list[Bag]:
    """Returns extra SYNTHETIC minority (attack) Bags to append to attack_bags.
    See module docstring point 7 for the reconstruction logic."""
    X0 = np.stack([summary_vec(b.sess, b.slots, b.active) for b in benign_bags])
    X1 = np.stack([summary_vec(b.sess, b.slots, b.active) for b in attack_bags])
    X = np.concatenate([X0, X1], axis=0)
    y = np.concatenate([np.zeros(len(X0), dtype=int), np.ones(len(X1), dtype=int)])

    n_minority = int(y.sum())
    k = min(cfg.adasyn_k_neighbors, max(1, n_minority - 1))
    if n_minority <= 1 or k < 1:
        return []  # not enough minority samples to run ADASYN meaningfully
    try:
        adasyn = ADASYN(n_neighbors=k, random_state=seed)
        X_res, y_res = adasyn.fit_resample(X, y)
    except ValueError:
        return []  # ADASYN can refuse if already ~balanced / degenerate neighborhoods

    n_orig = len(X)
    synth_mask = np.zeros(len(X_res), dtype=bool)
    synth_mask[n_orig:] = True
    synth_minority = X_res[synth_mask & (y_res == 1)]

    rng = np.random.default_rng(seed)
    out = []
    for row in synth_minority:
        e_demand, t_avail, t_start, t_end, p_mean, p_std, p_max = row
        t_start = max(float(t_start), 0.0)
        t_end = max(float(t_end), t_start + 1e-3)
        t_avail = max(float(t_end - t_start), 1e-3)
        e_demand = max(float(e_demand), 1e-3)
        sess = Session(t_start_h=t_start, t_end_h=t_end, e_demand_kwh=e_demand, t_avail_h=t_avail, dt_min=DT_MIN)
        r_new = e_demand / t_avail
        slot_h = np.arange(N_SLOTS) * (DT_MIN / 60.0)
        active = (slot_h >= t_start) & (slot_h < t_end)
        power = np.where(active, r_new, 0.0).astype(np.float32)
        out.append(
            Bag(
                slots=power,
                active=active.astype(bool),
                sess=sess,
                connection_time=None,  # synthetic -> random hour-of-day anchor at tensor-build time
                label=1,
                session_uid=f"adasyn_synth::scma{scma_type}::{len(out)}",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Full per-(split, scma_type) bag builder
# ---------------------------------------------------------------------------
@dataclass
class TypeSplitData:
    slots: np.ndarray  # (M, 288, 3)
    mask: np.ndarray  # (M, 288)
    static: np.ndarray  # (M, 4)
    labels: np.ndarray  # (M,)
    uids: list[str]


def _bags_to_arrays(bags: list[Bag], scalers: Scalers, seed: int) -> TypeSplitData:
    rng = np.random.default_rng(seed)
    slots_list, mask_list, static_list, labels, uids = [], [], [], [], []
    for b in bags:
        s, m, st = bag_to_tensors(b, scalers, rng)
        slots_list.append(s)
        mask_list.append(m)
        static_list.append(st)
        labels.append(b.label)
        uids.append(b.session_uid)
    return TypeSplitData(
        slots=np.stack(slots_list).astype(np.float32),
        mask=np.stack(mask_list).astype(bool),
        static=np.stack(static_list).astype(np.float32),
        labels=np.array(labels, dtype=np.int64),
        uids=uids,
    )


def generate_split_bags(
    pool: SplitPool,
    manifest_idx: pd.DataFrame,
    store: SlotStore,
    scma_type: int,
    attacker_model,
    device,
    c_adjust: float,
) -> tuple[list[Bag], list[Bag]]:
    """Public helper (reused by baselines_table3.py too): builds (benign_bags,
    attack_bags) for one SplitPool under one SCMA type, using the frozen PPO
    attacker's deterministic C_attack prediction for the attack-source
    sessions."""
    benign_rows = manifest_idx.loc[pool.benign_uids].reindex(pool.benign_uids)
    benign_bags = [make_benign_bag(u, row, store) for u, row in zip(pool.benign_uids, benign_rows.itertuples(index=False))]

    attack_rows = manifest_idx.loc[pool.attack_uids].reindex(pool.attack_uids)
    attack_row_list = list(attack_rows.itertuples(index=False))
    attack_sessions = [
        session_from_summary(r.connection_time, r.disconnection_time, r.kwh_delivered) for r in attack_row_list
    ]
    c_attacks = predict_c_attack(attack_sessions, attacker_model, device) if attack_sessions else np.array([])
    attack_bags = [
        make_attacked_bag(u, row, scma_type, float(c), c_adjust)
        for u, row, c in zip(pool.attack_uids, attack_row_list, c_attacks)
    ]
    return benign_bags, attack_bags


def build_type_dataset(
    manifest: pd.DataFrame,
    store: SlotStore,
    pools: dict,
    test_variants: dict,
    scma_type: int,
    attacker_model,
    device,
    cfg: ReproductionConfig,
    dataset_seed: int = 0,
) -> dict[str, TypeSplitData]:
    """Builds train (benign + real-attack + ADASYN-synthetic-attack), val,
    test_n200, test_full bags + tensors for one SCMA type. Scalers are fit on
    TRAIN only and reused for val/test (module docstring pt.5-6)."""
    manifest_idx = manifest.set_index("session_uid")
    c_adjust = cfg.attacker.c_adjust

    train_benign, train_attack = generate_split_bags(pools["train"], manifest_idx, store, scma_type, attacker_model, device, c_adjust)
    val_benign, val_attack = generate_split_bags(pools["val"], manifest_idx, store, scma_type, attacker_model, device, c_adjust)
    n200_benign, n200_attack = generate_split_bags(test_variants["test_n200"], manifest_idx, store, scma_type, attacker_model, device, c_adjust)
    full_benign, full_attack = generate_split_bags(test_variants["test_full"], manifest_idx, store, scma_type, attacker_model, device, c_adjust)

    scalers = fit_scalers(train_benign)

    synth = adasyn_augment(train_benign, train_attack, cfg.ids, scma_type, c_adjust, seed=dataset_seed)

    out = {
        "train": _bags_to_arrays(train_benign + train_attack + synth, scalers, seed=dataset_seed),
        "val": _bags_to_arrays(val_benign + val_attack, scalers, seed=dataset_seed),
        "test_n200": _bags_to_arrays(n200_benign + n200_attack, scalers, seed=dataset_seed),
        "test_full": _bags_to_arrays(full_benign + full_attack, scalers, seed=dataset_seed),
    }
    return out
