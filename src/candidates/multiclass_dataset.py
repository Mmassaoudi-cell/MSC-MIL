"""7-way multiclass dataset builder for the MSC-MIL attack-TYPE-attribution
task (Master Prompt Sec.17-18). Reuses src/reproduction/dataset.py's
building blocks VERBATIM (generate_split_bags, fit_scalers, bag_to_tensors,
make_attacked_bag, predict_c_attack, session_from_summary) -- nothing about
the underlying attack-generation or normalization pipeline is reimplemented.

Label scheme: benign session -> 0; a session attacked under SCMA-k -> k
(k in {1..6}).

**Documented session-overlap note** (per task brief, not "fixed"): the SAME
underlying attack-source session pool (`pools["train"].attack_uids` etc.,
fixed per split/seed by `build_split_pools`/`build_test_variants`) is reused
across all 6 SCMA types, exactly as `build_type_dataset` already does for
every prior binary per-type experiment in this project (dataset.py module
docstring point 4). This means the SAME real session, e.g. uid X, appears up
to 6 times in the combined multiclass dataset -- once per SCMA type -- each
time with a DIFFERENT attack transform applied and hence a DIFFERENT label
(1..6). This is not label leakage in the usual sense (each row's FEATURES are
the transformed bag for that specific type, not the raw session), but it does
mean the model can in principle see highly similar underlying baselines
across classes. Per the task brief's explicit instruction, this is
documented and NOT engineered around (no dedupe/held-out-session split by
attack type) -- a genuine scope simplification for this secondary evaluation
angle.

**Documented ADASYN scope reduction**: unlike the binary per-type datasets,
this combined multiclass dataset does NOT apply ADASYN train-time
augmentation (dataset.py's `adasyn_augment` is a binary
minority/majority-summary-vector procedure, not natively multiclass-aware,
and extending it 6-way was judged out of scope for this secondary
evaluation). Class balance is therefore whatever `build_split_pools`'s
90/10-per-type ratio naturally yields when combined: benign is the majority
class (~60% of TRAIN), the 6 attack classes split the remaining ~40% roughly
evenly (~6-7% each). Reported plainly in MULTICLASS_RESULTS.md, not smoothed
over.
"""
from __future__ import annotations

import numpy as np

from src.reproduction import dataset as dsmod

SCMA_TYPES_ALL = (1, 2, 3, 4, 5, 6)


def _generate_attack_bags_only(pool, manifest_idx, store, scma_type, attacker_model, device, c_adjust):
    """Attack-bag-only counterpart of dsmod.generate_split_bags -- avoids
    rebuilding the (identical, type-independent) benign bags redundantly for
    every one of the 6 SCMA types."""
    attack_rows = manifest_idx.loc[pool.attack_uids].reindex(pool.attack_uids)
    attack_row_list = list(attack_rows.itertuples(index=False))
    attack_sessions = [
        dsmod.session_from_summary(r.connection_time, r.disconnection_time, r.kwh_delivered)
        for r in attack_row_list
    ]
    c_attacks = dsmod.predict_c_attack(attack_sessions, attacker_model, device) if attack_sessions else np.array([])
    attack_bags = [
        dsmod.make_attacked_bag(u, row, scma_type, float(c), c_adjust)
        for u, row, c in zip(pool.attack_uids, attack_row_list, c_attacks)
    ]
    return attack_bags


def build_multiclass_dataset(
    manifest,
    store,
    pools: dict,
    test_variants: dict,
    attacker_model,
    device,
    cfg,
    dataset_seed: int = 0,
    scma_types: tuple = SCMA_TYPES_ALL,
) -> dict[str, "dsmod.TypeSplitData"]:
    """Returns {train, val, test_full} -> TypeSplitData with labels in
    {0, ..., len(scma_types)}. Scalers fit on TRAIN benign only (same
    convention as build_type_dataset). test_n200 is NOT built here (the
    n=200 paper-matched protocol is a per-type binary construct; the
    multiclass task uses only the {train, val, test_full}-style splits, per
    the task brief's framing of this as a secondary evaluation angle)."""
    manifest_idx = manifest.set_index("session_uid")
    c_adjust = cfg.attacker.c_adjust
    split_pool_map = {"train": pools["train"], "val": pools["val"], "test_full": test_variants["test_full"]}

    raw_bags: dict[str, tuple[list, list]] = {}
    for split_name, pool in split_pool_map.items():
        benign_bags, _ = dsmod.generate_split_bags(pool, manifest_idx, store, scma_types[0], attacker_model, device, c_adjust)
        bags = list(benign_bags)
        labels_override = [0] * len(benign_bags)
        for t in scma_types:
            attack_bags = _generate_attack_bags_only(pool, manifest_idx, store, t, attacker_model, device, c_adjust)
            bags.extend(attack_bags)
            labels_override.extend([t] * len(attack_bags))
        raw_bags[split_name] = (bags, labels_override)
        print(f"[multiclass_dataset] {split_name}: benign={len(benign_bags)} "
              f"attack_total={len(labels_override) - len(benign_bags)} "
              f"({', '.join(f'type{t}={labels_override.count(t)}' for t in scma_types)})")

    train_benign_bags, _ = dsmod.generate_split_bags(pools["train"], manifest_idx, store, scma_types[0], attacker_model, device, c_adjust)
    scalers = dsmod.fit_scalers(train_benign_bags)

    out = {}
    for split_name, (bags, labels_override) in raw_bags.items():
        rng = np.random.default_rng(dataset_seed)
        slots_list, mask_list, static_list, uids = [], [], [], []
        for b in bags:
            s, m, st = dsmod.bag_to_tensors(b, scalers, rng)
            slots_list.append(s)
            mask_list.append(m)
            static_list.append(st)
            uids.append(b.session_uid)
        out[split_name] = dsmod.TypeSplitData(
            slots=np.stack(slots_list).astype(np.float32),
            mask=np.stack(mask_list).astype(bool),
            static=np.stack(static_list).astype(np.float32),
            labels=np.array(labels_override, dtype=np.int64),
            uids=uids,
        )
    return out
