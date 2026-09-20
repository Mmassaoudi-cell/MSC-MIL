"""Multi-generator training-split builder for Candidate 1 (MGAT-MIL) and
Candidate 4 (GRMA-MIL).

Design decision (documented, not silently assumed -- MODEL_CANDIDATES.md
specifies WHAT the multi-generator pool is for -- training-time generator-
invariance -- but not how VAL/TEST attack generation should relate to it):

  * TRAIN split's attack bags are drawn from the K=4-generator pool
    (round-robin sharded across the shared attack-source session subset),
    each bag tagged with its generator_id for the GRL domain loss.
  * VAL / TEST_N200 / TEST_FULL splits, and the scalers, are reused
    UNCHANGED from the single-generator baseline dataset
    (dataset.build_type_dataset with the ORIGINAL frozen
    results/checkpoints/ppo_attacker.pt) -- so val/test macro-F1 numbers
    for MGAT-MIL/GRMA-MIL are directly comparable to the base MIL-BiLSTM
    IDS's own val/test numbers (same evaluation distribution), and only the
    TRAINING protocol differs. Testing generalization to a HELD-OUT 5th
    generator never seen in training is explicitly the cross-generator
    robustness experiment for Stage 2/3 (Master Prompt Sec.18/20), not this
    Stage-1 smoke test.
  * ADASYN-synthetic minority bags (train only) are appended with
    generator_id = -1 (excluded from the domain-classification loss --
    they have no real generator source) but included in the detection loss
    exactly as in the baseline.
  * Scalers are refit on train-benign bags built with the PRIMARY (baseline)
    attacker model; since benign bags never depend on the attacker at all
    (dataset.make_benign_bag takes no attacker argument), this produces
    IDENTICAL scalers to the ones dataset.build_type_dataset would fit
    internally for the same scma_type/pools/seed -- verified by construction,
    not re-derived independently.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.reproduction import dataset as dsmod
from src.reproduction.config import ReproductionConfig


@dataclass
class MultigenTypeSplitData:
    slots: np.ndarray  # (M, 288, 3)
    mask: np.ndarray  # (M, 288)
    static: np.ndarray  # (M, 4)
    labels: np.ndarray  # (M,)
    gen_id: np.ndarray  # (M,) int64; -1 for benign/synthetic, else 0..K-1
    uids: list


def _bags_to_multigen_arrays(bags_with_gen: list[tuple], scalers: dsmod.Scalers, seed: int) -> MultigenTypeSplitData:
    rng = np.random.default_rng(seed)
    slots_list, mask_list, static_list, labels, gen_ids, uids = [], [], [], [], [], []
    for bag, gid in bags_with_gen:
        s, m, st = dsmod.bag_to_tensors(bag, scalers, rng)
        slots_list.append(s)
        mask_list.append(m)
        static_list.append(st)
        labels.append(bag.label)
        gen_ids.append(gid)
        uids.append(bag.session_uid)
    return MultigenTypeSplitData(
        slots=np.stack(slots_list).astype(np.float32),
        mask=np.stack(mask_list).astype(bool),
        static=np.stack(static_list).astype(np.float32),
        labels=np.array(labels, dtype=np.int64),
        gen_id=np.array(gen_ids, dtype=np.int64),
        uids=uids,
    )


def build_multigen_train_split(
    manifest: pd.DataFrame,
    store: dsmod.SlotStore,
    pools: dict,
    scma_type: int,
    attacker_models: list,
    primary_attacker,
    device,
    cfg: ReproductionConfig,
    scalers: dsmod.Scalers,
    dataset_seed: int = 0,
) -> MultigenTypeSplitData:
    """Builds the TRAIN split only, with attack bags drawn round-robin from
    K=len(attacker_models) generators (see module docstring). VAL/TEST reuse
    dataset.build_type_dataset's single-generator splits directly -- callers
    should build those separately (see smoke_test.py)."""
    manifest_idx = manifest.set_index("session_uid")
    c_adjust = cfg.attacker.c_adjust
    K = len(attacker_models)
    train_pool = pools["train"]

    # benign bags: attacker-independent, built once with the primary attacker
    # (arbitrary choice since benign bags don't consume attacker output).
    benign_only_pool = dsmod.SplitPool(benign_uids=train_pool.benign_uids, attack_uids=[])
    train_benign, _ = dsmod.generate_split_bags(
        benign_only_pool, manifest_idx, store, scma_type, primary_attacker, device, c_adjust
    )

    # shard the shared attack-source session subset round-robin across K generators
    shards = [train_pool.attack_uids[i::K] for i in range(K)]
    train_attack_bags: list = []
    train_attack_gen_ids: list = []
    for gid, uids_shard in enumerate(shards):
        if not uids_shard:
            continue
        shard_pool = dsmod.SplitPool(benign_uids=[], attack_uids=uids_shard)
        _, atk_bags = dsmod.generate_split_bags(
            shard_pool, manifest_idx, store, scma_type, attacker_models[gid], device, c_adjust
        )
        train_attack_bags.extend(atk_bags)
        train_attack_gen_ids.extend([gid] * len(atk_bags))

    synth = dsmod.adasyn_augment(train_benign, train_attack_bags, cfg.ids, scma_type, c_adjust, seed=dataset_seed)

    bags_with_gen = (
        [(b, -1) for b in train_benign]
        + list(zip(train_attack_bags, train_attack_gen_ids))
        + [(b, -1) for b in synth]
    )
    return _bags_to_multigen_arrays(bags_with_gen, scalers, seed=dataset_seed)


def fit_scalers_for_type(
    manifest: pd.DataFrame, store: dsmod.SlotStore, pools: dict, scma_type: int, primary_attacker, device, cfg: ReproductionConfig
) -> dsmod.Scalers:
    """Refits scalers on TRAIN-benign bags exactly as dataset.build_type_dataset
    does internally -- see module docstring for why this is guaranteed
    identical rather than merely 'close'."""
    manifest_idx = manifest.set_index("session_uid")
    train_pool = pools["train"]
    benign_only_pool = dsmod.SplitPool(benign_uids=train_pool.benign_uids, attack_uids=[])
    train_benign, _ = dsmod.generate_split_bags(
        benign_only_pool, manifest_idx, store, scma_type, primary_attacker, device, cfg.attacker.c_adjust
    )
    return dsmod.fit_scalers(train_benign)
