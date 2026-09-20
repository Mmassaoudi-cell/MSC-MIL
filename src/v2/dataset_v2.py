"""v2 dataset builder: same aggregate attack logic and frozen PPO attacker as v1,
but attacked bags are the session's REAL per-slot curve transformed (rescaled /
truncated / shifted) instead of a flat-rate reconstruction, and train-set
augmentation uses several stochastic-C_attack variants instead of ADASYN.

Public API
    build_type_dataset_v2(scma_type, use_cache=True, rebuild=False) -> dict
        keys 'train','val','test_full','test_n200' -> dsmod.TypeSplitData
        (.slots (M,288,3) .mask (M,288) .static (M,4) .labels (M,) .uids)
    summary_features_v2(d) -> (M,7) tree features (v1-compatible, see common.summary_features)
    load_extras(scma_type) -> dict split -> dict of per-bag diagnostic arrays (raw units)
    build_all() builds/caches all six types.

Design record (also in results/v2/V2_DATA_REPORT.md)
  * Split: unchanged DATA_SPLIT_MANIFEST.csv; benign sessions identical to v1
    (train 4000, val 800, test_full all 7945, test_n200 the same 180).
  * Train attack sources: the SAME 444 source sessions as v1. Each source gets
    TRAIN_VARIANTS=9 attacked variants: variant 0 uses the policy-mean
    C_attack (as v1), variants 1..8 use C ~ N(mu_i, exp(log_std)^2) drawn from
    the frozen attacker's stochastic action distribution (seed TRAIN_C_SEED).
    444*9 = 3996 positives + 4000 benign = 7996 bags (49.98% positive; v1: 8028, 50.2%).
    No ADASYN.
  * Val/test attack sources are re-drawn with NEW seeds (VAL_ATTACK_SEED,
    N200_ATTACK_SEED, FULL_ATTACK_SEED), keeping 90/10: val 800+89, test_n200
    180+20, test_full 7945+883. C_attack = policy mean (as v1 deployment).
  * Scalers (power, static) are fit on TRAIN BENIGN bags only (same as v1);
    v1 code (`dsmod.fit_scalers`, `bag_to_tensors`) is reused unchanged.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from src.v2.common import DATA_V2, PROJECT_ROOT, dsmod, load_attacker, summary_features  # noqa: F401
from src.v2.attack_realize import realize_attacked_curve
from src.reproduction.scma_env import generate_scma

TRAIN_VARIANTS = 9
TRAIN_C_SEED = 71000
VAL_ATTACK_SEED = 20260101
N200_ATTACK_SEED = 20260102
FULL_ATTACK_SEED = 20260103
SPLITS = ("train", "val", "test_full", "test_n200")
SCMA_TYPES = (1, 2, 3, 4, 5, 6)
EXTRA_FIELDS = ["src_uid", "variant", "c_attack", "scale", "e_target", "e_new", "clipped", "corr",
                "n_orig", "n_new", "e_full", "e_orig_kwh", "raw_summary"]

_CTX: dict = {}


def cache_path(scma_type: int) -> Path:
    return DATA_V2 / f"scma{scma_type}_v2.npz"


def summary_features_v2(d) -> np.ndarray:
    """(M,7) summary features for tree models (same definition as v1 flatten_features)."""
    return summary_features(d)


def _ctx(device="cpu"):
    if not _CTX:
        attacker, cfg = load_attacker(device)
        manifest = dsmod.load_manifest()
        store = dsmod.SlotStore.get()
        pools = dsmod.build_split_pools(manifest, seed=0)
        tv = dsmod.build_test_variants(manifest, seed=0)
        log_std = float(torch.exp(attacker.pi_log_std).item())
        _CTX.update(attacker=attacker, cfg=cfg, manifest=manifest, mi=manifest.set_index("session_uid"),
                    store=store, pools=pools, tv=tv, policy_std=log_std, device=device)
    return _CTX


def attack_source_uids(split: str) -> list[str]:
    """Attack-source sessions per split. Train = v1's; val/test = re-drawn with new seeds."""
    c = _ctx()
    m = c["manifest"]
    if split == "train":
        return list(c["pools"]["train"].attack_uids)
    if split == "val":
        sub = m[m.split == "val"]
        n = dsmod._attack_count(len(c["pools"]["val"].benign_uids))
        idx = np.random.default_rng(VAL_ATTACK_SEED).choice(sub.index.to_numpy(), size=n, replace=False)
    elif split == "test_full":
        sub = m[m.split == "test"]
        n = dsmod._attack_count(len(sub))
        idx = np.random.default_rng(FULL_ATTACK_SEED).choice(sub.index.to_numpy(), size=min(n, len(sub)), replace=False)
    elif split == "test_n200":
        sub = m[m.split == "test"]
        ben = set(c["tv"]["test_n200"].benign_uids)
        rest = sub[~sub.session_uid.isin(ben)].index.to_numpy()
        idx = np.random.default_rng(N200_ATTACK_SEED).choice(rest, size=dsmod.TEST_N200_ATTACK, replace=False)
    else:
        raise ValueError(split)
    return m.loc[idx, "session_uid"].tolist()


def benign_uids(split: str) -> list[str]:
    c = _ctx()
    if split in ("train", "val"):
        return list(c["pools"][split].benign_uids)
    return list(c["tv"][split].benign_uids)


def _sessions(uids):
    c = _ctx()
    rows = list(c["mi"].loc[uids].itertuples(index=False))
    return rows, [dsmod.session_from_summary(r.connection_time, r.disconnection_time, r.kwh_delivered) for r in rows]


def _benign_bags(split):
    c = _ctx()
    uids = benign_uids(split)
    rows = list(c["mi"].loc[uids].itertuples(index=False))
    return [dsmod.make_benign_bag(u, r, c["store"]) for u, r in zip(uids, rows)]


def _attack_bags(split, scma_type):
    """Returns (bags, extras list-of-dicts)."""
    c = _ctx()
    uids = attack_source_uids(split)
    rows, sess = _sessions(uids)
    mu = dsmod.predict_c_attack(sess, c["attacker"], c["device"])
    n_var = TRAIN_VARIANTS if split == "train" else 1
    rng = np.random.default_rng(TRAIN_C_SEED)
    noise = rng.standard_normal((n_var, len(uids)))  # same draws for every SCMA type
    c_adj = c["cfg"].attacker.c_adjust
    bags, ext = [], []
    for k in range(n_var):
        c_k = mu if k == 0 else mu + c["policy_std"] * noise[k]
        for u, r, s, ck in zip(uids, rows, sess, c_k):
            att = generate_scma(s, scma_type, float(ck), c_adj)
            rp, ra = c["store"].get_slots(u)
            power, active, info = realize_attacked_curve(rp, ra, s, att)
            bag = dsmod.Bag(slots=power, active=active, sess=att, connection_time=r.connection_time,
                            label=1, session_uid=f"{u}::scma{scma_type}::v{k}")
            bags.append(bag)
            info.update(src_uid=u, variant=k, c_attack=float(ck), e_orig_kwh=float(s.e_demand_kwh))
            ext.append(info)
    return bags, ext


def _extras_arrays(bags, ext_list, n_benign):
    """Per-bag arrays aligned with the split arrays (benign first, then attacks)."""
    M = len(bags)
    e = {"src_uid": np.array([b.session_uid.split("::")[0] for b in bags]),
         "variant": np.full(M, -1, dtype=np.int16)}
    for f in ("c_attack", "scale", "e_target", "e_new", "corr", "e_full", "e_orig_kwh", "n_orig", "n_new"):
        e[f] = np.full(M, np.nan, dtype=np.float64)
    e["clipped"] = np.zeros(M, dtype=bool)
    for j, info in enumerate(ext_list):
        i = n_benign + j
        e["variant"][i] = info["variant"]
        for f in ("c_attack", "scale", "e_target", "e_new", "corr", "e_full", "e_orig_kwh", "n_orig", "n_new"):
            e[f][i] = info[f]
        e["clipped"][i] = info["clipped"]
    e["raw_summary"] = np.stack([dsmod.summary_vec(b.sess, b.slots, b.active) for b in bags]).astype(np.float32)
    return e


def _build(scma_type: int):
    _ctx()
    t0 = time.time()
    per_split = {}
    for sp in SPLITS:
        ben = _benign_bags(sp)
        att, ext = _attack_bags(sp, scma_type)
        per_split[sp] = (ben, att, ext)
    scalers = dsmod.fit_scalers(per_split["train"][0])
    out, extras = {}, {}
    for sp in SPLITS:
        ben, att, ext = per_split[sp]
        bags = ben + att
        out[sp] = dsmod._bags_to_arrays(bags, scalers, seed=0)
        extras[sp] = _extras_arrays(bags, ext, len(ben))
    meta = dict(power_mean=scalers.power.mean_, power_scale=scalers.power.scale_,
                static_mean=scalers.static.mean_, static_scale=scalers.static.scale_)
    print(f"[dataset_v2] built SCMA-{scma_type} in {time.time()-t0:.1f}s: "
          + ", ".join(f"{sp} n={len(out[sp].labels)} pos={out[sp].labels.mean():.3f}" for sp in SPLITS))
    return out, extras, meta


def _save(scma_type, out, extras, meta):
    DATA_V2.mkdir(parents=True, exist_ok=True)
    arrs = {}
    for sp in SPLITS:
        d = out[sp]
        arrs[f"{sp}__slots"] = d.slots
        arrs[f"{sp}__mask"] = d.mask
        arrs[f"{sp}__static"] = d.static
        arrs[f"{sp}__labels"] = d.labels
        arrs[f"{sp}__uids"] = np.array(d.uids)
        for k, v in extras[sp].items():
            arrs[f"{sp}__x_{k}"] = v
    for k, v in meta.items():
        arrs[f"meta__{k}"] = v
    tmp = cache_path(scma_type).with_suffix(".tmp.npz")
    np.savez(tmp, **arrs)
    tmp.replace(cache_path(scma_type))


def _load(scma_type):
    z = np.load(cache_path(scma_type), allow_pickle=False)
    out, extras = {}, {}
    for sp in SPLITS:
        out[sp] = dsmod.TypeSplitData(slots=z[f"{sp}__slots"], mask=z[f"{sp}__mask"], static=z[f"{sp}__static"],
                                      labels=z[f"{sp}__labels"], uids=[str(u) for u in z[f"{sp}__uids"]])
        extras[sp] = {k: z[f"{sp}__x_{k}"] for k in EXTRA_FIELDS}
    return out, extras


def build_type_dataset_v2(scma_type: int, use_cache: bool = True, rebuild: bool = False) -> dict:
    """Same return structure as dataset.build_type_dataset: dict split -> TypeSplitData."""
    if use_cache and not rebuild and cache_path(scma_type).exists():
        return _load(scma_type)[0]
    out, extras, meta = _build(scma_type)
    _save(scma_type, out, extras, meta)
    return out


def load_extras(scma_type: int) -> dict:
    """Per-split dict of diagnostic arrays (raw_summary (M,7) in raw kWh/h/kW units, c_attack, scale, e_target, e_new, clipped, corr, ...)."""
    if not cache_path(scma_type).exists():
        build_type_dataset_v2(scma_type)
    return _load(scma_type)[1]


def build_all(rebuild: bool = False):
    for t in SCMA_TYPES:
        build_type_dataset_v2(t, rebuild=rebuild)


def _self_test():
    print("[self-test] loading/building SCMA-1 ...")
    d = build_type_dataset_v2(1)
    for sp in SPLITS:
        x = d[sp]
        M = len(x.labels)
        assert x.slots.shape == (M, 288, 3) and x.mask.shape == (M, 288) and x.static.shape == (M, 4)
        assert len(x.uids) == M and set(np.unique(x.labels)) <= {0, 1}
        assert np.isfinite(x.slots).all() and np.isfinite(x.static).all()
        assert summary_features_v2(x).shape == (M, 7)
    assert abs(d["train"].labels.mean() - 0.5) < 0.01
    assert abs(d["val"].labels.mean() - 0.1) < 0.005 and abs(d["test_full"].labels.mean() - 0.1) < 0.005
    assert (d["test_full"].labels == 0).sum() == 7945
    assert len(d["test_n200"].labels) == 200
    # inactive slots carry the scaled-zero value only
    ex = load_extras(1)
    rs = ex["val"]["raw_summary"]
    assert rs.shape == (len(d["val"].labels), 7)
    # summary_features must match v1's flatten_features semantics on v2 data
    from src.benchmarks.data_utils import flatten_features
    assert np.allclose(flatten_features(d["val"]), summary_features_v2(d["val"]), atol=1e-6)
    print("[self-test] OK")


if __name__ == "__main__":
    import sys
    if "--build-all" in sys.argv:
        build_all(rebuild="--rebuild" in sys.argv)
    _self_test()
