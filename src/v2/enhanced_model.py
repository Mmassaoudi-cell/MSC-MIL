"""Enhanced MSC-MIL for the v2 (curve-preserving attack) benchmark.

Contains everything needed to (re)train any enhanced candidate from a config
dict for any seed:

  * `physics_features`  -- 8 physics-consistency side features computed in torch
                            from (scaled power, mask, scaled static); they follow
                            noisy inputs because they are computed inside the model.
  * `EnhancedMSCMIL`    -- MSC-MIL (src/candidates/msc_mil.py::MSCMIL) with optional
                            physics side features concatenated to the static branch.
  * `augment_batch`     -- sensor-noise + random slot-dropout augmentation.
  * `train_enhanced`    -- training function (cfg dict -> trained model + history).
  * `predict_proba`, `noisy_copies` -- inference helpers / legacy noise protocol.

Data conventions: TypeSplitData from the v2 cache; slots (M,288,3) with scaled
power in channel 0, mask (M,288) bool, static (M,4) scaled
[conn_hod, t_end_h, e_demand_kwh, t_avail_h]. `meta` = the cache's meta__* arrays
(power_mean/power_scale/static_mean/static_scale), i.e. the TRAIN-benign scalers.

This module never loads data itself and never touches test splits.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn

from src.benchmarks.robustness import perturb_noise  # legacy noise protocol (imported, not re-implemented)
from src.candidates.msc_mil import MSCMIL
from src.candidates.common import supcon_loss
from src.reproduction.metrics import macro_f1_score
from src.reproduction.mil_ids import bce_l2_loss

DT_H = 5.0 / 60.0
N_PHYS = 8
NOISE_LEVELS = (0.05, 0.15, 0.30)

# The frozen configuration of the ORIGINAL MSC-MIL (FINAL_MODEL_CONFIG.yaml) -- the reference row.
REFERENCE_CFG = dict(
    name="MSC-MIL (original, clean-trained)", physics=False, augment=False, noise_max=0.0, drop_max=0.0,
    lr=0.0005491643958141596, dropout=0.18913682178620395, supcon_beta=0.12341988028587486,
    supcon_temperature=0.06951585991602963, l2_reg=1e-4, batch_size=32, max_epochs=30, patience=8,
    cosine=False, es_metric="clean",
)


# ---------------------------------------------------------------------------
# physics features
# ---------------------------------------------------------------------------
def physics_features(p_scaled: torch.Tensor, mask: torch.Tensor, static: torch.Tensor, c: dict) -> torch.Tensor:
    """(B,288) scaled power, (B,288) bool mask, (B,4) scaled static -> (B, 8), all O(1).

    0 energy residual  (E_curve - E_declared)/max(E_declared,1) in [-2,2]; E_curve = sum(active kW)*5min
    1 log(peak/mean) / 2         peak-to-mean ratio
    2 mean/peak                  load factor
    3 clip fraction              soft fraction of active slots at the 20 kW EVSE limit
    4 log1p(active hours) / 2    window length
    5 active hours / t_avail     window length vs the plug-in duration (clamped [0,3])
    6 plateau fraction           fraction of active slots >= 0.95*peak
    7 log1p(std/mean)            coefficient of variation
    """
    p = p_scaled * c["power_scale"] + c["power_mean"]
    mf = mask.float()
    n = mf.sum(1).clamp(min=1.0)
    s = (p * mf).sum(1)
    mean = (s / n).clamp(min=0.05)
    mx = torch.where(mask, p, torch.full_like(p, -1e9)).max(1).values
    mx = torch.where(mask.any(1), mx, torch.zeros_like(mx)).clamp(min=0.05)
    e_curve = s * DT_H
    e_decl = static[:, 2] * c["static_scale"][2] + c["static_mean"][2]
    t_av = (static[:, 3] * c["static_scale"][3] + c["static_mean"][3]).clamp(min=0.1)
    var = (((p - (s / n)[:, None]) ** 2) * mf).sum(1) / n
    std = var.clamp(min=0).sqrt()
    hrs = n * DT_H * mask.any(1).float()
    f = torch.stack([
        ((e_curve - e_decl) / e_decl.clamp(min=1.0)).clamp(-2, 2),
        torch.log((mx / mean).clamp(min=1.0, max=50.0)) / 2.0,
        (mean / mx).clamp(0, 1.5),
        (torch.sigmoid((p - 19.5) / 0.5) * mf).sum(1) / n,
        torch.log1p(hrs) / 2.0,
        (hrs / t_av).clamp(0, 3),
        (((p >= 0.95 * mx[:, None]) & mask).float().sum(1) / n),
        torch.log1p(std / mean),
    ], dim=1)
    return f


def _consts(meta: dict, device) -> dict:
    return dict(power_mean=float(meta["power_mean"][0]), power_scale=float(meta["power_scale"][0]),
                static_mean=[float(x) for x in meta["static_mean"]], static_scale=[float(x) for x in meta["static_scale"]])


class EnhancedMSCMIL(MSCMIL):
    """MSCMIL + optional physics side features on the static branch."""

    def __init__(self, meta: dict, physics: bool = False, dropout: float = 0.2, encoder_hidden=64, encoder_out=64,
                 lstm_hidden_per_dir=64, static_hidden=32):
        super().__init__(input_dim=3, encoder_hidden=encoder_hidden, encoder_out=encoder_out,
                         lstm_hidden_per_dir=lstm_hidden_per_dir, static_dim=4 + (N_PHYS if physics else 0),
                         static_hidden=static_hidden, dropout=dropout)
        self.physics = physics
        self.c = _consts(meta, "cpu")

    def _encode(self, slots, mask, static_feats):
        if self.physics:
            phys = physics_features(slots[..., 0], mask, static_feats, self.c)
            static_feats = torch.cat([static_feats, phys], dim=-1)
        return super()._encode(slots, mask, static_feats)


# ---------------------------------------------------------------------------
# augmentation (sensor noise + slot dropout), redrawn for every batch
# ---------------------------------------------------------------------------
def augment_batch(slots, mask, static, fill, p_std, e_std, noise_max, drop_max, gen):
    """Per bag: r ~ U(0, noise_max); Gaussian noise with std r*p_std on ACTIVE slot power and r*e_std on the
    static energy (scaled units; p_std/e_std = train-set std, same reference as the legacy protocol);
    q ~ U(0, drop_max); each active slot is masked with probability q (its power set to `fill`, the
    scaled inactive value)."""
    B, N = mask.shape
    dev = slots.device
    r = torch.rand(B, 1, device=dev, generator=gen) * noise_max
    q = torch.rand(B, 1, device=dev, generator=gen) * drop_max
    slots = slots.clone()
    static = static.clone()
    noise = torch.randn(B, N, device=dev, generator=gen) * r * p_std
    slots[..., 0] = slots[..., 0] + noise * mask.float()
    static[:, 2] = static[:, 2] + torch.randn(B, device=dev, generator=gen) * r[:, 0] * e_std
    if drop_max > 0:
        drop = (torch.rand(B, N, device=dev, generator=gen) < q) & mask
        mask = mask & ~drop
        slots[..., 0] = torch.where(drop, torch.full_like(slots[..., 0], fill), slots[..., 0])
    return slots, mask, static


# ---------------------------------------------------------------------------
# inference / evaluation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def _net_proba(model, d, device, batch_size=512) -> np.ndarray:
    model.eval()
    out = []
    for a in range(0, len(d.labels), batch_size):
        s = torch.from_numpy(d.slots[a:a + batch_size]).to(device)
        m = torch.from_numpy(d.mask[a:a + batch_size]).to(device)
        st = torch.from_numpy(d.static[a:a + batch_size]).to(device)
        out.append(model(s, m, st)[0].cpu().numpy())
    return np.concatenate(out)


def noise_refs(train):
    """Legacy noise reference scales (robustness.py): std of active scaled power and of scaled static energy on TRAIN."""
    return float(train.slots[:, :, 0][train.mask].std()), float(train.static[:, 2].std())


def noisy_copies(d, refs, seed_offset: int = 0, levels=NOISE_LEVELS):
    """Legacy protocol copies of `d` at each level; RNG seed = int(level*1000)*100 + seed_offset (as in robustness.py)."""
    p_std, e_std = refs
    return [perturb_noise(d, p_std, e_std, lv, np.random.default_rng(int(lv * 1000) * 100 + seed_offset)) for lv in levels]


def _f1(y, prob):
    return macro_f1_score(y, (prob >= 0.5).astype(int))


def eval_clean_noisy(model, val, noisy_sets, device):
    """-> dict(clean_f1, noisy_f1 list, noisy_mean, recall_clean)."""
    pr = predict_proba(model, val, device)
    y = val.labels
    pred = (pr >= 0.5).astype(int)
    rec = float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
    nf = [_f1(y, predict_proba(model, ns, device)) for ns in noisy_sets] if noisy_sets else []
    return dict(clean_f1=_f1(y, pr), noisy_f1=nf, noisy_mean=float(np.mean(nf)) if nf else float("nan"), recall_clean=rec)


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train_enhanced(cfg: dict, train, val, meta: dict, seed: int, device, es_noise_offset: int = 1000, verbose: bool = False):
    """Train one model. `cfg` keys: physics, augment, noise_max, drop_max, lr, dropout, supcon_beta,
    supcon_temperature, l2_reg, batch_size, max_epochs, patience, cosine, es_metric ('clean' | 'score').

    Early stopping / checkpoint selection uses VALIDATION only:
      'clean': val macro-F1 @0.5 (original protocol);
      'score': 0.5*clean val macro-F1 + 0.5*mean val macro-F1 under legacy noise at 5/15/30 %
               (fixed noise draws, RNG seeds offset by `es_noise_offset` from the seed used for reporting).
    Returns dict(model (best state loaded), history, best_epoch, best_score, converged, train_time_s)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = EnhancedMSCMIL(meta, physics=cfg["physics"], dropout=cfg["dropout"]).to(device)
    refs = noise_refs(train)
    fill = float(-meta["power_mean"][0] / meta["power_scale"][0])  # scaled value of an inactive (0 kW) slot
    gen = torch.Generator(device=device).manual_seed(seed)
    perm_gen = torch.Generator().manual_seed(seed)

    X = torch.from_numpy(train.slots).to(device)
    Mk = torch.from_numpy(train.mask).to(device)
    S = torch.from_numpy(train.static).to(device)
    Y = torch.from_numpy(train.labels).float().to(device)
    N = len(Y)
    bs = cfg["batch_size"]

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["max_epochs"], eta_min=cfg["lr"] * 0.01)
             if cfg.get("cosine") else None)
    noisy_val = noisy_copies(val, refs, seed_offset=seed + es_noise_offset) if cfg["es_metric"] == "score" else []

    best, best_state, best_epoch, no_imp, hist, converged = -1.0, None, -1, 0, [], True
    t0 = time.time()
    for ep in range(cfg["max_epochs"]):
        model.train()
        perm = torch.randperm(N, generator=perm_gen).to(device)
        tot, nb = 0.0, 0
        for a in range(0, N, bs):
            idx = perm[a:a + bs]
            s, m, st, y = X[idx], Mk[idx], S[idx], Y[idx]
            if cfg["augment"]:
                s, m, st = augment_batch(s, m, st, fill, refs[0], refs[1], cfg["noise_max"], cfg["drop_max"], gen)
            logit, _, H = model.logits_with_embedding(s, m, st)
            loss = bce_l2_loss(logit, y, model, cfg["l2_reg"]) + cfg["supcon_beta"] * supcon_loss(H, y, temperature=cfg["supcon_temperature"])
            if not torch.isfinite(loss):
                converged = False
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        if not converged:
            break
        if sched:
            sched.step()
        ev = eval_clean_noisy(model, val, noisy_val, device)
        score = ev["clean_f1"] if cfg["es_metric"] == "clean" else 0.5 * ev["clean_f1"] + 0.5 * ev["noisy_mean"]
        hist.append(dict(epoch=ep, loss=tot / max(nb, 1), es_score=score, **{k: ev[k] for k in ("clean_f1", "noisy_mean")}))
        if verbose:
            print(f"  ep{ep:02d} loss={tot/max(nb,1):.4f} clean={ev['clean_f1']:.4f} noisy={ev['noisy_mean']:.4f} es={score:.4f}", flush=True)
        if score > best:
            best, best_epoch, no_imp = score, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= cfg["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return dict(model=model, history=hist, best_epoch=best_epoch, best_score=best, converged=converged,
                train_time_s=time.time() - t0, n_params=n_params(model))


# ---------------------------------------------------------------------------
# FROZEN configuration -- ROUND 2 winner: E5 PF-Fusion (see FINAL_MODEL_CONFIG_v2.yaml).
# Round-1 winner (E4 network alone) = FROZEN_NET_CFG; it is still the network component of E5.
# ---------------------------------------------------------------------------
FROZEN_NET_CFG = dict(
    name="E4 MSC-MIL-Long", physics=False, augment=True, noise_max=0.3, drop_max=0.25,
    lr=0.0029022182175632034, dropout=0.2607637796221153, supcon_beta=0.13088807147306578,
    supcon_temperature=0.13083402373105277, l2_reg=1e-4, batch_size=32, max_epochs=60, patience=12,
    cosine=True, es_metric="score",
)
FROZEN_TREE_PARAMS = dict(n_estimators=350, max_depth=7, learning_rate=0.0815241575370242, subsample=0.9718740392573121,
                          colsample_bytree=0.8409101495517417, reg_lambda=0.11981845126013872, min_child_weight=5)
FROZEN_W_TREE = 0.2          # fusion weight on the tree probability; chosen on SCMA-1 validation (seed 0)
TREE_AUG_COPIES = 4          # tree training set = clean + 4 augmented copies (noise_max 0.3, drop_max 0.25)
FROZEN_CFG = dict(name="E5 PF-Fusion", net=FROZEN_NET_CFG, tree_params=FROZEN_TREE_PARAMS, w_tree=FROZEN_W_TREE,
                  tree_aug_copies=TREE_AUG_COPIES, alert_threshold=0.5)


def tree_features(slots: np.ndarray, mask: np.ndarray, static: np.ndarray, meta: dict) -> np.ndarray:
    """15-dim tree features: 7-dim summary (src.v2.common.summary_features) + 8 physics features, from (possibly noisy) tensors."""
    from types import SimpleNamespace
    from src.v2.common import summary_features
    X = summary_features(SimpleNamespace(slots=slots, mask=mask, static=static, labels=np.zeros(len(mask))))
    with torch.no_grad():
        ph = physics_features(torch.from_numpy(slots[..., 0]), torch.from_numpy(mask), torch.from_numpy(static), _consts(meta, "cpu")).numpy()
    return np.hstack([X, ph])


def fit_frozen_tree(train, meta: dict, seed: int, params: dict | None = None, copies: int = TREE_AUG_COPIES):
    """XGBoost-PF trained on clean + `copies` augmented copies of TRAIN (augment_batch with a CPU generator seeded by `seed`)."""
    from xgboost import XGBClassifier
    refs = noise_refs(train)
    fill = float(-meta["power_mean"][0] / meta["power_scale"][0])
    gen = torch.Generator().manual_seed(seed)
    S, M, ST = torch.from_numpy(train.slots), torch.from_numpy(train.mask), torch.from_numpy(train.static)
    Xs = [tree_features(train.slots, train.mask, train.static, meta)]
    for _ in range(copies):
        s, m, st = augment_batch(S, M, ST, fill, refs[0], refs[1], 0.3, 0.25, gen)
        Xs.append(tree_features(s.numpy(), m.numpy(), st.numpy(), meta))
    X = np.vstack(Xs)
    y = np.tile(train.labels, copies + 1)
    pos = y.sum()
    est = XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=8, tree_method="hist",
                        scale_pos_weight=float((len(y) - pos) / max(pos, 1)), **(FROZEN_TREE_PARAMS if params is None else params))
    est.fit(X, y)
    return est


class FusedDetector:
    """p = w_tree * p_tree + (1 - w_tree) * p_net; alert if p >= 0.5. Features / physics are recomputed from whatever input is passed."""

    def __init__(self, net, tree, meta, w_tree=FROZEN_W_TREE):
        self.net, self.tree, self.meta, self.w_tree = net, tree, meta, w_tree

    def predict_proba(self, d, device, batch_size=512) -> np.ndarray:
        p_net = _net_proba(self.net, d, device, batch_size)
        p_tree = self.tree.predict_proba(tree_features(d.slots, d.mask, d.static, self.meta))[:, 1]
        return self.w_tree * p_tree + (1.0 - self.w_tree) * p_net


def predict_proba(model, d, device, batch_size=512) -> np.ndarray:
    """Works for a FusedDetector (frozen model) and for a bare network (component / earlier candidates)."""
    if isinstance(model, FusedDetector):
        return model.predict_proba(d, device, batch_size)
    return _net_proba(model, d, device, batch_size)


def train_frozen(train, val, meta: dict, seed: int, device, cfg: dict | None = None):
    """Retrain the frozen E5 model for any seed. train/val = TypeSplitData of the v2 builder, meta = cache meta__* arrays.
    Network: E4 recipe via train_enhanced(seed) (val used for the same score-based early stopping as in selection).
    Tree: fit_frozen_tree(seed). Returns dict(model=FusedDetector, net_result=train_enhanced output, n_params_net, ...).
    Use: p = predict_proba(result['model'], split, device); alert = p >= 0.5."""
    c = FROZEN_CFG if cfg is None else cfg
    nr = train_enhanced(dict(c["net"]), train, val, meta, seed, device)
    tree = fit_frozen_tree(train, meta, seed, c["tree_params"], c["tree_aug_copies"])
    fd = FusedDetector(nr["model"], tree, meta, c["w_tree"])
    return dict(model=fd, net_result=nr, tree=tree, n_params_net=nr["n_params"], converged=nr["converged"])
