"""PI-Fusion: physics-informed fusion for weakly supervised detection of stealthy
charge-manipulation attacks in EV charging sessions.

PI-Fusion combines two branches that observe the same charging session:

* a noise-augmented multi-scale contrastive attention MIL network (NA-MSC-MIL)
  that reads the full 288-slot power curve (three temporal scales, BiLSTM +
  attention pooling, supervised-contrastive regularisation), and
* an XGBoost classifier on 15 session-level features, eight of which are
  physics-informed consistency checks between declared and measured quantities.

The two probabilities are fused as ``p = w * p_tree + (1 - w) * p_net`` with
``w = 0.2``; an alert is raised when ``p >= 0.5``.

Input convention (one session = one bag of 288 five-minute slots)::

    slots   float32 (M, 288, 3)  [standardised power, sin(hour), cos(hour)]
    mask    bool    (M, 288)     True for slots in which the vehicle is charging
    static  float32 (M, 4)       standardised [connection hour, end time (h),
                                 declared energy (kWh), available time (h)]
    labels  int     (M,)         0 = benign, 1 = manipulated (session level)

Power and static inputs are standardised with a :class:`Calibration` fitted on
benign training sessions; the physics features are computed on the original
(kW, kWh, h) scale by inverting that calibration.

Public API: :class:`Bags`, :class:`Calibration`, :class:`PIFusion`,
:func:`macro_f1`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

__all__ = ["Bags", "Calibration", "PIFusion", "macro_f1"]

N_SLOTS = 288
DT_H = 5.0 / 60.0                  # slot length in hours
N_PHYS = 8                         # number of physics-informed features
POOL_SIZES = (1, 3, 12)            # slots per token: 5 min, 15 min, 60 min
NOISE_LEVELS = (0.05, 0.15, 0.30)  # stress-noise levels used for model selection

# Network and training hyper-parameters.
NET_CFG = dict(
    noise_max=0.3, drop_max=0.25, lr=0.0029022182175632034, dropout=0.2607637796221153,
    supcon_beta=0.13088807147306578, supcon_temperature=0.13083402373105277, l2_reg=1e-4,
    batch_size=32, max_epochs=60, patience=12, encoder_hidden=64, encoder_out=64,
    lstm_hidden_per_dir=64, static_hidden=32,
)
# XGBoost hyper-parameters (tree branch).
TREE_PARAMS = dict(
    n_estimators=350, max_depth=7, learning_rate=0.0815241575370242, subsample=0.9718740392573121,
    colsample_bytree=0.8409101495517417, reg_lambda=0.11981845126013872, min_child_weight=5,
)
W_TREE = 0.2          # fusion weight of the tree probability
TREE_AUG_COPIES = 4   # tree training set = clean data + 4 noise-augmented copies
THRESHOLD = 0.5       # alert threshold on the fused probability


# ---------------------------------------------------------------------------
# data containers and calibration
# ---------------------------------------------------------------------------
@dataclass
class Bags:
    """A set of charging sessions in the input convention described above."""
    slots: np.ndarray
    mask: np.ndarray
    static: np.ndarray
    labels: np.ndarray | None = None

    @classmethod
    def from_npz(cls, path, split: str) -> "Bags":
        """Read ``<split>__slots/mask/static/labels`` arrays from an ``.npz`` file."""
        z = np.load(path, allow_pickle=False)
        labels = z[f"{split}__labels"] if f"{split}__labels" in z.files else None
        return cls(z[f"{split}__slots"], z[f"{split}__mask"], z[f"{split}__static"], labels)

    def __len__(self) -> int:
        return len(self.mask)


@dataclass
class Calibration:
    """Input standardisation fitted on benign training sessions.

    ``power_*`` standardises active-slot power (kW); ``static_*`` standardises the
    four session-level fields [connection hour, end time (h), energy (kWh),
    available time (h)].
    """
    power_mean: float
    power_scale: float
    static_mean: np.ndarray
    static_scale: np.ndarray

    @classmethod
    def fit(cls, active_power_kw: np.ndarray, static_raw: np.ndarray) -> "Calibration":
        """``active_power_kw``: powers of all active slots of the training sessions;
        ``static_raw``: (n, 4) raw static fields of the benign training sessions."""
        p = np.asarray(active_power_kw, dtype=np.float64).reshape(-1)
        s = np.asarray(static_raw, dtype=np.float64)
        scale = s.std(0)
        return cls(float(p.mean()), float(p.std()), s.mean(0), np.where(scale == 0, 1.0, scale))

    @classmethod
    def from_arrays(cls, power_mean, power_scale, static_mean, static_scale) -> "Calibration":
        return cls(float(np.ravel(power_mean)[0]), float(np.ravel(power_scale)[0]),
                   np.asarray(static_mean, dtype=np.float64), np.asarray(static_scale, dtype=np.float64))

    def standardise(self, power_kw: np.ndarray, static_raw: np.ndarray):
        """Return (standardised power, standardised static) as float32 arrays."""
        return ((np.asarray(power_kw) - self.power_mean) / self.power_scale).astype(np.float32), \
               ((np.asarray(static_raw) - self.static_mean) / self.static_scale).astype(np.float32)

    def _consts(self) -> dict:
        return dict(power_mean=float(self.power_mean), power_scale=float(self.power_scale),
                    static_mean=[float(x) for x in self.static_mean], static_scale=[float(x) for x in self.static_scale])


def macro_f1(y_true, y_pred) -> float:
    """Mean of the F1 scores of the benign and the manipulated class."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# physics-informed consistency features
# ---------------------------------------------------------------------------
def physics_features(p_scaled: torch.Tensor, mask: torch.Tensor, static: torch.Tensor, c: dict) -> torch.Tensor:
    """Eight consistency features, shape (B, 8), from standardised power (B, 288),
    the active-slot mask (B, 288) and standardised static fields (B, 4).

    0  energy residual (E_curve - E_declared) / max(E_declared, 1), clipped to [-2, 2]
    1  log(peak / mean) / 2                     peakiness
    2  mean / peak                              load factor
    3  soft fraction of active slots at 20 kW   EVSE clipping
    4  log(1 + active hours) / 2                window length
    5  active hours / available time (<= 3)    window vs. plug-in duration
    6  fraction of active slots >= 0.95 * peak  plateau share
    7  log(1 + std / mean)                      variability
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
    return torch.stack([
        ((e_curve - e_decl) / e_decl.clamp(min=1.0)).clamp(-2, 2),
        torch.log((mx / mean).clamp(min=1.0, max=50.0)) / 2.0,
        (mean / mx).clamp(0, 1.5),
        (torch.sigmoid((p - 19.5) / 0.5) * mf).sum(1) / n,
        torch.log1p(hrs) / 2.0,
        (hrs / t_av).clamp(0, 3),
        (((p >= 0.95 * mx[:, None]) & mask).float().sum(1) / n),
        torch.log1p(std / mean),
    ], dim=1)


def _summary_features(slots: np.ndarray, mask: np.ndarray, static: np.ndarray) -> np.ndarray:
    """Seven session summaries: declared energy, available time, connection hour,
    end time, and mean / std / max of active-slot power (standardised units)."""
    x = slots[:, :, 0].astype(np.float64)
    m = mask
    n = m.sum(1)
    mean = (x * m).sum(1) / np.maximum(n, 1)
    std = np.sqrt((((x - mean[:, None]) ** 2) * m).sum(1) / np.maximum(n, 1))
    mx = np.where(m, x, -np.inf).max(1)
    empty = n == 0
    mean[empty] = 0.0
    std[empty] = 0.0
    mx[empty] = 0.0
    out = np.zeros((len(mask), 7), dtype=np.float64)
    out[:, 0], out[:, 1], out[:, 2], out[:, 3] = static[:, 2], static[:, 3], static[:, 0], static[:, 1]
    out[:, 4], out[:, 5], out[:, 6] = mean, std, mx
    return out


def tree_features(slots: np.ndarray, mask: np.ndarray, static: np.ndarray, calib: Calibration) -> np.ndarray:
    """15-dimensional tree input: 7 summary features + 8 physics features."""
    summary = _summary_features(slots, mask, static)
    with torch.no_grad():
        phys = physics_features(torch.from_numpy(slots[..., 0]), torch.from_numpy(mask),
                                torch.from_numpy(static), calib._consts()).numpy()
    return np.hstack([summary, phys])


# ---------------------------------------------------------------------------
# network branch: multi-scale contrastive attention MIL
# ---------------------------------------------------------------------------
class _InstanceEncoder(nn.Module):
    """Shared per-slot MLP."""

    def __init__(self, input_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim), nn.ReLU())

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.net(r)


class _AttentionPooling(nn.Module):
    """Soft attention over active tokens."""

    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Linear(dim, 1, bias=False)

    def forward(self, z: torch.Tensor, mask: torch.Tensor):
        scores = self.w(z).squeeze(-1).masked_fill(~mask, float("-inf"))
        alpha = torch.nan_to_num(torch.softmax(scores, dim=1), nan=0.0)
        return torch.einsum("bn,bnd->bd", alpha, z), alpha


class _ScaleBranch(nn.Module):
    """Average-pool to ``pool_size`` slots per token -> BiLSTM -> attention pooling."""

    def __init__(self, encoder_out: int, hidden_per_dir: int, pool_size: int, dropout: float):
        super().__init__()
        self.pool_size = pool_size
        self.bilstm = nn.LSTM(encoder_out, hidden_per_dir, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.attn = _AttentionPooling(2 * hidden_per_dir)

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        b, n, d = h.shape
        k = n // self.pool_size
        h_p = h.view(b, k, self.pool_size, d).mean(dim=2)
        m_p = mask.view(b, k, self.pool_size).any(dim=2)   # a token is active if any of its slots is
        z, _ = self.bilstm(h_p)
        return self.attn(self.dropout(z), m_p)


class MSCMIL(nn.Module):
    """Multi-scale contrastive MIL network: three BiLSTM + attention branches over the
    slot encoding, concatenated with a projection of the static session fields."""

    def __init__(self, dropout: float, encoder_hidden=64, encoder_out=64, lstm_hidden_per_dir=64,
                 static_dim=4, static_hidden=32, input_dim=3):
        super().__init__()
        self.encoder = _InstanceEncoder(input_dim, encoder_hidden, encoder_out)
        self.branches = nn.ModuleList(
            [_ScaleBranch(encoder_out, lstm_hidden_per_dir, s, dropout) for s in POOL_SIZES])
        self.static_proj = nn.Sequential(nn.Linear(static_dim, static_hidden), nn.ReLU())
        self.classifier = nn.Linear(2 * lstm_hidden_per_dir * len(POOL_SIZES) + static_hidden, 1)

    def _embed(self, slots, mask, static):
        h = self.encoder(slots)
        parts = [branch(h, mask)[0] for branch in self.branches]
        return torch.cat(parts + [self.static_proj(static)], dim=-1)

    def forward(self, slots, mask, static):
        """Returns the manipulation probability (B,)."""
        return torch.sigmoid(self.classifier(self._embed(slots, mask, static)).squeeze(-1))

    def logits_with_embedding(self, slots, mask, static):
        H = self._embed(slots, mask, static)
        return self.classifier(H).squeeze(-1), H


def _supcon_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """Supervised contrastive loss (benign vs. manipulated) on L2-normalised embeddings."""
    b = z.shape[0]
    if b < 2:
        return z.new_zeros(())
    z = F.normalize(z, dim=-1)
    sim = torch.matmul(z, z.T) / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    not_self = ~torch.eye(b, dtype=torch.bool, device=z.device)
    exp_sim = torch.exp(sim) * not_self
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
    lab = labels.view(-1, 1)
    pos = (lab == lab.T) & not_self
    cnt = pos.sum(dim=1)
    valid = cnt > 0
    if valid.sum() == 0:
        return z.new_zeros(())
    return -((pos.float() * log_prob).sum(dim=1)[valid] / cnt[valid].float()).mean()


def _bce_l2_loss(logits, labels, model: nn.Module, l2_reg: float) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, labels.float())
    return bce + l2_reg * sum((p ** 2).sum() for p in model.parameters())


# ---------------------------------------------------------------------------
# noise augmentation and stress-noise copies
# ---------------------------------------------------------------------------
def _augment(slots, mask, static, fill, p_std, e_std, noise_max, drop_max, gen):
    """Per bag: r ~ U(0, noise_max); Gaussian noise with std r * p_std on active-slot power and
    r * e_std on declared energy; q ~ U(0, drop_max); each active slot is masked with probability q."""
    b, n = mask.shape
    dev = slots.device
    r = torch.rand(b, 1, device=dev, generator=gen) * noise_max
    q = torch.rand(b, 1, device=dev, generator=gen) * drop_max
    slots = slots.clone()
    static = static.clone()
    noise = torch.randn(b, n, device=dev, generator=gen) * r * p_std
    slots[..., 0] = slots[..., 0] + noise * mask.float()
    static[:, 2] = static[:, 2] + torch.randn(b, device=dev, generator=gen) * r[:, 0] * e_std
    if drop_max > 0:
        drop = (torch.rand(b, n, device=dev, generator=gen) < q) & mask
        mask = mask & ~drop
        slots[..., 0] = torch.where(drop, torch.full_like(slots[..., 0], fill), slots[..., 0])
    return slots, mask, static


def _noise_refs(train: Bags):
    """Reference scales for noise: std of active standardised power and of standardised energy."""
    return float(train.slots[:, :, 0][train.mask].std()), float(train.static[:, 2].std())


def _noisy_copy(d: Bags, p_std: float, e_std: float, level: float, rng: np.random.Generator) -> Bags:
    """Additive Gaussian sensor noise of ``level`` times the reference std on active-slot power and energy."""
    d2 = Bags(d.slots.copy(), d.mask.copy(), d.static.copy(), d.labels)
    noise_p = rng.normal(0.0, level * p_std, size=d2.slots[:, :, 0].shape).astype(np.float32)
    d2.slots[:, :, 0] = d2.slots[:, :, 0] + noise_p * d2.mask
    noise_e = rng.normal(0.0, level * e_std, size=d2.static[:, 2].shape).astype(np.float32)
    d2.static[:, 2] = d2.static[:, 2] + noise_e
    return d2


def _noisy_copies(d: Bags, refs, seed_offset: int, levels=NOISE_LEVELS):
    return [_noisy_copy(d, refs[0], refs[1], lv, np.random.default_rng(int(lv * 1000) * 100 + seed_offset))
            for lv in levels]


@torch.no_grad()
def _net_proba(net: nn.Module, d: Bags, device, batch_size: int = 512) -> np.ndarray:
    net.eval()
    out = []
    for a in range(0, len(d), batch_size):
        out.append(net(torch.from_numpy(d.slots[a:a + batch_size]).to(device),
                       torch.from_numpy(d.mask[a:a + batch_size]).to(device),
                       torch.from_numpy(d.static[a:a + batch_size]).to(device)).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# the fused detector
# ---------------------------------------------------------------------------
@dataclass
class PIFusion:
    """Physics-informed fusion detector.

    >>> model = PIFusion(device="cuda").fit(train, val, calibration, seed=0)
    >>> scores = model.predict_proba(test)      # fused manipulation probability
    >>> alerts = model.predict(test)            # scores >= 0.5
    """
    device: str | None = None
    w_tree: float = W_TREE
    net_cfg: dict = field(default_factory=lambda: dict(NET_CFG))
    tree_params: dict = field(default_factory=lambda: dict(TREE_PARAMS))
    net: MSCMIL | None = None
    tree: XGBClassifier | None = None
    calibration: Calibration | None = None
    noise_refs: tuple | None = None   # (power std, energy std) of the training set, the noise reference
    best_epoch: int = -1
    converged: bool = True

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- fitting ----------------------------------------------------------
    def fit(self, train: Bags, val: Bags, calibration: Calibration, seed: int = 0, verbose: bool = False) -> "PIFusion":
        """Train the network branch and the tree branch. ``val`` selects the network checkpoint."""
        self.calibration = calibration
        self._fit_net(train, val, calibration, seed, verbose)
        self._fit_tree(train, calibration, seed)
        return self

    def _val_score(self, d: Bags, noisy: list) -> tuple[float, float, float]:
        """(selection score, clean macro-F1, mean stressed macro-F1) of the current network."""
        clean = macro_f1(d.labels, (_net_proba(self.net, d, self.device) >= THRESHOLD).astype(int))
        stress = [macro_f1(d.labels, (_net_proba(self.net, ns, self.device) >= THRESHOLD).astype(int)) for ns in noisy]
        mean_stress = float(np.mean(stress))
        return 0.5 * clean + 0.5 * mean_stress, clean, mean_stress

    def _fit_net(self, train: Bags, val: Bags, calib: Calibration, seed: int, verbose: bool):
        """Noise-augmented training. The checkpoint with the best validation score
        (0.5 * clean macro-F1 + 0.5 * mean macro-F1 under 5/15/30 % noise) is kept."""
        cfg = self.net_cfg
        dev = self.device
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.net = MSCMIL(cfg["dropout"], cfg["encoder_hidden"], cfg["encoder_out"],
                          cfg["lstm_hidden_per_dir"], 4, cfg["static_hidden"]).to(dev)
        refs = self.noise_refs = _noise_refs(train)
        fill = float(-calib.power_mean / calib.power_scale)   # standardised value of an inactive (0 kW) slot
        gen = torch.Generator(device=dev).manual_seed(seed)
        perm_gen = torch.Generator().manual_seed(seed)

        X = torch.from_numpy(train.slots).to(dev)
        Mk = torch.from_numpy(train.mask).to(dev)
        S = torch.from_numpy(train.static).to(dev)
        Y = torch.from_numpy(train.labels).float().to(dev)
        n, bs = len(Y), cfg["batch_size"]

        opt = torch.optim.Adam(self.net.parameters(), lr=cfg["lr"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["max_epochs"], eta_min=cfg["lr"] * 0.01)
        noisy_val = _noisy_copies(val, refs, seed_offset=seed + 1000)

        best, best_state, no_imp = -1.0, None, 0
        self.best_epoch, self.converged = -1, True
        for ep in range(cfg["max_epochs"]):
            self.net.train()
            perm = torch.randperm(n, generator=perm_gen).to(dev)
            tot, nb = 0.0, 0
            for a in range(0, n, bs):
                idx = perm[a:a + bs]
                s, m, st, y = X[idx], Mk[idx], S[idx], Y[idx]
                s, m, st = _augment(s, m, st, fill, refs[0], refs[1], cfg["noise_max"], cfg["drop_max"], gen)
                logit, H = self.net.logits_with_embedding(s, m, st)
                loss = _bce_l2_loss(logit, y, self.net, cfg["l2_reg"]) \
                    + cfg["supcon_beta"] * _supcon_loss(H, y, cfg["supcon_temperature"])
                if not torch.isfinite(loss):
                    self.converged = False
                    break
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss.item())
                nb += 1
            if not self.converged:
                break
            sched.step()
            score, clean, stress = self._val_score(val, noisy_val)
            if verbose:
                print(f"epoch {ep:02d} loss={tot / max(nb, 1):.4f} clean={clean:.4f} stress={stress:.4f} score={score:.4f}", flush=True)
            if score > best:
                best, self.best_epoch, no_imp = score, ep, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                no_imp += 1
                if no_imp >= cfg["patience"]:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)

    def _fit_tree(self, train: Bags, calib: Calibration, seed: int):
        """XGBoost on clean training data plus four noise-augmented copies, with the class
        ratio of the augmented set as positive-class weight."""
        cfg = self.net_cfg
        refs = _noise_refs(train)
        fill = float(-calib.power_mean / calib.power_scale)
        gen = torch.Generator().manual_seed(seed)
        S, M, ST = torch.from_numpy(train.slots), torch.from_numpy(train.mask), torch.from_numpy(train.static)
        feats = [tree_features(train.slots, train.mask, train.static, calib)]
        for _ in range(TREE_AUG_COPIES):
            s, m, st = _augment(S, M, ST, fill, refs[0], refs[1], cfg["noise_max"], cfg["drop_max"], gen)
            feats.append(tree_features(s.numpy(), m.numpy(), st.numpy(), calib))
        X = np.vstack(feats)
        y = np.tile(train.labels, TREE_AUG_COPIES + 1)
        pos = y.sum()
        self.tree = XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=8, tree_method="hist",
                                  scale_pos_weight=float((len(y) - pos) / max(pos, 1)), **self.tree_params)
        self.tree.fit(X, y)

    # -- calibration of the fusion weight ---------------------------------
    def calibrate_weight(self, val: Bags, grid=None, seed: int = 0) -> float:
        """Choose the fusion weight on validation data by maximising
        0.5 * clean macro-F1 + 0.5 * mean macro-F1 under 5/15/30 % noise, and store it."""
        grid = np.round(np.arange(0.0, 1.0001, 0.05), 2) if grid is None else grid
        sets = [val] + _noisy_copies(val, self.noise_refs, seed_offset=seed)
        p_net = [_net_proba(self.net, d, self.device) for d in sets]
        p_tree = [self.tree.predict_proba(tree_features(d.slots, d.mask, d.static, self.calibration))[:, 1] for d in sets]
        best_w, best_s = self.w_tree, -1.0
        for w in grid:
            f1 = [macro_f1(val.labels, ((w * pt + (1 - w) * pn) >= THRESHOLD).astype(int)) for pn, pt in zip(p_net, p_tree)]
            score = 0.5 * f1[0] + 0.5 * float(np.mean(f1[1:]))
            if score > best_s + 1e-12:
                best_w, best_s = float(w), score
        self.w_tree = best_w
        return best_w

    # -- scoring ----------------------------------------------------------
    def predict_proba_parts(self, d: Bags) -> tuple[np.ndarray, np.ndarray]:
        """(network probability, tree probability). Physics features are recomputed from the input."""
        p_net = _net_proba(self.net, d, self.device)
        p_tree = self.tree.predict_proba(tree_features(d.slots, d.mask, d.static, self.calibration))[:, 1]
        return p_net, p_tree

    def predict_proba(self, d: Bags) -> np.ndarray:
        """Fused probability ``w * p_tree + (1 - w) * p_net``."""
        p_net, p_tree = self.predict_proba_parts(d)
        return self.w_tree * p_tree + (1.0 - self.w_tree) * p_net

    def predict(self, d: Bags) -> np.ndarray:
        """Alert (1) when the fused probability is at least 0.5."""
        return (self.predict_proba(d) >= THRESHOLD).astype(int)

    def score(self, d: Bags) -> float:
        """Macro-F1 on labelled sessions."""
        return macro_f1(d.labels, self.predict(d))

    # -- persistence ------------------------------------------------------
    def save(self, path) -> None:
        torch.save(dict(net=self.net.state_dict(), tree=self.tree, calibration=self.calibration,
                        noise_refs=self.noise_refs, w_tree=self.w_tree, net_cfg=self.net_cfg, tree_params=self.tree_params), path)

    @classmethod
    def load(cls, path, device: str | None = None) -> "PIFusion":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(device=device, w_tree=ck["w_tree"], net_cfg=ck["net_cfg"], tree_params=ck["tree_params"])
        c = m.net_cfg
        m.net = MSCMIL(c["dropout"], c["encoder_hidden"], c["encoder_out"], c["lstm_hidden_per_dir"], 4, c["static_hidden"])
        m.net.load_state_dict(ck["net"])
        m.net.to(m.device).eval()
        m.tree, m.calibration, m.noise_refs = ck["tree"], ck["calibration"], ck["noise_refs"]
        return m
