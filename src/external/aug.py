"""E4-style augmentation recipe (frozen ACN recipe, transferred): per-bag Gaussian relative noise r~U(0,.3) on
continuous features + per-bag instance dropout q~U(0,.25), 60 epochs / patience 12 / cosine LR; val-score checkpointing.
Also noise-augmented tree training. Validation only unless called from run_final."""
from __future__ import annotations
import numpy as np, torch
from src.reproduction.mil_ids import bce_l2_loss
from src.candidates.common import supcon_loss
from src.external.data import summary_features, train_feature_std, perturb_noise
from src.external.models import build_deep, _predict, macro_f1, BATCH, FittedTree, make_tree

E_EPOCHS, E_PATIENCE, NOISE_MAX, DROP_MAX = 60, 12, 0.30, 0.25
VAL_NOISE = (0.05, 0.15, 0.30)


def val_sets(va, fstd, cont):
    """Fixed validation noise draws (rng offset 5000) shared by every model."""
    rng = np.random.default_rng(5000)
    return [perturb_noise(va["X"], s, fstd, cont, rng) for s in VAL_NOISE]


def val_score(pred_fn, va, noisy):
    """0.5*clean macro-F1 + 0.5*mean macro-F1 under val noise 5/15/30. pred_fn(X, mask)->proba."""
    c = macro_f1(va["y"], (pred_fn(va["X"], None) >= 0.5).astype(int))
    n = [macro_f1(va["y"], (pred_fn(X, None) >= 0.5).astype(int)) for X in noisy]
    return 0.5 * c + 0.5 * float(np.mean(n)), c, float(np.mean(n))


def augment_np(X, rng, fstd, cont):
    """numpy per-bag augmentation -> (X', mask)."""
    n, W, d = X.shape
    r = rng.uniform(0, NOISE_MAX, (n, 1, 1)).astype(np.float32)
    sd = r * (fstd * cont.astype(np.float32))[None, None, :]
    Xn = X + rng.standard_normal(X.shape).astype(np.float32) * sd
    q = rng.uniform(0, DROP_MAX, (n, 1))
    keep = rng.random((n, W)) >= q
    keep[~keep.any(1), 0] = True
    return np.where(keep[:, :, None], Xn, np.float32(0)).astype(np.float32), keep


def fit_deep_e4(name, params, tr, va, seed, device, fstd, cont, noisy_val, max_epochs=E_EPOCHS, patience=E_PATIENCE):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr = torch.as_tensor(tr["X"], device=device); ytr = torch.as_tensor(tr["y"], device=device).float()
    n, W, d = Xtr.shape
    sdv = torch.as_tensor(fstd * cont.astype(np.float32), device=device)[None, None, :]
    model = build_deep(name, d, W, params).to(device)
    lr = float(params.get("lr", 1e-3))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=0.01 * lr)
    l2 = float(params.get("l2", 1e-4)); beta = float(params.get("beta", 0.1)); temp = float(params.get("temperature", 0.1))
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, bad, hist = -1.0, None, 0, []
    pf = lambda X, m: _predict(model, X, m, device)
    for ep in range(max_epochs):
        model.train(); perm = torch.randperm(n, generator=g).to(device); tot = 0.0; nb = 0
        for a in range(0, n, BATCH):
            idx = perm[a:a + BATCH]; x, y = Xtr[idx], ytr[idx]; B = len(idx)
            r = torch.rand(B, 1, 1, device=device) * NOISE_MAX
            x = x + torch.randn_like(x) * r * sdv
            q = torch.rand(B, 1, device=device) * DROP_MAX
            keep = torch.rand(B, W, device=device) >= q
            keep[~keep.any(1), 0] = True
            x = torch.where(keep.unsqueeze(-1), x, torch.zeros((), device=device))
            logit, H = model(x, keep)
            loss = bce_l2_loss(logit, y, model, l2)
            if name == "msc_mil" and beta > 0:
                loss = loss + beta * supcon_loss(H, y.long(), temp)
            if not torch.isfinite(loss):
                return None, {"failed": "non-finite loss"}
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach()); nb += 1
        sched.step()
        s, c, nz = val_score(pf, va, noisy_val); hist.append((ep, tot / nb, s, c, nz))
        if s > best:
            best, bad = s, 0; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)
    from src.external.models import FittedDeep
    return FittedDeep(model, device), {"val_score": best, "epochs": len(hist), "history": hist}


N_COPIES = 3


def fit_tree_aug(name, params, tr, seed, fstd, cont, Saug=None):
    """train + N_COPIES augmented copies (summaries recomputed from perturbed, masked windows)."""
    if Saug is None:
        Saug = build_tree_aug_matrix(tr, fstd, cont)
    S, y = Saug
    est = make_tree(name, params, seed).fit(S, y)
    return FittedTree(est)


def build_tree_aug_matrix(tr, fstd, cont, seed=0):
    rng = np.random.default_rng(31337 + seed)
    Ss = [summary_features(tr["X"])]; ys = [tr["y"]]
    for _ in range(N_COPIES):
        X2, m2 = augment_np(tr["X"], rng, fstd, cont)
        Ss.append(summary_features(X2, m2)); ys.append(tr["y"])
    return np.concatenate(Ss), np.concatenate(ys)
