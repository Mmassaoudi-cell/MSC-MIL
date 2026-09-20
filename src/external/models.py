"""Model zoo for the external validation. Mirrors the EV-charging study:
  * deep MIL nets reuse InstanceEncoder / AttentionPooling / ScaleBranch / supcon_loss / bce_l2_loss
    from src.reproduction.mil_ids and src.candidates.common (imported, not re-implemented);
  * tree/classical models are fit on window-summary features (mean,std,min,max per feature).
Deep nets see ONLY the raw window (B,W,d) + bool mask (no static branch; the EV static features have no
analogue here). All deep nets: forward(x, mask) -> (logit, embedding|None).
Common interface for evaluation: Fitted.predict_proba(X, mask=None) -> P(attack) per bag.
"""
from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.reproduction.mil_ids import AttentionPooling, InstanceEncoder, bce_l2_loss
from src.candidates.common import ScaleBranch, supcon_loss
from src.external.data import summary_features

DEEP = ["plain_lstm_mil", "bilstm_attn_mil", "transformer_mil", "msc_mil"]
TREE = ["logreg", "random_forest", "xgboost", "lightgbm", "catboost"]
ALL_MODELS = TREE + DEEP
MAX_EPOCHS, PATIENCE, BATCH = 30, 8, 128
POOL_SIZES = (1, 4, 16)  # W=64 -> 64/16/4 tokens (EV analogue: 288 -> 1/3/12 slots per token)


def macro_f1(y, p):
    return float(f1_score(y, p, average="macro"))


# ----------------------------------------------------------------- deep models
class MeanPool(nn.Module):
    def forward(self, z, mask):
        m = mask.unsqueeze(-1).float()
        return (z * m).sum(1) / m.sum(1).clamp(min=1.0)


class PlainLSTMMIL(nn.Module):
    """Instance encoder -> BiLSTM -> masked MEAN pooling -> linear (attention ablation)."""
    def __init__(self, d, h=64, dropout=0.2, **_):
        super().__init__()
        self.enc = InstanceEncoder(d, h, h)
        self.lstm = nn.LSTM(h, h, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout); self.pool = MeanPool(); self.cls = nn.Linear(2 * h, 1)

    def forward(self, x, mask):
        z, _ = self.lstm(self.enc(x)); H = self.pool(self.drop(z), mask)
        return self.cls(H).squeeze(-1), H


class BiLSTMAttnMIL(nn.Module):
    """Source-paper style (MILBiLSTMIDS minus static branch): encoder -> BiLSTM -> attention pooling."""
    def __init__(self, d, h=64, dropout=0.2, **_):
        super().__init__()
        self.enc = InstanceEncoder(d, h, h)
        self.lstm = nn.LSTM(h, h, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout); self.attn = AttentionPooling(2 * h); self.cls = nn.Linear(2 * h, 1)

    def forward(self, x, mask):
        z, _ = self.lstm(self.enc(x)); H, _ = self.attn(self.drop(z), mask)
        return self.cls(H).squeeze(-1), H


class TransformerMIL(nn.Module):
    """Encoder + learnable pos-emb + TransformerEncoder (d_model=h, 4 heads) + attention pooling."""
    def __init__(self, d, h=64, dropout=0.2, W=64, n_layers=2, **_):
        super().__init__()
        self.enc = InstanceEncoder(d, h, h)
        self.pos = nn.Parameter(torch.zeros(1, W, h)); nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=h, nhead=4, dim_feedforward=2 * h, dropout=dropout,
                                           batch_first=True, activation="relu")
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.attn = AttentionPooling(h); self.cls = nn.Linear(h, 1)

    def forward(self, x, mask):
        z = self.tr(self.enc(x) + self.pos, src_key_padding_mask=~mask)
        z = torch.nan_to_num(z)
        H, _ = self.attn(z, mask)
        return self.cls(H).squeeze(-1), H


class MSCMILExt(nn.Module):
    """MSC-MIL analogue (src/candidates/msc_mil.py): three BiLSTM+attention ScaleBranches over the
    mean-pooled instance-encoder sequence at scales {1,4,16}, concat -> linear; SupCon on the fused
    embedding at train time. No static branch."""
    def __init__(self, d, h=64, dropout=0.2, pool_sizes=POOL_SIZES, **_):
        super().__init__()
        self.enc = InstanceEncoder(d, h, h)
        self.branches = nn.ModuleList([ScaleBranch(h, h, s, dropout) for s in pool_sizes])
        self.cls = nn.Linear(2 * h * len(pool_sizes), 1)

    def forward(self, x, mask):
        e = self.enc(x)
        H = torch.cat([b(e, mask)[0] for b in self.branches], -1)
        return self.cls(H).squeeze(-1), H


def build_deep(name, d, W, params):
    kw = dict(h=int(params.get("hidden", 64)), dropout=float(params.get("dropout", 0.2)))
    if name == "plain_lstm_mil": return PlainLSTMMIL(d, **kw)
    if name == "bilstm_attn_mil": return BiLSTMAttnMIL(d, **kw)
    if name == "transformer_mil": return TransformerMIL(d, W=W, n_layers=int(params.get("n_layers", 2)), **kw)
    if name == "msc_mil": return MSCMILExt(d, **kw)
    raise ValueError(name)


@torch.no_grad()
def _predict(model, X, mask, device, bs=2048):
    model.eval(); out = []
    for a in range(0, len(X), bs):
        x = torch.as_tensor(X[a:a + bs], device=device)
        m = torch.as_tensor(mask[a:a + bs], device=device) if mask is not None else torch.ones(x.shape[:2], dtype=torch.bool, device=device)
        out.append(torch.sigmoid(model(x, m)[0]).float().cpu().numpy())
    return np.concatenate(out)


class FittedDeep:
    def __init__(self, model, device): self.model, self.device = model, device
    def predict_proba(self, X, mask=None): return _predict(self.model, X, mask, self.device)


def fit_deep(name, params, tr, va, seed, device, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    """Adam, BCE+L2 (+ beta*SupCon for msc_mil), batch 128, early stop on val macro-F1@0.5, best-state restore."""
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr = torch.as_tensor(tr["X"], device=device); ytr = torch.as_tensor(tr["y"], device=device).float()
    n, W, d = Xtr.shape
    model = build_deep(name, d, W, params).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(params.get("lr", 1e-3)))
    l2 = float(params.get("l2", 1e-4)); beta = float(params.get("beta", 0.1)); temp = float(params.get("temperature", 0.1))
    g = torch.Generator(device="cpu").manual_seed(seed)
    ones = torch.ones((BATCH, W), dtype=torch.bool, device=device)
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(max_epochs):
        model.train(); perm = torch.randperm(n, generator=g).to(device); tot = 0.0
        for a in range(0, n, BATCH):
            idx = perm[a:a + BATCH]
            x, y = Xtr[idx], ytr[idx]; m = ones[: len(idx)]
            logit, H = model(x, m)
            loss = bce_l2_loss(logit, y, model, l2)
            if name == "msc_mil" and beta > 0:
                loss = loss + beta * supcon_loss(H, y.long(), temp)
            if not torch.isfinite(loss):
                return None, {"failed": "non-finite loss", "history": hist}
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
        p = _predict(model, va["X"], None, device)
        f1 = macro_f1(va["y"], (p >= 0.5).astype(int)); hist.append((ep, tot / max(1, n // BATCH), f1))
        if f1 > best:
            best, bad = f1, 0; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)
    return FittedDeep(model, device), {"val_macro_f1": best, "epochs": len(hist), "history": hist}


# ----------------------------------------------------------------- tree / classical
class FittedTree:
    def __init__(self, est): self.est = est
    def predict_proba(self, X, mask=None):
        return self.est.predict_proba(summary_features(X, mask))[:, 1]


def make_tree(name, p, seed):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=p["C"], max_iter=2000, random_state=seed))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                      min_samples_leaf=p["min_samples_leaf"], n_jobs=-1, random_state=seed)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"], learning_rate=p["learning_rate"],
                             subsample=p["subsample"], colsample_bytree=p["colsample_bytree"], tree_method="hist",
                             n_jobs=-1, random_state=seed, verbosity=0)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=p["n_estimators"], num_leaves=p["num_leaves"], learning_rate=p["learning_rate"],
                              subsample=p["subsample"], subsample_freq=1, colsample_bytree=p["colsample_bytree"],
                              n_jobs=-1, random_state=seed, verbose=-1)
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=p["iterations"], depth=p["depth"], learning_rate=p["learning_rate"],
                                  random_seed=seed, verbose=0, thread_count=-1, allow_writing_files=False)
    raise ValueError(name)


def fit_tree(name, params, tr, seed, Str=None):
    """Str: optional precomputed summary features of tr (same for all trials/seeds)."""
    S = Str if Str is not None else summary_features(tr["X"])
    est = make_tree(name, params, seed).fit(S, tr["y"])
    return FittedTree(est)


def fit_model(name, params, tr, va, seed, device, Str=None):
    if name in TREE:
        f = fit_tree(name, params, tr, seed, Str)
        return f, {"val_macro_f1": macro_f1(va["y"], (f.predict_proba(va["X"]) >= 0.5).astype(int))}
    return fit_deep(name, params, tr, va, seed, device)
