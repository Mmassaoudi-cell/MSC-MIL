"""Benchmarks 8-11 (Part A): deep sequence-aware models trained on the SAME
288-slot input as MSC-MIL, at the SAME training budget as MSC-MIL for
fairness (max_epochs=30, patience=8 on val macro-F1@0.5, Adam) but using
reproduction/config.py's IDSConfig lr/l2_reg/batch_size (NOT MSC-MIL's tuned
hyperparameters -- these are independent benchmarks, not MSC-MIL variants,
per the task brief).

  8.  MLP        -- flatten(288*3) + static -> feedforward. "Does sequence
                     order even matter" control: no recurrence/attention/conv.
  9.  CNN1D       -- 1D conv+pool stack over the 288-slot sequence (mask given
                     as an extra input channel; convolution has no native
                     masking primitive, documented simplification).
  10. PlainLSTMMIL -- src.reproduction.mil_ids.MILBiLSTMIDS verbatim, with
                     AttentionPooling replaced by masked MEAN-pooling. Doubles
                     as an attention ablation.
  11. TransformerMIL -- InstanceEncoder + learnable positional embedding +
                     a 2-layer/4-head TransformerEncoder (d_model=64, matching
                     MSC-MIL/MILBiLSTMIDS's encoder_out=64 for parameter-
                     comparable backbone capacity) + AttentionPooling + the
                     same static-fusion classifier.

All four models share the (slots, mask, static) -> (y_hat, aux) / logits(...)
-> (logit, aux) interface used throughout src/candidates so the same generic
training loop, bce_l2_loss, and infer_forward pattern apply uniformly.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.benchmarks.data_utils import PROJECT_ROOT, get_device, load_scma_data
from src.reproduction.config import IDSConfig
from src.reproduction.metrics import macro_f1_score, binary_metrics
from src.reproduction.mil_ids import AttentionPooling, InstanceEncoder, bce_l2_loss

RAW_DIR = PROJECT_ROOT / "results" / "raw" / "benchmarks"
CKPT_DIR = PROJECT_ROOT / "results" / "checkpoints" / "benchmarks"
MAX_EPOCHS = 30
PATIENCE = 8
SEEDS = (0, 1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Model 8: MLP (flattened, non-sequence control)
# ---------------------------------------------------------------------------
class MLPClassifier(nn.Module):
    def __init__(self, n_slots: int, input_dim: int, static_dim: int, hidden=(256, 64), dropout: float = 0.2):
        super().__init__()
        in_dim = n_slots * input_dim + static_dim
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)

    def _forward_logit(self, slots, mask, static_feats):
        B = slots.shape[0]
        x = torch.cat([slots.reshape(B, -1), static_feats], dim=-1)
        h = self.body(x)
        return self.head(h).squeeze(-1), None

    def forward(self, slots, mask, static_feats):
        logit, aux = self._forward_logit(slots, mask, static_feats)
        return torch.sigmoid(logit), aux

    def logits(self, slots, mask, static_feats):
        return self._forward_logit(slots, mask, static_feats)


# ---------------------------------------------------------------------------
# Model 9: 1D-CNN over the 288-slot sequence
# ---------------------------------------------------------------------------
class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim: int, static_dim: int, static_hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        in_ch = input_dim + 1  # + mask channel
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.static_proj = nn.Sequential(nn.Linear(static_dim, static_hidden), nn.ReLU())
        self.head = nn.Linear(64 + static_hidden, 1)

    def _forward_logit(self, slots, mask, static_feats):
        x = torch.cat([slots, mask.unsqueeze(-1).float()], dim=-1)  # (B,N,C)
        x = x.transpose(1, 2)  # (B,C,N)
        feat = self.conv(x).squeeze(-1)  # (B,64)
        feat = self.dropout(feat)
        H_static = self.static_proj(static_feats)
        H = torch.cat([feat, H_static], dim=-1)
        return self.head(H).squeeze(-1), None

    def forward(self, slots, mask, static_feats):
        logit, aux = self._forward_logit(slots, mask, static_feats)
        return torch.sigmoid(logit), aux

    def logits(self, slots, mask, static_feats):
        return self._forward_logit(slots, mask, static_feats)


# ---------------------------------------------------------------------------
# Model 10: Plain-LSTM-MIL (MILBiLSTMIDS with mean-pooling instead of attn)
# ---------------------------------------------------------------------------
class MeanPooling(nn.Module):
    def forward(self, z: torch.Tensor, mask: torch.Tensor):
        m = mask.unsqueeze(-1).float()
        denom = m.sum(dim=1).clamp(min=1.0)
        H_seq = (z * m).sum(dim=1) / denom
        return H_seq, None


class PlainLSTMMIL(nn.Module):
    def __init__(self, input_dim, encoder_hidden, encoder_out, lstm_hidden_per_dir, static_dim, static_hidden, dropout=0.2):
        super().__init__()
        self.encoder = InstanceEncoder(input_dim, encoder_hidden, encoder_out)
        self.bilstm = nn.LSTM(encoder_out, lstm_hidden_per_dir, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.pool = MeanPooling()
        dim_2p = 2 * lstm_hidden_per_dir
        self.static_proj = nn.Sequential(nn.Linear(static_dim, static_hidden), nn.ReLU())
        self.classifier = nn.Linear(dim_2p + static_hidden, 1)

    def _forward_logit(self, slots, mask, static_feats):
        h = self.encoder(slots)
        z, _ = self.bilstm(h)
        z = self.dropout(z)
        H_seq, _ = self.pool(z, mask)
        H_static = self.static_proj(static_feats)
        H = torch.cat([H_seq, H_static], dim=-1)
        return self.classifier(H).squeeze(-1), None

    def forward(self, slots, mask, static_feats):
        logit, aux = self._forward_logit(slots, mask, static_feats)
        return torch.sigmoid(logit), aux

    def logits(self, slots, mask, static_feats):
        return self._forward_logit(slots, mask, static_feats)


# ---------------------------------------------------------------------------
# Model 11: Transformer-MIL
# ---------------------------------------------------------------------------
class TransformerMIL(nn.Module):
    def __init__(self, input_dim, encoder_hidden, encoder_out, static_dim, static_hidden,
                 n_slots=288, n_layers=2, n_heads=4, dim_feedforward=128, dropout=0.2):
        super().__init__()
        self.encoder = InstanceEncoder(input_dim, encoder_hidden, encoder_out)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_slots, encoder_out))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=encoder_out, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="relu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.attn = AttentionPooling(encoder_out)
        self.static_proj = nn.Sequential(nn.Linear(static_dim, static_hidden), nn.ReLU())
        self.classifier = nn.Linear(encoder_out + static_hidden, 1)

    def _forward_logit(self, slots, mask, static_feats):
        h = self.encoder(slots) + self.pos_emb
        z = self.transformer(h, src_key_padding_mask=~mask)
        H_seq, alpha = self.attn(z, mask)
        H_static = self.static_proj(static_feats)
        H = torch.cat([H_seq, H_static], dim=-1)
        return self.classifier(H).squeeze(-1), alpha

    def forward(self, slots, mask, static_feats):
        logit, aux = self._forward_logit(slots, mask, static_feats)
        return torch.sigmoid(logit), aux

    def logits(self, slots, mask, static_feats):
        return self._forward_logit(slots, mask, static_feats)


def build_model(name: str, ids_cfg: IDSConfig):
    if name == "mlp":
        return MLPClassifier(ids_cfg.n_slots, ids_cfg.input_dim, ids_cfg.static_dim, dropout=ids_cfg.dropout)
    if name == "cnn1d":
        return CNN1DClassifier(ids_cfg.input_dim, ids_cfg.static_dim, ids_cfg.static_hidden, dropout=ids_cfg.dropout)
    if name == "plain_lstm_mil":
        return PlainLSTMMIL(
            ids_cfg.input_dim, ids_cfg.encoder_hidden, ids_cfg.encoder_out, ids_cfg.lstm_hidden_per_dir,
            ids_cfg.static_dim, ids_cfg.static_hidden, dropout=ids_cfg.dropout,
        )
    if name == "transformer_mil":
        return TransformerMIL(
            ids_cfg.input_dim, ids_cfg.encoder_hidden, ids_cfg.encoder_out, ids_cfg.static_dim,
            ids_cfg.static_hidden, n_slots=ids_cfg.n_slots, dropout=ids_cfg.dropout,
        )
    raise ValueError(name)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def infer_forward(model, slots, mask, static, device, batch_size=256):
    model.eval()
    scores = []
    for start in range(0, len(slots), batch_size):
        sl = slice(start, start + batch_size)
        s = torch.from_numpy(slots[sl]).to(device)
        m = torch.from_numpy(mask[sl]).to(device)
        st = torch.from_numpy(static[sl]).to(device)
        y_hat, _ = model(s, m, st)
        scores.append(y_hat.cpu().numpy())
    return np.concatenate(scores)


def generic_train_loop(model, ids_cfg: IDSConfig, data, device, seed, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = TensorDataset(
        torch.from_numpy(data["train"].slots), torch.from_numpy(data["train"].mask),
        torch.from_numpy(data["train"].static), torch.from_numpy(data["train"].labels).float(),
    )
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=ids_cfg.batch_size, shuffle=True, generator=gen)
    optimizer = torch.optim.Adam(model.parameters(), lr=ids_cfg.lr)

    val = data["val"]
    best_val_f1, best_state, best_epoch, no_improve = -1.0, None, -1, 0
    history = []
    converged, fail_reason = True, None

    for epoch in range(max_epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for slots, mask, static, labels in loader:
            slots, mask, static, labels = slots.to(device), mask.to(device), static.to(device), labels.to(device)
            logit, _ = model.logits(slots, mask, static)
            loss = bce_l2_loss(logit, labels, model, ids_cfg.l2_reg)
            if not torch.isfinite(loss):
                converged = False
                fail_reason = f"non-finite loss at epoch {epoch}"
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        if not converged:
            break
        val_score = infer_forward(model, val.slots, val.mask, val.static, device)
        if not np.isfinite(val_score).all():
            converged = False
            fail_reason = f"non-finite val score at epoch {epoch}"
            break
        val_pred = (val_score >= 0.5).astype(int)
        val_f1 = macro_f1_score(val.labels, val_pred)
        history.append({"epoch": epoch, "train_loss": epoch_loss / max(n_batches, 1), "val_macro_f1@0.5": val_f1})
        if val_f1 > best_val_f1:
            best_val_f1, best_epoch, no_improve = val_f1, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    return history, best_state, best_epoch, best_val_f1, converged, fail_reason


def main():
    scma_type = 1
    device = get_device()
    ids_cfg = IDSConfig()  # unmodified reproduction defaults: lr=1e-3, l2_reg=1e-4, batch_size=32
    data = load_scma_data(scma_type, device)
    print(f"[deep_models] SCMA-{scma_type} device={device} train={len(data['train'].labels)} val={len(data['val'].labels)}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    model_names = ["mlp", "cnn1d", "plain_lstm_mil", "transformer_mil"]
    for name in model_names:
        for seed in SEEDS:
            out_path = RAW_DIR / f"{name}_scma{scma_type}_seed{seed}.json"
            ckpt_path = CKPT_DIR / f"{name}_scma{scma_type}_seed{seed}.pt"
            if out_path.exists():
                print(f"[deep_models] SKIP (exists) {out_path}")
                continue
            print(f"[deep_models] === {name} / SCMA-{scma_type} / seed={seed} ===")
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = build_model(name, ids_cfg).to(device)
            t0 = time.time()
            history, best_state, best_epoch, best_val_f1, converged, fail_reason = generic_train_loop(
                model, ids_cfg, data, device, seed
            )
            train_time_s = time.time() - t0
            result = {
                "model": name, "scma_type": scma_type, "seed": seed, "converged": converged,
                "fail_reason": fail_reason, "n_params": n_params(model), "train_time_s": train_time_s,
                "epochs_trained": len(history), "best_epoch": best_epoch, "best_val_macro_f1@0.5": best_val_f1,
            }
            if converged and best_state is not None:
                model.load_state_dict(best_state)
                val = data["val"]
                val_score = infer_forward(model, val.slots, val.mask, val.static, device)
                val_pred = (val_score >= 0.5).astype(int)
                result["val_metrics"] = binary_metrics(val.labels, val_pred, val_score)
                result["val_metrics"]["macro_f1"] = macro_f1_score(val.labels, val_pred)
                torch.save({"model_state": model.state_dict(), "model": name, "scma_type": scma_type, "seed": seed}, ckpt_path)
            else:
                result["val_metrics"] = None
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            vm = result.get("val_metrics")
            if vm:
                print(f"[deep_models] {name} seed={seed}: OK val_macro_f1={vm['macro_f1']:.4f} "
                      f"recall={vm['recall']:.4f} far={vm['false_alarm_rate']:.4f} "
                      f"epochs={result['epochs_trained']} time={train_time_s:.1f}s n_params={result['n_params']}")
            else:
                print(f"[deep_models] {name} seed={seed}: FAIL ({fail_reason})")
    print("[deep_models] done.")


if __name__ == "__main__":
    main()
