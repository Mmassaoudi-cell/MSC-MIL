"""MIL-BiLSTM IDS, reproducing SOURCE_PAPER_AUDIT.md Sec. 6 (Eq. 3-9).

Session = bag of up to 288 five-minute slots. Session-level label only
(weakly supervised MIL). Architecture:
  instance encoder (Eq.4, per-slot MLP)
  -> BiLSTM (Eq.3)
  -> attention pooling (Eq.5-6)
  -> static-feature fusion (Eq.7-8)
  -> sigmoid classifier (Eq.9)
"""
import torch
import torch.nn as nn


class InstanceEncoder(nn.Module):
    """Eq. 4: h_ji = ReLU(W2 . ReLU(W1 . r_ji + b1) + b2), shared across slots."""

    def __init__(self, input_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        # r: (B, N, input_dim) -> (B, N, out_dim); Linear broadcasts over N.
        return self.net(r)


class AttentionPooling(nn.Module):
    """Eq. 5-6: soft-attention weighted sum over BiLSTM outputs z_i in R^{2p}."""

    def __init__(self, dim_2p: int):
        super().__init__()
        self.w = nn.Linear(dim_2p, 1, bias=False)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # z: (B, N, 2p); mask: (B, N) bool, True = active (real) slot.
        scores = self.w(z).squeeze(-1)  # (B, N)
        scores = scores.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(scores, dim=1)  # (B, N), Eq. 5
        alpha = torch.nan_to_num(alpha, nan=0.0)  # bags with zero active slots -> zero vector, guarded downstream
        H_seq = torch.einsum("bn,bnd->bd", alpha, z)  # Eq. 6
        return H_seq, alpha


class MILBiLSTMIDS(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_hidden: int,
        encoder_out: int,
        lstm_hidden_per_dir: int,
        static_dim: int,
        static_hidden: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = InstanceEncoder(input_dim, encoder_hidden, encoder_out)
        self.bilstm = nn.LSTM(
            input_size=encoder_out,
            hidden_size=lstm_hidden_per_dir,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        dim_2p = 2 * lstm_hidden_per_dir
        self.attn = AttentionPooling(dim_2p)
        self.static_proj = nn.Sequential(nn.Linear(static_dim, static_hidden), nn.ReLU())  # Eq. 7
        self.classifier = nn.Linear(dim_2p + static_hidden, 1)  # Eq. 9

    def forward(
        self, slots: torch.Tensor, mask: torch.Tensor, static_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        slots: (B, N, input_dim) per-slot raw features
        mask: (B, N) bool, True = active/real slot (per paper's masking rule for
              sessions shorter than 24h -- inactive slots excluded from attention)
        static_feats: (B, static_dim)
        returns: y_hat (B,) in [0,1], attention weights alpha (B, N) for slot attribution
        """
        h = self.encoder(slots)  # Eq. 4
        z, _ = self.bilstm(h)  # Eq. 3
        z = self.dropout(z)
        H_seq, alpha = self.attn(z, mask)  # Eq. 5-6
        H_static = self.static_proj(static_feats)  # Eq. 7
        H = torch.cat([H_seq, H_static], dim=-1)  # Eq. 8
        logit = self.classifier(H).squeeze(-1)  # Eq. 9 (pre-sigmoid)
        y_hat = torch.sigmoid(logit)
        return y_hat, alpha

    def logits(self, slots, mask, static_feats):
        """Return pre-sigmoid logits (for use with BCEWithLogitsLoss)."""
        h = self.encoder(slots)
        z, _ = self.bilstm(h)
        z = self.dropout(z)
        H_seq, alpha = self.attn(z, mask)
        H_static = self.static_proj(static_feats)
        H = torch.cat([H_seq, H_static], dim=-1)
        logit = self.classifier(H).squeeze(-1)
        return logit, alpha


def bce_l2_loss(logits: torch.Tensor, labels: torch.Tensor, model: nn.Module, l2_reg: float) -> torch.Tensor:
    """Eq. 10/11: regularized BCE. logits are pre-sigmoid (numerically stable)."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
    l2 = sum((p**2).sum() for p in model.parameters())
    return bce + l2_reg * l2
