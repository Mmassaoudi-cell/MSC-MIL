"""Candidate 3 -- SSM-MIL (Selective State-Space MIL). MODEL_CANDIDATES.md
Candidate 3.

Same instance encoder and static-fusion/classifier as the reproduction; the
BiLSTM is replaced by a small stack of 2 uni-directional selective-SSM layers
(common.SelectiveSSMLayer -- a Mamba-lite block, see that module's docstring
for the "no mamba_ssm package installed, naive sequential scan" note),
followed by the same attention pooling.

Because the SSM is uni-directional (not bidirectional like the baseline's
BiLSTM), its output dim is `encoder_out` (no x2 doubling) -- so the pooled
H_seq / classifier input dims are correspondingly smaller than the baseline's
2*lstm_hidden_per_dir. This is the intended, honest efficiency trade-off the
candidate targets (MODEL_CANDIDATES.md: "Roughly halved recurrent compute vs.
BiLSTM"), not a bug to "fix" by artificially widening the SSM to match.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .common import AttentionPooling, InstanceEncoder, SelectiveSSMLayer


class SSMMIL(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_hidden: int,
        encoder_out: int,
        static_dim: int,
        static_hidden: int,
        d_state: int = 8,
        n_ssm_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = InstanceEncoder(input_dim, encoder_hidden, encoder_out)
        self.ssm_layers = nn.ModuleList(
            [SelectiveSSMLayer(encoder_out, d_state=d_state, dropout=dropout) for _ in range(n_ssm_layers)]
        )
        self.attn = AttentionPooling(encoder_out)
        self.static_proj = nn.Sequential(nn.Linear(static_dim, static_hidden), nn.ReLU())
        self.classifier = nn.Linear(encoder_out + static_hidden, 1)

    def _encode(self, slots: torch.Tensor, mask: torch.Tensor, static_feats: torch.Tensor):
        h = self.encoder(slots)
        for layer in self.ssm_layers:
            h = layer(h)
        H_seq, alpha = self.attn(h, mask)
        H_static = self.static_proj(static_feats)
        H = torch.cat([H_seq, H_static], dim=-1)
        return H, alpha

    def forward(self, slots, mask, static_feats):
        H, alpha = self._encode(slots, mask, static_feats)
        logit = self.classifier(H).squeeze(-1)
        return torch.sigmoid(logit), alpha

    def logits(self, slots, mask, static_feats):
        H, alpha = self._encode(slots, mask, static_feats)
        logit = self.classifier(H).squeeze(-1)
        return logit, alpha
