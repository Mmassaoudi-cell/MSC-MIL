"""Candidate 1 -- MGAT-MIL (Multi-Generator Adversarially-invariant MIL).
MODEL_CANDIDATES.md Candidate 1.

Same instance-encoder -> BiLSTM -> attention-pooling -> static-fusion ->
classifier backbone as the reproduction's MILBiLSTMIDS (mil_ids.py) -- so any
gain over the baseline is attributable to the TRAINING PROTOCOL (multi-
generator attack pool + GRL domain-invariance head), not a bigger network.
The multi-generator training-data construction lives in
src/candidates/multigen_dataset.py; this module only defines the model.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .common import AttentionPooling, DomainClassifierHead, InstanceEncoder, grad_reverse


class MGATMIL(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_hidden: int,
        encoder_out: int,
        lstm_hidden_per_dir: int,
        static_dim: int,
        static_hidden: int,
        n_generators: int,
        dropout: float = 0.2,
        domain_hidden: int = 32,
    ):
        super().__init__()
        self.encoder = InstanceEncoder(input_dim, encoder_hidden, encoder_out)
        self.bilstm = nn.LSTM(encoder_out, lstm_hidden_per_dir, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        dim_2p = 2 * lstm_hidden_per_dir
        self.attn = AttentionPooling(dim_2p)
        self.static_proj = nn.Sequential(nn.Linear(static_dim, static_hidden), nn.ReLU())
        self.classifier = nn.Linear(dim_2p + static_hidden, 1)
        # GRL domain-adversarial head sits on the pooled session embedding
        # H_seq (pre-static-fusion), per MODEL_CANDIDATES.md Candidate 1.
        self.domain_head = DomainClassifierHead(dim_2p, domain_hidden, n_generators)

    def _encode(self, slots: torch.Tensor, mask: torch.Tensor, static_feats: torch.Tensor):
        h = self.encoder(slots)
        z, _ = self.bilstm(h)
        z = self.dropout(z)
        H_seq, alpha = self.attn(z, mask)
        H_static = self.static_proj(static_feats)
        H = torch.cat([H_seq, H_static], dim=-1)
        return H, H_seq, alpha

    def forward(self, slots, mask, static_feats):
        H, _, alpha = self._encode(slots, mask, static_feats)
        logit = self.classifier(H).squeeze(-1)
        return torch.sigmoid(logit), alpha

    def logits(self, slots, mask, static_feats):
        H, _, alpha = self._encode(slots, mask, static_feats)
        logit = self.classifier(H).squeeze(-1)
        return logit, alpha

    def logits_with_domain(self, slots, mask, static_feats, grl_lambda: float):
        """Train-time only: also returns domain-classifier logits over
        GRL(H_seq) for L_dom = CE(g(GRL_lambda(H_seq)), generator_id)."""
        H, H_seq, alpha = self._encode(slots, mask, static_feats)
        logit = self.classifier(H).squeeze(-1)
        domain_logits = self.domain_head(grad_reverse(H_seq, grl_lambda))
        return logit, alpha, domain_logits
