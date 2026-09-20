"""Candidate 4 -- GRMA-MIL (Generator-Robust Multi-scale Attention MIL).
MODEL_CANDIDATES.md Candidate 4 -- combined hybrid.

Architecture = Candidate 2's multi-scale BiLSTM+attention backbone (3
resolution branches + static fusion), with Candidate 1's GRL domain-
invariance head substituted onto the FUSED embedding H_j (per
MODEL_CANDIDATES.md: "the multi-scale fusion embedding H_j from Candidate 2
substituted wherever Candidate 1 used the single-scale H_seq"), plus
Candidate 2's supervised-contrastive term on that same H_j. Training-loop
loss: L = L_det + lambda(t)*L_dom(GRL) + beta*L_con(SupCon).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .common import DomainClassifierHead, InstanceEncoder, ScaleBranch, grad_reverse

POOL_SIZES = (1, 3, 12)


class GRMAMIL(nn.Module):
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
        self.branches = nn.ModuleList(
            [ScaleBranch(encoder_out, lstm_hidden_per_dir, s, dropout) for s in POOL_SIZES]
        )
        dim_2p = 2 * lstm_hidden_per_dir
        fused_dim = dim_2p * len(POOL_SIZES) + static_hidden
        self.static_proj = nn.Sequential(nn.Linear(static_dim, static_hidden), nn.ReLU())
        self.classifier = nn.Linear(fused_dim, 1)
        self.domain_head = DomainClassifierHead(fused_dim, domain_hidden, n_generators)

    def _encode(self, slots: torch.Tensor, mask: torch.Tensor, static_feats: torch.Tensor):
        h = self.encoder(slots)
        H_seqs = []
        native_alpha = None
        for pool_size, branch in zip(POOL_SIZES, self.branches):
            H_seq, alpha = branch(h, mask)
            H_seqs.append(H_seq)
            if pool_size == 1:
                native_alpha = alpha
        H_static = self.static_proj(static_feats)
        H = torch.cat(H_seqs + [H_static], dim=-1)  # H_j
        return H, native_alpha

    def forward(self, slots, mask, static_feats):
        H, alpha = self._encode(slots, mask, static_feats)
        logit = self.classifier(H).squeeze(-1)
        return torch.sigmoid(logit), alpha

    def logits(self, slots, mask, static_feats):
        H, alpha = self._encode(slots, mask, static_feats)
        logit = self.classifier(H).squeeze(-1)
        return logit, alpha

    def logits_with_domain(self, slots, mask, static_feats, grl_lambda: float):
        """Train-time only: returns (logit, alpha, domain_logits, H_j) --
        domain_logits = g(GRL_lambda(H_j)) and H_j is exposed for the SupCon
        term too (both auxiliary losses sit on the same fused embedding)."""
        H, alpha = self._encode(slots, mask, static_feats)
        logit = self.classifier(H).squeeze(-1)
        domain_logits = self.domain_head(grad_reverse(H, grl_lambda))
        return logit, alpha, domain_logits, H
