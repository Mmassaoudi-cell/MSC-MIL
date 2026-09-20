"""Candidate 2 -- MSC-MIL (Multi-Scale Contrastive MIL). MODEL_CANDIDATES.md
Candidate 2.

Three parallel BiLSTM+attention branches over the instance-encoder output,
pooled at {1,3,12} slots/token (native 5-min / 15-min / 60-min), concatenated
with the static branch. A supervised-contrastive loss (benign-vs-attack, see
common.supcon_loss docstring for the documented subtype-separation skip) is
computed on the fused embedding during training (see smoke_test.py's training
loop -- `logits_with_embedding` exposes the fused H for that purpose).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .common import InstanceEncoder, ScaleBranch

POOL_SIZES = (1, 3, 12)


class MSCMIL(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_hidden: int,
        encoder_out: int,
        lstm_hidden_per_dir: int,
        static_dim: int,
        static_hidden: int,
        dropout: float = 0.2,
        pool_sizes: tuple = POOL_SIZES,
    ):
        """pool_sizes defaults to the frozen MSC-MIL (1,3,12) config. Exposed
        as a constructor arg (ablation-only use, post-freeze -- the frozen
        FINAL_MODEL_CONFIG.yaml model never passes anything but the default)
        so ABLATION_RESULTS.md's 'MSC-MIL minus multi-scale' variant
        (pool_sizes=(1,)) and the parameter-matched single-branch-wider
        variant can reuse this exact class/forward/loss wiring instead of a
        parallel reimplementation."""
        super().__init__()
        self.pool_sizes = tuple(pool_sizes)
        self.encoder = InstanceEncoder(input_dim, encoder_hidden, encoder_out)
        self.branches = nn.ModuleList(
            [ScaleBranch(encoder_out, lstm_hidden_per_dir, s, dropout) for s in self.pool_sizes]
        )
        dim_2p = 2 * lstm_hidden_per_dir
        fused_dim = dim_2p * len(self.pool_sizes) + static_hidden
        self.static_proj = nn.Sequential(nn.Linear(static_dim, static_hidden), nn.ReLU())
        self.classifier = nn.Linear(fused_dim, 1)

    def _encode(self, slots: torch.Tensor, mask: torch.Tensor, static_feats: torch.Tensor):
        h = self.encoder(slots)
        H_seqs = []
        native_alpha = None
        for pool_size, branch in zip(self.pool_sizes, self.branches):
            H_seq, alpha = branch(h, mask)
            H_seqs.append(H_seq)
            if pool_size == 1:
                native_alpha = alpha  # (B, 288) -- matches MILBiLSTMIDS's alpha interface exactly
        H_static = self.static_proj(static_feats)
        H = torch.cat(H_seqs + [H_static], dim=-1)  # H_j, MODEL_CANDIDATES.md Candidate 2
        return H, native_alpha

    def forward(self, slots, mask, static_feats):
        H, alpha = self._encode(slots, mask, static_feats)
        logit = self.classifier(H).squeeze(-1)
        return torch.sigmoid(logit), alpha

    def logits(self, slots, mask, static_feats):
        H, alpha = self._encode(slots, mask, static_feats)
        logit = self.classifier(H).squeeze(-1)
        return logit, alpha

    def logits_with_embedding(self, slots, mask, static_feats):
        """Train-time only: also returns the fused embedding H_j for the
        SupCon contrastive loss term."""
        H, alpha = self._encode(slots, mask, static_feats)
        logit = self.classifier(H).squeeze(-1)
        return logit, alpha, H


class MSCMILMulticlass(nn.Module):
    """Multiclass variant of MSC-MIL (Master Prompt Sec.17-18 -- attack-TYPE
    attribution, not just binary detection). Reuses the EXACT SAME backbone
    (InstanceEncoder + 3 ScaleBranch instances at pool_sizes={1,3,12} +
    static projection) as the frozen binary MSC-MIL, constructed with the
    identical FINAL_MODEL_CONFIG.yaml hyperparameters -- the ONLY difference
    from the frozen model is the final classifier layer, which here maps the
    fused embedding to `n_classes` raw logits (softmax/cross-entropy) instead
    of a single logit (sigmoid/BCE). `supcon_loss` (common.py) is label-
    equality-based and therefore already works unchanged for >2 classes (same
    positive-pair definition, just now grouping by attack TYPE instead of
    binary benign-vs-attack), so it is reused here with no modification --
    consistent with the task's 'same hyperparameters, just a 7-way softmax
    output' framing."""

    def __init__(
        self,
        input_dim: int,
        encoder_hidden: int,
        encoder_out: int,
        lstm_hidden_per_dir: int,
        static_dim: int,
        static_hidden: int,
        n_classes: int,
        dropout: float = 0.2,
        pool_sizes: tuple = POOL_SIZES,
    ):
        super().__init__()
        self.pool_sizes = tuple(pool_sizes)
        self.n_classes = n_classes
        self.encoder = InstanceEncoder(input_dim, encoder_hidden, encoder_out)
        self.branches = nn.ModuleList(
            [ScaleBranch(encoder_out, lstm_hidden_per_dir, s, dropout) for s in self.pool_sizes]
        )
        dim_2p = 2 * lstm_hidden_per_dir
        fused_dim = dim_2p * len(self.pool_sizes) + static_hidden
        self.static_proj = nn.Sequential(nn.Linear(static_dim, static_hidden), nn.ReLU())
        self.classifier = nn.Linear(fused_dim, n_classes)

    def _encode(self, slots: torch.Tensor, mask: torch.Tensor, static_feats: torch.Tensor):
        h = self.encoder(slots)
        H_seqs = []
        native_alpha = None
        for pool_size, branch in zip(self.pool_sizes, self.branches):
            H_seq, alpha = branch(h, mask)
            H_seqs.append(H_seq)
            if pool_size == 1:
                native_alpha = alpha
        H_static = self.static_proj(static_feats)
        H = torch.cat(H_seqs + [H_static], dim=-1)
        return H, native_alpha

    def forward(self, slots, mask, static_feats):
        H, alpha = self._encode(slots, mask, static_feats)
        logits = self.classifier(H)  # (B, n_classes)
        return torch.softmax(logits, dim=-1), alpha

    def logits(self, slots, mask, static_feats):
        H, alpha = self._encode(slots, mask, static_feats)
        logits = self.classifier(H)
        return logits, alpha

    def logits_with_embedding(self, slots, mask, static_feats):
        H, alpha = self._encode(slots, mask, static_feats)
        logits = self.classifier(H)
        return logits, alpha, H
