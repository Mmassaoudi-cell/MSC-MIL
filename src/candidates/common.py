"""Shared building blocks for the 4 hybrid candidate models (MODEL_CANDIDATES.md).

Reuses src/reproduction/mil_ids.py's InstanceEncoder and AttentionPooling
directly (imported, not reimplemented) so the instance-encoder + attention-
pooling mechanism is bit-identical to the baseline reproduction wherever a
candidate keeps it -- consistent with the task brief's "apples-to-apples on
backbone capacity" instruction.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.reproduction.mil_ids import AttentionPooling, InstanceEncoder

__all__ = [
    "InstanceEncoder",
    "AttentionPooling",
    "grad_reverse",
    "dann_lambda",
    "DomainClassifierHead",
    "avg_pool_seq",
    "ScaleBranch",
    "supcon_loss",
    "SelectiveSSMLayer",
]


# ---------------------------------------------------------------------------
# Candidate 1 / 4: Gradient Reversal Layer (GRL) for domain-adversarial
# (multi-generator invariance) training. MODEL_CANDIDATES.md Candidate 1
# "Core math": GRL_lambda(x) = x forward, -lambda*grad_x backward.
# ---------------------------------------------------------------------------
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


def dann_lambda(progress: float, lambda_max: float = 1.0, gamma: float = 10.0) -> float:
    """Standard DANN ramp schedule (Ganin & Lempitsky 2015): lambda(t) ramps
    0 -> lambda_max as training progress p in [0,1] increases, avoiding early
    training instability from a strong reversed gradient (MODEL_CANDIDATES.md
    Candidate 1 core math: "lambda(t) ramped 0->lambda_max over training,
    standard DANN schedule"). gamma=10 is the DANN paper's own standard
    constant; MODEL_CANDIDATES.md doesn't give a number for it -- ASSUMPTION,
    documented here in the same style as reproduction/config.py's ASSUMPTION
    defaults, and not separately tuned for this Stage-1 smoke test."""
    p = max(0.0, min(1.0, progress))
    return float(lambda_max * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0))


class DomainClassifierHead(nn.Module):
    """g(H) -> generator_id logits, a small 2-layer MLP (MODEL_CANDIDATES.md
    Candidate 1: "a small classifier head")."""

    def __init__(self, in_dim: int, hidden: int, n_generators: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_generators))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# ---------------------------------------------------------------------------
# Candidate 2 / 4: multi-scale BiLSTM+attention branch.
# MODEL_CANDIDATES.md Candidate 2: H_seq^(s) = AttnPool(BiLSTM(AvgPool_s(h))),
# s in {1,3,12} slots/token.
# ---------------------------------------------------------------------------
def avg_pool_seq(h: torch.Tensor, mask: torch.Tensor, pool_size: int):
    """Non-overlapping mean-pool of encoder output h (B,N,D) and its bool
    mask (B,N) down to (B, N/pool_size, D) / (B, N/pool_size).

    Mask-downsampling rule (MODEL_CANDIDATES.md doesn't specify one --
    documented choice): a pooled token is 'active' if ANY of its constituent
    raw slots are active, extending mil_ids.py's "inactive slots excluded
    from attention" rule to coarser tokens. The coarse token's pooled FEATURE
    value is the plain mean over its pool_size raw-slot encoder outputs
    (including any inactive raw slots inside a partially-active window) --
    the attention step then masks the whole pooled token, not individual
    raw-slot contributions within it."""
    B, N, D = h.shape
    assert N % pool_size == 0, f"N={N} not divisible by pool_size={pool_size}"
    n_tok = N // pool_size
    h_pooled = h.view(B, n_tok, pool_size, D).mean(dim=2)
    mask_pooled = mask.view(B, n_tok, pool_size).any(dim=2)
    return h_pooled, mask_pooled


class ScaleBranch(nn.Module):
    """One resolution branch: AvgPool_s -> BiLSTM -> AttentionPooling."""

    def __init__(self, encoder_out: int, lstm_hidden_per_dir: int, pool_size: int, dropout: float):
        super().__init__()
        self.pool_size = pool_size
        self.bilstm = nn.LSTM(encoder_out, lstm_hidden_per_dir, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.attn = AttentionPooling(2 * lstm_hidden_per_dir)

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        h_p, mask_p = avg_pool_seq(h, mask, self.pool_size)
        z, _ = self.bilstm(h_p)
        z = self.dropout(z)
        H_seq, alpha = self.attn(z, mask_p)
        return H_seq, alpha  # alpha is at this branch's own (coarser) token resolution


# ---------------------------------------------------------------------------
# Candidate 2 / 4: Supervised Contrastive loss (SupCon, Khosla et al. 2020).
# Binary group (benign vs. attack) only -- MODEL_CANDIDATES.md Candidate 2
# explicitly permits this simplification for a smoke test ("benign-vs-attack
# as the two groups is sufficient for a smoke test; the finer attack-subtype
# separation ... can be added later if time allows"). The finer attack-
# subtype variant (using attention-argmax slot region as a coarse proxy) is
# SKIPPED here -- documented, not silently dropped.
# ---------------------------------------------------------------------------
def supcon_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """z: (B, d) fused embeddings (L2-normalized internally); labels: (B,) in
    {0,1}. Returns 0.0 if no anchor in the batch has another same-label,
    non-self partner (guards the small/imbalanced-batch degenerate case that
    MODEL_CANDIDATES.md's Candidate 2 'Expected weakness' flags)."""
    device = z.device
    B = z.shape[0]
    if B < 2:
        return z.new_zeros(())
    z = F.normalize(z, dim=-1)
    sim = torch.matmul(z, z.T) / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(B, dtype=torch.bool, device=device)
    logits_mask = ~eye
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    lab = labels.view(-1, 1)
    pos_mask = (lab == lab.T) & logits_mask
    pos_counts = pos_mask.sum(dim=1)
    valid = pos_counts > 0
    if valid.sum() == 0:
        return z.new_zeros(())
    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1)[valid] / pos_counts[valid].float()
    return -mean_log_prob_pos.mean()


# ---------------------------------------------------------------------------
# Candidate 3 (SSM-MIL): minimal selective state-space (Mamba-lite) layer,
# pure PyTorch, uni-directional. MODEL_CANDIDATES.md Candidate 3 "Core math":
#   x_i = Abar(u_i)*x_{i-1} + Bbar(u_i)*h_i ,  y_i = C*x_i
# with Abar, Bbar input-selective discretized SSM params.
#
# `mamba_ssm` is NOT installed in this environment (verified: `import
# mamba_ssm` -> ModuleNotFoundError). Per the task brief we implement a small
# selective-SSM directly instead of installing an unvetted external package.
# This is a NAIVE sequential-scan reference implementation (a plain Python
# loop over the 288 slots) -- not the parallelized associative-scan CUDA
# kernel real Mamba uses -- documented simplification, adequate at N=288 for
# Stage-1 smoke testing (see SelectiveSSMLayer docstring).
# ---------------------------------------------------------------------------
class SelectiveSSMLayer(nn.Module):
    """Diagonal selective-SSM block ("S4/Mamba-lite"): input-dependent
    discretization step (delta), input-dependent B/C projections (the
    "selection mechanism"), a diagonal (per-channel) state matrix A held
    negative via -softplus(A_log) for numerical stability, and a SiLU-style
    multiplicative gate on the output (Mamba block convention). Forward-only
    (uni-directional) by construction, per Candidate 3's rationale: BiLSTM's
    backward pass needs the full sequence, so it cannot honestly stream;
    a forward-only SSM can emit y_t from slots <= t only."""

    def __init__(self, d_model: int, d_state: int = 8, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.delta_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.A_log = nn.Parameter(torch.zeros(d_model, d_state))
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (B, N, d_model)
        Bsz, N, D = u.shape
        delta = F.softplus(self.delta_proj(u))  # (B,N,D), input-selective discretization step
        Bp = self.B_proj(u)  # (B,N,d_state)
        Cp = self.C_proj(u)  # (B,N,d_state)
        A = -F.softplus(self.A_log)  # (D,d_state), stable (real part < 0)

        # Abar_t = exp(delta_t (x) A)   (B,N,D,d_state) -- ZOH-style discretization
        A_bar = torch.exp(delta.unsqueeze(-1) * A.view(1, 1, D, self.d_state))
        # Bbar_t . u_t = delta_t * B_t (x) u_t   (B,N,D,d_state)
        Bu = delta.unsqueeze(-1) * Bp.unsqueeze(2) * u.unsqueeze(-1)

        state = u.new_zeros(Bsz, D, self.d_state)
        ys = []
        for t in range(N):  # sequential scan, see class docstring
            state = A_bar[:, t] * state + Bu[:, t]
            y_t = torch.einsum("bds,bs->bd", state, Cp[:, t])
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B,N,D)

        gate = torch.sigmoid(self.gate_proj(u))
        y = y * gate
        y = self.out_proj(y)
        y = self.dropout(y)
        return self.norm(u + y)  # residual + norm, standard block convention
