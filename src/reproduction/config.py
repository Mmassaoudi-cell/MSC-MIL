"""Hyperparameters for SOURCE_METHOD_REPRODUCTION.

Values are taken directly from the paper (Table I, Eq. 10-11, Sec. IV/V) where
specified. Values the paper never gives a number for are marked ASSUMPTION and
are tuned on the validation split rather than guessed once and frozen silently
-- see SOURCE_PAPER_AUDIT.md sec. "Partially Specified" for the exact list.
"""
from dataclasses import dataclass, field


@dataclass
class AttackerConfig:
    # --- from Table I (fully specified) ---
    gamma: float = 0.99
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    lr: float = 4e-3  # alpha_adv
    train_iters: int = 200  # PPO "epochs" (policy update iterations)
    optimizer: str = "adam"

    # --- architecture (Fig. 2 shows an LSTM-based actor-critic; hidden sizes
    # are not given anywhere in the paper -> ASSUMPTION, tuned on validation) ---
    state_dim: int = 3  # {T_start, T_end, E_demand}
    lstm_hidden: int = 64  # ASSUMPTION
    mlp_hidden: int = 64  # ASSUMPTION
    action_dim: int = 1  # scalar C_attack

    # --- PPO training loop mechanics not specified numerically by the paper ---
    rollout_len: int = 256  # ASSUMPTION
    minibatch_size: int = 64  # ASSUMPTION
    ppo_epochs_per_update: int = 4  # ASSUMPTION (standard PPO default)
    gae_lambda: float = 0.95  # ASSUMPTION (standard PPO default)
    value_coef: float = 0.5  # ASSUMPTION (standard PPO default)
    max_grad_norm: float = 0.5  # ASSUMPTION

    # --- attack semantics (Algorithm 1) — never numerically specified ---
    theta_thr: float = 0.15  # ASSUMPTION: stealth budget on induced deviation E_i (fraction of E_demand)
    c_adjust: float = 1.0  # ASSUMPTION: tuning factor in ACR_attack = C_attack * C_adjust
    compromise_ratio: float = 0.3  # ASSUMPTION: rho_c, fraction of active EVCPs compromised

    seed: int = 0


@dataclass
class IDSConfig:
    # --- from Sec. IV-C / Eq. 10-11 (fully specified) ---
    lr: float = 1e-3  # alpha_IDS
    l2_reg: float = 1e-4  # lambda_2 / lambda_reg
    batch_size: int = 32
    optimizer: str = "adam"

    # --- architecture (Eq. 3-9): hidden sizes "chosen via cross-validation",
    # no values given -> ASSUMPTION, tuned on validation split via Optuna ---
    n_slots: int = 288
    input_dim: int = 3  # per-slot raw feature: normalized {T_start,T_end,E_demand}-derived fields
    encoder_hidden: int = 64  # m in Eq. 4 (W1,W2 hidden size)
    encoder_out: int = 64  # embedding dim h_ji before BiLSTM
    lstm_hidden_per_dir: int = 64  # p in Eq. 3 (BiLSTM hidden size per direction)
    static_dim: int = 4  # F_static: connection time, disconnection time, requested energy, available duration
    static_hidden: int = 32  # m in Eq. 7
    dropout: float = 0.2  # ASSUMPTION ("dropout applied to BiLSTM outputs", no rate given)

    # --- training loop, not numerically specified beyond "200 epochs" shown in Fig. 7 ---
    max_epochs: int = 200
    early_stopping_patience: int = 20  # ASSUMPTION
    adasyn_k_neighbors: int = 5  # ASSUMPTION (ADASYN mentioned in Fig. 2, params never given)

    alert_threshold: float = 0.5  # tau, Sec. IV-C.6 -- ASSUMPTION default, tuned on validation

    seed: int = 0


@dataclass
class ReproductionConfig:
    attacker: AttackerConfig = field(default_factory=AttackerConfig)
    ids: IDSConfig = field(default_factory=IDSConfig)
    scma_types: tuple = (1, 2, 3, 4, 5, 6)
    dominance_threshold: float = 0.4  # Algorithm 1 lines 33-35
    test_ratio_attack: float = 0.10  # 180 normal : 20 attack per Sec. V-C.3
    n_test_per_type: int = 200
