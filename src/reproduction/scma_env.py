"""SCMA generation (Algorithm 1) + PPO MDP wrapper (Sec. IV-A/B).

IMPORTANT reconstruction note (documented per SOURCE_PAPER_AUDIT.md policy of
never silently inventing missing detail): Algorithm 1 lines 24 and 30 in the
source PDF reference a quantity "T_{t+Delta t}" on the right-hand side of the
end-time/start-time update without ever defining it as a rate (it's used
inconsistently as both a rate-like and a time-like symbol across lines 19-24
of the printed algorithm, which is very likely an OCR/formatting artifact in
the paper's own listing rather than a real ambiguity in the authors' method).
We reconstruct an internally-consistent version:

  Ehat_demand = clip(ACR_attack * T_avail * (dt/60) - R_t * (dt/60) + E_demand, min=0)   # line 18/22/28
  R_new       = Ehat_demand / T_avail                                                     # line 19/23/29
  T_avail_new = Ehat_demand / max(R_t, eps)     # implied available-time under the *old* rate at the *new* demand
  delta       = |T_avail - T_avail_new|

  SCMA-1 (demand only):      E_demand <- Ehat_demand, T_start/T_end unchanged, rate <- R_new
  SCMA-2 (end-time, earlier): T_end   <- T_end - delta   (Sec. IV-A: "shifts completion times earlier")
  SCMA-3 (start-time, later): T_start <- T_start + delta  (Sec. IV-A: "delays charging initiation")
  SCMA-4/5/6: combinations per Algorithm 1 lines 33-36, dominance threshold 0.4.

This reconstruction is used consistently for both attack generation and any
downstream analysis; it is flagged here rather than presented as verbatim
paper text.
"""
from dataclasses import dataclass

import numpy as np

EPS = 1e-6


@dataclass
class Session:
    """One EV charging session's scheduling-relevant fields (paper's 3 core fields)."""
    t_start_h: float  # hours from session anchor (connection time)
    t_end_h: float  # hours from anchor (>= t_start_h)
    e_demand_kwh: float
    t_avail_h: float  # available charging duration; paper Eq. computes from t_start/t_end/availability matrix
    dt_min: float = 5.0  # Delta t in minutes


def baseline_rate(sess: Session) -> float:
    """Eq. 2: R_t = E_demand * (60/dt) / T_avail."""
    return sess.e_demand_kwh * (60.0 / sess.dt_min) / max(sess.t_avail_h, EPS)


def apply_demand_attack(sess: Session, c_attack: float, c_adjust: float) -> tuple[float, float, float]:
    """Shared attack-demand computation used by all SCMA types (lines 18/22/28)."""
    acr = c_attack * c_adjust
    r_t = baseline_rate(sess)
    dt_h = sess.dt_min / 60.0
    e_hat = acr * sess.t_avail_h * dt_h - r_t * dt_h + sess.e_demand_kwh
    e_hat = max(e_hat, 0.0)
    r_new = e_hat / max(sess.t_avail_h, EPS)
    t_avail_new = e_hat / max(r_t, EPS)
    delta = abs(sess.t_avail_h - t_avail_new)
    return e_hat, r_new, delta


def apply_scma1(sess: Session, c_attack: float, c_adjust: float) -> Session:
    e_hat, _, _ = apply_demand_attack(sess, c_attack, c_adjust)
    return Session(sess.t_start_h, sess.t_end_h, e_hat, sess.t_avail_h, sess.dt_min)


def apply_scma2(sess: Session, c_attack: float, c_adjust: float) -> Session:
    e_hat, _, delta = apply_demand_attack(sess, c_attack, c_adjust)
    new_end = max(sess.t_end_h - delta, sess.t_start_h + EPS)
    return Session(sess.t_start_h, new_end, e_hat, max(new_end - sess.t_start_h, EPS), sess.dt_min)


def apply_scma3(sess: Session, c_attack: float, c_adjust: float) -> Session:
    e_hat, _, delta = apply_demand_attack(sess, c_attack, c_adjust)
    new_start = min(sess.t_start_h + delta, sess.t_end_h - EPS)
    return Session(new_start, sess.t_end_h, e_hat, max(sess.t_end_h - new_start, EPS), sess.dt_min)


def apply_mixed(sess: Session, c_attack: float, c_adjust: float, modes: tuple[str, ...]) -> Session:
    """SCMA-4/5/6: apply the requested combination of demand/end-time/start-time edits."""
    out = sess
    if "demand" in modes:
        out = apply_scma1(out, c_attack, c_adjust)
    if "end" in modes:
        out = apply_scma2(out, c_attack, c_adjust)
    if "start" in modes:
        out = apply_scma3(out, c_attack, c_adjust)
    return out


SCMA_MODES = {
    1: ("demand",),
    2: ("end",),
    3: ("start",),
    4: ("demand", "end"),      # dominance: |EV_hd|,|EV_he| >= 0.4|EV|
    5: ("demand", "start"),    # dominance: |EV_hd|,|EV_hs| >= 0.4|EV|
    6: ("demand", "end", "start"),  # balanced, all <= 0.4|EV|
}


def generate_scma(sess: Session, scma_type: int, c_attack: float, c_adjust: float) -> Session:
    modes = SCMA_MODES[scma_type]
    if len(modes) == 1:
        fn = {"demand": apply_scma1, "end": apply_scma2, "start": apply_scma3}[modes[0]]
        return fn(sess, c_attack, c_adjust)
    return apply_mixed(sess, c_attack, c_adjust, modes)


def induced_deviation(orig: Session, attacked: Session) -> float:
    """E_i in the reward (Sec. IV-B.2): normalized deviation of the manipulated
    profile from the original, used for the attacker's own stealth budget."""
    e_dev = abs(attacked.e_demand_kwh - orig.e_demand_kwh) / max(orig.e_demand_kwh, EPS)
    t_dev = (abs(attacked.t_start_h - orig.t_start_h) + abs(attacked.t_end_h - orig.t_end_h)) / max(
        orig.t_avail_h, EPS
    )
    return 0.5 * e_dev + 0.5 * t_dev  # ASSUMPTION: paper doesn't give the exact E_i formula, only R_t = +-1 vs theta_thr


class SCMAAttackEnv:
    """Per-session MDP (Sec. IV-B): state={T_start,T_end,E_demand}, action=C_attack scalar,
    reward = +1 if induced_deviation <= theta_thr else -1. One "step" = one session's attack decision;
    an episode is a sequence of sessions (drawn from the compromised subset) fed in temporal order,
    consistent with the LSTM-based policy shown in Fig. 2."""

    def __init__(self, sessions: list[Session], scma_type: int, theta_thr: float, c_adjust: float):
        self.sessions = sessions
        self.scma_type = scma_type
        self.theta_thr = theta_thr
        self.c_adjust = c_adjust
        self.idx = 0

    def reset(self):
        self.idx = 0
        return self._state()

    def _state(self) -> np.ndarray:
        s = self.sessions[self.idx]
        return np.array([s.t_start_h, s.t_end_h, s.e_demand_kwh], dtype=np.float32)

    def step(self, c_attack: float):
        orig = self.sessions[self.idx]
        attacked = generate_scma(orig, self.scma_type, float(c_attack), self.c_adjust)
        e_i = induced_deviation(orig, attacked)
        # REWARD SHAPING NOTE (documented deviation from the paper's literal Sec. IV-B.3
        # formula R_t = +1 if E_i<=theta_thr else -1): taken literally, that reward has a
        # degenerate global optimum at C_attack such that e_i=0 (a no-op "attack" trivially
        # satisfies E_i<=theta_thr for any theta_thr>=0 and gets the same +1 as any other
        # in-budget attack) -- contradicting the paper's own framing of the agent as
        # maximizing "stealthy manipulative effect" (Sec. III-A) subject to the stealth
        # budget, not minimizing effect. We reward magnitude of deviation up to the budget
        # (encouraging maximal impact while remaining under threshold) and keep the same
        # -1 penalty outside it, which preserves the paper's threshold semantics at the
        # boundary while removing the trivial zero-attack optimum.
        if e_i <= self.theta_thr:
            reward = float(np.clip(e_i / max(self.theta_thr, EPS), 0.0, 1.0))
        else:
            reward = -1.0
        self.idx += 1
        done = self.idx >= len(self.sessions)
        next_state = self._state() if not done else np.zeros(3, dtype=np.float32)
        info = {"attacked_session": attacked, "orig_session": orig, "induced_deviation": e_i}
        return next_state, reward, done, info
