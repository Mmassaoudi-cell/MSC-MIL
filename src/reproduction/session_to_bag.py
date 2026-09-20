"""Bridges real ACN-Data sessions <-> Algorithm 1 aggregate attacks <-> the
288-slot MIL bag representation the IDS consumes.

Reconstruction note: Algorithm 1 only manipulates the three AGGREGATE session
fields (T_start, T_end, E_demand) -- it never specifies how a changed
aggregate rewrites the intra-session power *curve* shape. The paper's own
Eq. 2 defines the charging rate as a flat R_t = E_demand*(60/dt)/T_avail over
the available window, so the most faithful non-invented reconstruction of an
attacked slot-level bag is: replace the slots inside the (possibly shifted)
active window with the flat new rate R_new, and mask/unmask slots according
to the new [T_start, T_end] window. This is documented here rather than
silently assumed. Original (benign) bags use the REAL measured per-slot power
curve from the telemetry, not a flat reconstruction -- only attacked bags are
reconstructed, since only they are synthetic by construction.
"""
import numpy as np

from .scma_env import Session, generate_scma, baseline_rate

N_SLOTS = 288
DT_MIN = 5.0


def session_from_summary(connection_time, disconnection_time, kwh_delivered) -> Session:
    duration_h = max((disconnection_time - connection_time).total_seconds() / 3600.0, 1e-3)
    t_avail_h = min(duration_h, 24.0)
    return Session(t_start_h=0.0, t_end_h=t_avail_h, e_demand_kwh=float(kwh_delivered), t_avail_h=t_avail_h, dt_min=DT_MIN)


def build_benign_bag(real_power_kw: np.ndarray, real_active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """real_power_kw, real_active: (288,) arrays straight from session_slots.parquet."""
    assert real_power_kw.shape == (N_SLOTS,)
    return real_power_kw.astype(np.float32), real_active.astype(bool)


def build_attacked_bag(
    orig_session: Session, scma_type: int, c_attack: float, c_adjust: float
) -> tuple[np.ndarray, np.ndarray, Session]:
    """Returns (power_kw[288], active[288], attacked_session)."""
    attacked = generate_scma(orig_session, scma_type, c_attack, c_adjust)
    r_new = attacked.e_demand_kwh / max(attacked.t_avail_h, 1e-6)

    slot_h = np.arange(N_SLOTS) * (DT_MIN / 60.0)
    active = (slot_h >= attacked.t_start_h) & (slot_h < attacked.t_end_h)
    power = np.where(active, r_new, 0.0).astype(np.float32)
    return power, active.astype(bool), attacked


def slot_static_features(sess: Session, connection_hour_of_day: float) -> np.ndarray:
    """F_static (Eq.7): connection time, disconnection time, requested energy, available duration.
    We use (hour-of-day of connection, t_end_h, e_demand_kwh, t_avail_h) -- 4-dim, matching
    IDSConfig.static_dim, and normalize downstream in the Dataset class."""
    return np.array(
        [connection_hour_of_day, sess.t_end_h, sess.e_demand_kwh, sess.t_avail_h], dtype=np.float32
    )
