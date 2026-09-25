"""All VPS/VIO tunables in one place. NOT active until Phase 1 (after demo).

Defaults mirror the proven marker-map behaviour from the live daemon. The two
correction constants MUST stay flat (never scale with height): scaling them cost
two flights and was reverted (see memory flow-velocity-above-40m / d75a29f).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class VpsConfig:
    # --- FC link (reuses the existing bridge; same values the daemon uses) ---
    fc_port: str = "/dev/ttyTHS1"
    fc_baud: int = 115200

    # --- VIO -> NED axis mapping (Phase-0 bench measured; identity is a PLACEHOLDER) ---
    R_vio_to_ned: list = field(default_factory=lambda: np.eye(3).tolist())

    # --- ArUco-as-anchor fusion (mirror of marker_fix_step; keep FLAT) ---
    corr_rate_mps: float = 4.0     # max slew speed of the offset correction, m/s. FLAT.
    reanchor_m: float = 6.0        # horizontal residual above which we snap. FLAT.
    marker_timeout_s: float = 2.0  # ignore marker fixes older than this

    # --- covariance sent to EKF (quality-dependent, like the daemon) ---
    pos_cov_good: float = 0.05     # m^2 when VIO tracking is healthy
    pos_cov_poor: float = 0.50     # m^2 when tracking is weak

    # --- VIO health gating ---
    min_vio_rate_hz: float = 15.0  # below this we stop trusting X/Y
    send_rate_hz: float = 20.0     # how often we push VISION_POSITION_ESTIMATE

    def R(self) -> np.ndarray:
        return np.asarray(self.R_vio_to_ned, float)
