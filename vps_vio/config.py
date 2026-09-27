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

    # --- VIO -> BODY axis mapping (MEASURED 27 Sep, no longer a placeholder) ---
    # Derived from geometry, not from drift-prone motion tests:
    #   * cam_z is DOWN  - the accelerometer's gravity vector, rotated into the
    #     camera frame by the IMU->CAM_B extrinsics, lands on cam_z at 99.8%.
    #   * the drone's nose points at the TOP of the image, and cam_y is image
    #     DOWN, so body forward = -vio_y.
    #   * body right = down x forward = +vio_x (right-handed, no measurement needed).
    # Cross-checked against the one clean motion run: a forward move gave
    # vio_y = -0.225 with 100% on that axis, exactly as this matrix predicts.
    # Rows map a VIO vector to [forward, right, down] in the body frame; the
    # compass supplies body->NED heading at runtime (VioToNed.heading_align_rad).
    R_vio_to_ned: list = field(default_factory=lambda: [[0.0, -1.0, 0.0],
                                                        [1.0,  0.0, 0.0],
                                                        [0.0,  0.0, 1.0]])

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
