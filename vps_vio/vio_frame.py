"""VIO pose -> ArduPilot NED transform.

This is the ONE piece of the VPS upgrade that is pure math and can be built and
tested with no hardware. It converts a VIO engine's world-frame pose into the
North/East/Down convention that ArduPilot's VISION_POSITION_ESTIMATE expects,
and it applies a heading alignment (so VIO "forward" points at true North, the
same job Stride's "Auto VPS Origin / heading align" does with the compass).

IMPORTANT — bench verification pending (Phase 0):
    The default R_vio_to_ned is identity, i.e. it assumes the VIO engine already
    reports x=North, y=East, z=Down. Real engines do NOT: SpectacularAI/OpenVINS
    use their own world frame (often x-right/z-up or first-frame-aligned). The
    exact 3x3 axis-permutation matrix MUST be measured on the bench by moving the
    aircraft in known directions and reading the VIO output. Until then this
    module is structurally correct but the constant is a placeholder.

Nothing here talks to the flight controller or the camera. The daemon plug-in
point is cuav_mavlink_bridge.CuavMavlinkClient.send_vision_estimate(x,y,z,...).
"""
from __future__ import annotations
import numpy as np


def quat_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    """Heading (rotation about the vertical axis) from a quaternion, radians."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def yaw_rot(yaw_rad: float) -> np.ndarray:
    """Rotation about the Down axis in NED (positive yaw turns North toward East)."""
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def wrap_pi(a: float) -> float:
    """Wrap an angle to [-pi, pi). +pi and -pi are the same heading."""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


class VioToNed:
    """Rigid transform from a VIO world frame into ArduPilot NED.

    Pipeline for a raw VIO position p:
        p_ned = Rh @ (R @ p) - origin
    where R maps VIO axes to NED axes (bench-measured) and Rh rotates the
    horizontal plane so VIO heading lines up with the compass at init.
    """

    def __init__(self, R_vio_to_ned: np.ndarray | None = None,
                 heading_align_rad: float = 0.0):
        self.R = np.eye(3) if R_vio_to_ned is None else np.asarray(R_vio_to_ned, float)
        if self.R.shape != (3, 3):
            raise ValueError("R_vio_to_ned must be 3x3")
        self.heading_align_rad = float(heading_align_rad)
        self.Rh = yaw_rot(self.heading_align_rad)
        self.origin_ned: np.ndarray | None = None

    @classmethod
    def from_compass(cls, compass_yaw_rad: float, vio_yaw_at_init_rad: float,
                     R_vio_to_ned: np.ndarray | None = None) -> "VioToNed":
        """Build with heading alignment = (compass heading) - (VIO heading) at init."""
        return cls(R_vio_to_ned=R_vio_to_ned,
                   heading_align_rad=wrap_pi(compass_yaw_rad - vio_yaw_at_init_rad))

    def _raw_ned(self, pos_vio) -> np.ndarray:
        return self.Rh @ (self.R @ np.asarray(pos_vio, float))

    def set_origin(self, pos_vio) -> None:
        """Latch the takeoff point so subsequent positions are relative to it."""
        self.origin_ned = self._raw_ned(pos_vio)

    def to_ned(self, pos_vio) -> np.ndarray:
        """Return [north, east, down] in metres, origin-subtracted if latched."""
        ned = self._raw_ned(pos_vio)
        if self.origin_ned is not None:
            ned = ned - self.origin_ned
        return ned

    def yaw_to_ned(self, vio_yaw_rad: float) -> float:
        """Map a VIO heading into the NED/compass frame."""
        return wrap_pi(vio_yaw_rad + self.heading_align_rad)
