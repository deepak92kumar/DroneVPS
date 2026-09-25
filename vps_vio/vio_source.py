"""VIO pose sources.

A source yields VioPose samples: position (VIO world frame, metres), orientation
quaternion, a timestamp, and a tracking flag. Two implementations:

- SpectacularAiSource: real VIO on the OAK-D (Phase 1+). Imports are lazy so the
  module loads on any machine for testing the rest of the pipeline.
- SyntheticSource: scripted motion + injectable drift/noise, so the fusion and
  node logic can be exercised with NO hardware (used by tests and dry-runs).

The exact SpectacularAI field names are per its published API and must be
confirmed on the bench during Phase 0 (flagged below).
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np


@dataclass
class VioPose:
    t: float                 # seconds (source clock)
    pos: np.ndarray          # (3,) VIO world frame, metres
    quat: np.ndarray         # (4,) w, x, y, z
    tracking: bool           # True when the engine reports a healthy track


class SyntheticSource:
    """Deterministic motion for tests/dry-runs: a circle in the horizontal plane
    plus optional constant drift and gaussian noise, so we can prove the fusion
    corrects drift without a camera."""

    def __init__(self, radius=2.0, period_s=20.0, drift_mps=(0.0, 0.0, 0.0),
                 noise_m=0.0, rate_hz=20.0, seed=0):
        self.radius = radius
        self.period = period_s
        self.drift = np.asarray(drift_mps, float)
        self.noise = noise_m
        self.dt = 1.0 / rate_hz
        self.t = 0.0
        self.rng = np.random.default_rng(seed)

    def read(self) -> VioPose:
        w = 2 * np.pi / self.period
        true = np.array([self.radius * np.cos(w * self.t),
                         self.radius * np.sin(w * self.t), 0.0])
        pos = true + self.drift * self.t + self.rng.normal(0, self.noise, 3) * (self.noise > 0)
        yaw = w * self.t
        quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        p = VioPose(self.t, pos, quat, tracking=True)
        self.t += self.dt
        return p


class SpectacularAiSource:
    """Real VIO on the OAK-D. Lazy imports; construct only when a camera is free."""

    def __init__(self):
        self._device = None
        self._session = None

    def open(self):
        import depthai            # noqa
        import spectacularAI.depthai
        pipeline = depthai.Pipeline()
        vio = spectacularAI.depthai.Pipeline(pipeline)
        self._device = depthai.Device(pipeline)
        self._session = vio.startSession(self._device)

    def read(self) -> VioPose:
        out = self._session.waitForOutput()
        p = out.pose
        # NOTE (Phase 0 verify): confirm these attribute names against the
        # installed SpectacularAI version before trusting them in flight.
        pos = np.array([p.position.x, p.position.y, p.position.z])
        q = np.array([p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z])
        status = str(getattr(out, "status", "TRACKING"))
        return VioPose(p.time, pos, q, tracking=("TRACKING" in status.upper() or status == ""))

    def close(self):
        try:
            if self._device is not None:
                self._device.close()
        except Exception:
            pass
