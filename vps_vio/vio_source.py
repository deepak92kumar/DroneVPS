"""VIO pose sources.

A source yields VioPose samples: position (VIO world frame, metres), orientation
quaternion, a timestamp, and a tracking flag. Two implementations:

- CuVslamSource: real stereo(-inertial) VIO on the OAK-D via cuVSLAM, using the
  verified rig in oak_rig.py. Imports are lazy so this module loads anywhere.
- SyntheticSource: scripted motion + injectable drift/noise, so the fusion and
  node logic can be exercised with NO hardware (used by tests and dry-runs).

SpectacularAI was dropped: its free wheels are x86_64-only and non-commercial,
and ARM builds are commercial-licence-only. cuVSLAM replaced it — see
oak_rig.py for the measured constants.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
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

    def close(self):
        pass


class CuVslamSource:
    """Stereo(-inertial) VIO on the OAK-D Pro via cuVSLAM.

    read() pulls the next hardware-synced stereo pair, registers any IMU samples
    up to that image timestamp (cuVSLAM requires track and IMU calls to be
    ordered by timestamp), runs the tracker and returns the pose.
    """

    def __init__(self, inertial: bool = False, fps: int | None = None,
                 exposure_us: int | None = None, iso: int | None = None,
                 max_stereo_skew_ms: float = 5.0):
        self.inertial = inertial
        self._fps = fps
        self._exposure_us = exposure_us
        self._iso = iso
        self.max_skew_ms = max_stereo_skew_ms
        self._dev = None
        self._tracker = None
        self._pend_imu: deque = deque()
        self._last_imu_ts = None
        self._bufL: dict = {}
        self._bufR: dict = {}
        self.dropped = 0          # stereo pairs rejected as unsynced

    def open(self):
        import depthai as dai
        import cuvslam as vslam
        import oak_rig

        vslam.warm_up_gpu()
        self._dai = dai
        self._dev = dai.Device()
        calib = self._dev.readCalibration()
        rig = oak_rig.build_rig(calib, with_imu=self.inertial)
        self._tracker = oak_rig.make_tracker(rig, inertial=self.inertial)
        self._td_ns = oak_rig.td_ns
        kw = {}
        if self._fps is not None:
            kw["fps"] = self._fps
        if self._exposure_us is not None:
            kw["exposure_us"] = self._exposure_us
        if self._iso is not None:
            kw["iso"] = self._iso
        self._dev.startPipeline(oak_rig.build_pipeline(with_imu=self.inertial, **kw))
        self._qL = self._dev.getOutputQueue("left", 8, False)
        self._qR = self._dev.getOutputQueue("right", 8, False)
        self._qI = self._dev.getOutputQueue("imu", 300, False) if self.inertial else None
        return self

    def _drain_imu(self):
        if self._qI is None:
            return
        while self._qI.has():
            for pk in self._qI.get().packets:
                a, g = pk.acceleroMeter, pk.gyroscope
                self._pend_imu.append((self._td_ns(a.getTimestampDevice()),
                                       [a.x, a.y, a.z], [g.x, g.y, g.z]))

    def _next_pair(self, timeout_s: float):
        """Return the next hardware-synced stereo pair, or None on timeout."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            self._drain_imu()
            while self._qL.has():
                f = self._qL.get(); self._bufL[f.getSequenceNum()] = f
            while self._qR.has():
                f = self._qR.get(); self._bufR[f.getSequenceNum()] = f
            common = sorted(set(self._bufL) & set(self._bufR))
            for seq in common:
                fl, fr = self._bufL.pop(seq), self._bufR.pop(seq)
                ts = self._td_ns(fl.getTimestampDevice())
                if abs(ts - self._td_ns(fr.getTimestampDevice())) / 1e6 > self.max_skew_ms:
                    self.dropped += 1
                    continue
                return ts, fl.getCvFrame(), fr.getCvFrame()
            # never let one stalled side grow the buffers without bound
            for b in (self._bufL, self._bufR):
                while len(b) > 20:
                    b.pop(sorted(b)[0]); self.dropped += 1
            time.sleep(0.002)
        return None

    def read(self, timeout_s: float = 2.0) -> VioPose | None:
        import cuvslam as vslam
        got = self._next_pair(timeout_s)
        if got is None:
            return None
        ts, left, right = got

        if self.inertial:
            # cuVSLAM needs track() and register_imu_measurement() ordered by time
            while self._pend_imu and self._pend_imu[0][0] <= ts:
                its, la, av = self._pend_imu.popleft()
                if self._last_imu_ts is not None and its < self._last_imu_ts:
                    continue
                m = vslam.ImuMeasurement()
                m.timestamp_ns = its
                m.linear_accelerations = la
                m.angular_velocities = av
                self._tracker.register_imu_measurement(0, m)
                self._last_imu_ts = its

        est, _ = self._tracker.track(ts, [left, right])
        pose = getattr(est, "world_from_rig", None) if est is not None else None
        pose = getattr(pose, "pose", pose)
        if pose is None or not np.all(np.isfinite(pose.translation)):
            return VioPose(ts / 1e9, np.zeros(3), np.array([1.0, 0, 0, 0]), tracking=False)

        q = np.asarray(pose.rotation, float)          # cuVSLAM gives x, y, z, w
        quat_wxyz = np.array([q[3], q[0], q[1], q[2]])
        return VioPose(ts / 1e9, np.asarray(pose.translation, float), quat_wxyz, tracking=True)

    def close(self):
        try:
            if self._dev is not None:
                self._dev.close()
        except Exception:
            pass
        self._dev = None
