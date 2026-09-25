"""ArUco-as-anchor fusion for the VPS/VIO pipeline.

VIO gives a smooth position that slowly drifts. ArUco (when a marker is visible)
gives an absolute fix in the home/world frame. This fuser corrects the VIO drift
using the marker, exactly the role the live daemon's marker_fix_step plays for
the flow integrator — but here the smooth source is VIO, not optical flow.

Model (horizontal N/E; Z stays with the FC barometer, so it is passed through):
    world = vio_ned + offset
At takeoff the aircraft is the world origin, VIO is origin-subtracted, so
offset starts at zero. Each marker fix nudges `offset` to cancel accumulated
drift:
    residual = marker_world - (vio_ned + offset)
    |residual_horizontal| > reanchor_m  -> snap  (offset += residual)
    otherwise                            -> slew  (offset += clamp(residual, rate*dt))

The slew rate and reanchor distance are FLAT (independent of height). Scaling
them with altitude cost two flights and was reverted (see memory d75a29f). Do
not reintroduce height scaling here.
"""
from __future__ import annotations
import numpy as np


def _horiz_norm(v: np.ndarray) -> float:
    return float(np.hypot(v[0], v[1]))


class AnchorFuser:
    def __init__(self, corr_rate_mps: float = 4.0, reanchor_m: float = 6.0):
        if corr_rate_mps <= 0:
            raise ValueError("corr_rate_mps must be > 0")
        if reanchor_m <= 0:
            raise ValueError("reanchor_m must be > 0")
        self.corr_rate_mps = float(corr_rate_mps)
        self.reanchor_m = float(reanchor_m)
        self.offset = np.zeros(3)      # NED, world = vio_ned + offset
        self.reanchor_count = 0
        self.marker_count = 0
        self.last_residual = np.zeros(3)

    def fused(self, vio_ned) -> np.ndarray:
        """World-frame position the FC should see."""
        return np.asarray(vio_ned, float) + self.offset

    def update_marker(self, vio_ned, marker_world_ned, dt_s: float) -> np.ndarray:
        """Correct the offset from an absolute marker fix. Returns the residual
        that was observed (marker - current world estimate)."""
        vio_ned = np.asarray(vio_ned, float)
        marker = np.asarray(marker_world_ned, float)
        dt_s = max(float(dt_s), 0.0)
        residual = marker - (vio_ned + self.offset)
        self.last_residual = residual
        self.marker_count += 1

        if _horiz_norm(residual) > self.reanchor_m:
            # drift too large -> snap the whole offset (re-anchor)
            self.offset = self.offset + residual
            self.reanchor_count += 1
        else:
            # slew the offset toward the fix, flat rate limit on the step size
            max_step = self.corr_rate_mps * dt_s
            step = residual
            n = np.linalg.norm(step)
            if max_step > 0 and n > max_step:
                step = step * (max_step / n)
            self.offset = self.offset + step
        return residual
