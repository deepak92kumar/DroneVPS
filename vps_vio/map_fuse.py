"""Blend map fixes into the camera's dead reckoning, with an outlier gate.

AnchorFuser is the wrong tool for this. It snaps whenever the residual exceeds
reanchor_m, which is right for an ArUco marker - a marker fix is either correct
or absent - but wrong for a map fix, which is normally good to about ten metres
and occasionally out by a hundred. Snapping onto every fix measured WORSE than
using no map at all: median error went from 7.0 m to 12.3 m with spikes to 81 m,
because one bad fix threw the position further than the drift ever did.

So the map fix is treated as a noisy measurement instead of the truth:

  predict  each flow step grows the uncertainty by a fraction of the distance
           travelled, because flow error accumulates with distance, not time
  gate     a fix further away than the combined uncertainty allows is DROPPED,
           not snapped to - that is the whole point
  update   what survives is blended by its weight, so a confident estimate
           moves little and a drifted one moves a lot

State is horizontal only. Height stays with the barometer.
"""
from __future__ import annotations
import numpy as np


class MapFuser:
    def __init__(self, *,
                 flow_drift_frac: float = 0.04,
                 flow_noise_floor_m: float = 0.15,
                 fix_sigma_m: float = 30.0,
                 gate_sigmas: float = 3.0,
                 max_sigma_m: float = 400.0,
                 init_sigma_m: float = 2.0):
        """Defaults are swept on the 27 Sep flight, not assumed.

        fix_sigma_m is deliberately larger than the fixes actually are (they
        come in at a median 11.4 m): over a few hundred metres the flow is the
        better of the two sources, so the filter should lean on it and let the
        map correct slowly. On a long mission, where flow drift grows past the
        map's error, the same setting hands authority to the map on its own -
        that is what the growing variance is for."""
        self.pos = np.zeros(2)               # North, East, metres from origin
        self.var = float(init_sigma_m) ** 2
        self.k = float(flow_drift_frac)
        self.floor = float(flow_noise_floor_m)
        self.fix_var = float(fix_sigma_m) ** 2
        self.gate = float(gate_sigmas)
        self.max_var = float(max_sigma_m) ** 2
        self.n_fix = self.n_rejected = 0
        self.consec_rejected = 0
        self.last_residual_m = 0.0

    @property
    def sigma_m(self) -> float:
        return float(np.sqrt(self.var))

    def predict(self, d_north: float, d_east: float,
                trust: float = 1.0) -> np.ndarray:
        """Advance the estimate by one step. trust scales how fast it doubts.

        trust is 1.0 for a step optical flow measured. A step DEAD RECKONED
        from the accelerometer between frames deserves worse, because nothing
        outside the IMU saw it happen - the caller passes a larger number and
        the variance grows by that much more.
        """
        step = np.array([d_north, d_east], float)
        self.pos = self.pos + step
        grow = trust * (self.k * float(np.linalg.norm(step)) + self.floor)
        self.var = min(self.var + grow ** 2, self.max_var)
        return self.pos

    def update(self, fix_north: float, fix_east: float,
               fix_sigma_m: float | None = None) -> bool:
        """Fold in one map fix. Returns False if it was rejected as an outlier.

        The gate widens with every consecutive rejection. Without that the
        filter can lock itself out: once the estimate has drifted further than
        the gate allows, every correct fix looks like an outlier and is thrown
        away, so the drift is never corrected and the rejections continue
        forever. In the drift sweep this showed up as 30 of 41 fixes rejected
        with the error still growing - the one failure mode that turns a
        position aid into a position hazard. After a run of rejections the
        filter has to concede that the fixes are probably right and it is
        probably lost.
        """
        r = np.array([fix_north, fix_east], float) - self.pos
        d = float(np.linalg.norm(r))
        self.last_residual_m = d
        fv = self.fix_var if fix_sigma_m is None else float(fix_sigma_m) ** 2
        allowed = self.gate * (1.0 + self.consec_rejected) * np.sqrt(self.var + fv)
        if d > allowed:
            self.n_rejected += 1
            self.consec_rejected += 1
            return False
        self.consec_rejected = 0
        gain = self.var / (self.var + fv)
        self.pos = self.pos + gain * r
        # A fix accepted only because the gate had widened means the estimate
        # was lost, so the uncertainty must reflect that rather than collapsing.
        self.var = max((1.0 - gain) * self.var, min(fv, self.var))
        self.n_fix += 1
        return True
