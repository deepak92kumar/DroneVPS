"""Rotation is removed from every track, not from the answer.

gyro_shift_px returns ONE vector and live_node subtracted it from the result.
That is exact only at the optical centre. A rotation moves a point at radius
r by f*theta*(1 + r^2/f^2), and this camera's field is 77.7 degrees:

    position            1 deg of rotation      against the flat 13.88 px
    centre                   13.88 px                   +0%
    v edge (400 px)          17.39 px                  +25%
    u edge (640 px)          22.87 px                  +65%

The tracker reduces ~400 tracks spread over the whole frame to one number, so
a single centre-valued subtraction leaves a residual. Over points spread
uniformly the median shortfall is +25.6%, and the residual MEASURED on two
flights, fitting step_px = a * rotation_px + b, was a = 0.157 and 0.317.

That is the same quantity, and it is why nothing else moved it: the
coefficient sweep went 0.026 -> 0.024 over 14.80 down to 13.40, and the
integration window change was bit-identical on 6400 frames. Both were
hunting a gain error inside a model error.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vps_vio"))
from flow_odom import (FlowTracker, gyro_shift_px,  # noqa: E402
                       gyro_shift_px_at)

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        fails.append(name)


C = (644.1290, 404.1647)
F = 795.0983

print("at the optical centre it IS the old shift - the gains did not change")
for dr, dp in ((1.0, 0.0), (0.0, 1.0), (-0.7, 0.4)):
    a = gyro_shift_px_at([C], dr, dp, centre_xy=C, focal_px=F)[0]
    b = gyro_shift_px(dr, dp)
    check(f"roll {dr:+.1f} pitch {dp:+.1f}", np.allclose(a, b, atol=1e-9),
          f"{a} vs {b}")

print("\naway from the centre it is LARGER, quadratically")
flat = abs(gyro_shift_px(1.0, 0.0)[0])
for off, want in ((400, 1.0), (640, 1.65)):
    a = gyro_shift_px_at([(C[0] - off, C[1])], 1.0, 0.0, centre_xy=C, focal_px=F)[0]
    ratio = np.hypot(*a) / flat
    check(f"{off} px from the centre along u", ratio > want - 0.06,
          f"{ratio:.3f}x the flat shift")
a = gyro_shift_px_at([(C[0], C[1] - 400)], 0.0, 1.0, centre_xy=C, focal_px=F)[0]
check("400 px along v, in pitch", np.hypot(*a) / abs(gyro_shift_px(0.0, 1.0)[1]) > 1.19,
      f"{np.hypot(*a) / abs(gyro_shift_px(0.0, 1.0)[1]):.3f}x")

print("\nzero rotation moves nothing, anywhere")
z = gyro_shift_px_at([(0, 0), (1279, 799), C], 0.0, 0.0, centre_xy=C, focal_px=F)
check("all zero", np.allclose(z, 0.0))

print("\nthe tracker takes it, and says whether it used it")
t = FlowTracker(K=[[F, 0, C[0]], [0, F, C[1]], [0, 0, 1]])
check("focal is available to curve the correction", abs(t.focal - F) < 1e-6)
check("the flag starts false", t.derot_applied is False)
src = open(os.path.join(os.path.dirname(__file__), "..",
                        "vps_vio", "flow_odom.py")).read()
check("step takes the rotation", "derot_deg=None" in src)
# Scoped to FlowTracker.step: estimateAffinePartial2D also appears earlier in
# the module, inside frame_shift_px, and the first occurrence is not this one.
_step = src[src.rindex("    def step(self, img: np.ndarray"):]
check("it is removed from the tracks, before the fit",
      "_sh = gyro_shift_px_at(" in _step
      and _step.index("_sh = gyro_shift_px_at(")
      < _step.index("M, inl = cv2.estimateAffinePartial2D"))
check("and the flag is cleared every frame",
      src.count("self.derot_applied = False") >= 1)

print("\nthe node must not then subtract it a second time")
node = open(os.path.join(os.path.dirname(__file__), "..",
                         "vps_vio", "live_node.py")).read()
check("the gyro goes INTO the tracker", "derot_deg=((_gr, _gp)" in node)
check("the old subtraction is conditional on the tracker NOT having done it",
      'if not getattr(self.tracker, "derot_applied", False):' in node,
      "otherwise the rotation comes off twice and the sign flips")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
