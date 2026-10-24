"""A map fix describes the moment its image was taken, not the moment it lands.

The matcher is handed a position and an image, takes 0.098 s at 42 m and
0.319 s at 110 m to answer, and answers about THAT moment. Until 24 Oct the
fuser subtracted the fix from the CURRENT position, so everything the
estimate did while the matcher was thinking was folded into the correction.

It is not a rounding error. Over 0.1-0.34 s at the 0.8-2.8 m/s this node
reports, the estimate moves 0.3-0.95 m, against a correction the Kalman gain
holds to about 0.49 m. The contamination is the size of the correction, and
it comes from the phantom velocity - the very thing the map exists to bound.

FixResult has carried prior_lon/prior_lat since it was written, with a comment
saying the caller should "apply it as a correction to that, not to 'now'".
The caller never read them.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vps_vio"))
from map_fuse import MapFuser  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        fails.append(name)


def fuser(**kw):
    f = MapFuser(**kw)
    f.pos = __import__("numpy").array([0.0, 0.0])
    return f


print("the residual is measured against the prior, not against 'now'")
# The matcher looked at the world when we thought we were at the origin and
# said we were 10 m north of that. While it thought, we drifted 3 m north.
f = fuser()
f.var = 400.0
f.pos = __import__("numpy").array([3.0, 0.0])      # where we are now
f.update(10.0, 0.0, prior_north=0.0, prior_east=0.0)
with_prior = float(f.last_residual_m)
f2 = fuser()
f2.var = 400.0
f2.pos = __import__("numpy").array([3.0, 0.0])
f2.update(10.0, 0.0)                               # the old behaviour
without = float(f2.last_residual_m)
check("with the prior, the residual is what the matcher measured",
      abs(with_prior - 10.0) < 1e-6, f"{with_prior:.3f} m, and the matcher said 10")
check("without it, our own drift is subtracted from it",
      abs(without - 7.0) < 1e-6,
      f"{without:.3f} m - the 3 m we travelled while it was thinking is gone")
check("so the two differ by exactly that travel",
      abs((with_prior - without) - 3.0) < 1e-6)

print("\nthe correction still moves the PRESENT position")
f3 = fuser()
f3.var = 400.0
import numpy as np  # noqa: E402
f3.pos = np.array([3.0, 0.0])
before = f3.pos.copy()
f3.update(10.0, 0.0, prior_north=0.0, prior_east=0.0)
check("it is applied to where we are, not to where we were",
      f3.pos[0] > before[0],
      "whatever the aircraft did since is already in the present position")

print("\nomitting the prior keeps the old behaviour, so nothing else changes")
f4 = fuser()
f4.var = 400.0
f4.pos = np.array([3.0, 0.0])
f4.update(10.0, 0.0, prior_north=None, prior_east=None)
check("a caller that passes no prior is unaffected",
      abs(float(f4.last_residual_m) - 7.0) < 1e-6)

print("\nthe node passes the prior and logs how stale the fix was")
src = open(os.path.join(os.path.dirname(__file__), "..",
                        "vps_vio", "live_node.py")).read()
check("the prior reaches the fuser",
      "prior_north=pN, prior_east=pE" in src)
check("and it is the MATCHER's prior, not the current position",
      "res.prior_lat" in src and "res.prior_lon" in src)
check("the round trip is logged", '"fix_age_s"' in src)

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
