"""The camera's own FeatureTracker as a flow backend, and what it is worth.

MEASURED 2 NOV, same camera, same resolution, 20 s each:

    host: goodFeatures + two-way LK   62.4 ms CPU/frame   1.85 cores
    device: FeatureTracker             1.7 ms CPU/frame   0.05 cores

97% less. Two trackers fly, so the host path ceilings at 8.02 Hz and the
device at 292 - against a loop measured at 9.06 Hz while the camera gave
24.7 fps, with the map matcher already in its own process and therefore not
the cause. Frames are what the velocity is made of: on flight 382 the phantom
was +0.770 m/s p50 where under 3 frames a second were dropped and +3.705
where 6 to 10 were.

WHAT IT COSTS. The device holds about 51 features against the 300 the host is
asked for. Scored on 962 of flight 382's saved frames against the 300-corner
answer ON THE SAME PAIR, 51 features is our 100-corner setting:

    corners  tracks  answered  vs 300 p50
      100      51     863/962    0.633 px
      300     100     935/962    0.000
      600     162     946/962    0.491

600 against 300 is 0.491 px, which is this comparison's own floor, so most of
100's 0.633 is not a disagreement.

WHAT IS NOT MEASURED, and it is the honest half. There is no
forward-backward check on the device path - the maintainer drops a feature on
its own tracking error, a different test. On the ground in daylight 95% of
ids carry to the next frame and features live 300 frames; at night on a
low-texture scene the identical configuration carried 30%. The air is neither
scene and none of this has flown, which is why flow_device is False.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vps_vio"))
from flow_odom import FlowTracker  # noqa: E402
from config import VpsConfig  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        fails.append(name)


class Feat:
    """What dai.TrackedFeature gives us: an id and a position."""

    def __init__(self, fid, x, y, age=5):
        self.id = fid
        self.age = age
        self.position = type("P", (), {"x": float(x), "y": float(y)})()


print("off until it has flown")
check("flow_device is False", VpsConfig().flow_device is False,
      "no forward-backward check, and nothing measured in the air")
check("and it asks for 300 features",
      VpsConfig().flow_device_features == 300,
      "the board clamps to about 51; asking for more costs nothing")

print("\nthe correspondence is the id, and it recovers a known shift")
rng = np.random.default_rng(5)
img = rng.integers(0, 255, (400, 640), dtype=np.uint8).astype(np.uint8)
pts = rng.random((60, 2)) * np.array([600.0, 380.0]) + 20.0
t = FlowTracker(use_gpu=False)
first = t.step_device(img, [Feat(i, *p) for i, p in enumerate(pts)])
check("the first frame has nothing to compare to", first[0] is None,
      f"{first}")
check("and it reports how many it saw", first[1] == 60)
sh, n = t.step_device(img, [Feat(i, p[0] + 3.0, p[1] - 2.0)
                            for i, p in enumerate(pts)])
got = np.asarray(sh, float).ravel()[:2]
check("a 3, -2 shift comes back as 3, -2",
      sh is not None and np.allclose(got, [3.0, -2.0], atol=0.01),
      f"{got.round(3)}")
check("over the ids that were shared", n == 60)

print("\nids that vanish are dropped, not matched by position")
# The previous frame left these at pts + (3, -2), so moving them to
# pts + (1, 0) is a shift of (-2, +2) - which is the point: the step is
# measured against where the ids WERE, not against where they started.
half = [Feat(i, p[0] + 1.0, p[1]) for i, p in enumerate(pts) if i % 2 == 0]
sh, n = t.step_device(img, half)
check("only the shared half is used", n == 30, f"{n}")
check("and the shift is measured from the last frame, not the first",
      np.allclose(np.asarray(sh, float).ravel()[:2], [-2.0, 2.0], atol=0.01),
      f"{np.asarray(sh, float).ravel()[:2].round(3)}")

print("\nnew ids with no history give no shift, and say so")
t2 = FlowTracker(use_gpu=False)
t2.step_device(img, [Feat(i, *p) for i, p in enumerate(pts)])
sh, n = t2.step_device(img, [Feat(1000 + i, *p) for i, p in enumerate(pts)])
check("nothing shared -> no shift", sh is None, f"{sh}")

print("\nempty features are a miss, not a crash")
sh, n = t2.step_device(img, [])
check("no features -> (None, 0)", sh is None and n == 0)

print("\nboth paths end in the SAME code")
src = open(os.path.join(os.path.dirname(__file__), "..", "vps_vio",
                        "flow_odom.py"), encoding="utf-8").read()
check("step_device calls _finish", "self._finish(img, dst, good," in src)
check("and so does step", "return self._finish(img, p1, good," in src)
check("_finish holds the depth gate",
      "_depth_gate" in src.split("def _finish")[1])

print("\nthe node keeps both cameras on ONE backend")
live = open(os.path.join(os.path.dirname(__file__), "..", "vps_vio",
                         "live_node.py"), encoding="utf-8").read()
check("the partner follows the primary's choice for this frame",
      '_fb = f.get("feats_partner") if _fe else None' in live,
      "flight 163: one camera derotated and the other not made "
      "flow_disagree_px measure the rotation, p50 0.06 -> 2.32 px")
check("a primary on device with no partner features reports none",
      "db, ntrk_b = None, 0" in live)

print("\nthe crop is taken off the device's positions, once")
check("_shift_features exists", "def _shift_features" in live)
check("and subtracts crop_left",
      "f.position.x -= c" in live,
      "the device sees the full sensor frame; everything else sees it cropped")

print()
if fails:
    print(f"{len(fails)} FAILED: " + ", ".join(fails))
    sys.exit(1)
print("ALL PASS")
