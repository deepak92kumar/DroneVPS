"""Does the velocity actually reach the sender, and is it the right number.

The pieces are tested separately elsewhere - the bridge against the 8 Oct
collapse, the ignore flags against every case. This one tests the wiring:
frames in at a known pixel shift, and a velocity out of VpsNode.step() that
matches the ground speed that shift implies.

Synthetic frames, a stub map cache, no hardware and no GeoTIFF.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "vps_vio"))
from live_node import VpsNode               # noqa: E402

FOCAL = 791.72        # measured on this OAK-D
ALT = 110.0
HZ = 25.0
SHIFT_PX = 2.0        # per frame
GSD = ALT / FOCAL                       # m per pixel on the ground
EXPECTED = SHIFT_PX * GSD * HZ          # m/s

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


class StubCache:
    """Enough of MapCache for VpsNode to run with the map path idle."""
    gsd = 0.509

    def covers(self, lon, lat):
        return False                    # no fixes; this test is about velocity

    def window(self, lon, lat, radius_m, kind="grey"):
        raise AssertionError("should not be reached")


# A textured scene the LK tracker can actually hold on to.
rng = np.random.default_rng(7)
base = rng.integers(0, 255, (700, 700), dtype=np.uint8)
base = cv2.GaussianBlur(base, (5, 5), 0)

node = VpsNode(StubCache(), FOCAL, fix_period_s=1e9)
node.set_origin(74.6072, 30.7844)       # the site in SCR_USER1/2

print(f"expecting {EXPECTED:.2f} m/s "
      f"({SHIFT_PX} px/frame x {GSD:.4f} m/px x {HZ:.0f} Hz)")

speeds = []
for i in range(40):
    # Crop a window that slides by SHIFT_PX each frame: the scene translates
    # under a camera moving at a constant ground speed.
    y = 200 + int(round(i * SHIFT_PX))
    img = base[y:y + 400, 200:600].copy()
    f = dict(img=img, alt=ALT, yaw=0.0, roll=0.0, pitch=0.0,
             gps=None, t=i / HZ)
    est = node.step(f)
    if est is None:
        continue
    v = est["vel"]
    if v.valid:
        speeds.append(v.speed_mps)

print(f"  frames stepped, {len(speeds)} valid velocity samples")
check("velocity became valid", len(speeds) > 10, f"{len(speeds)} samples")

if speeds:
    settled = speeds[len(speeds) // 2:]
    got = float(np.median(settled))
    check("settled speed matches the shift",
          abs(got - EXPECTED) < 0.25 * EXPECTED,
          f"{got:.2f} m/s vs expected {EXPECTED:.2f} "
          f"({100*got/EXPECTED:.0f}%)")
    check("not attenuated the way the old bridge was",
          got > 0.75 * EXPECTED,
          f"{100*got/EXPECTED:.0f}% of expected; the 8 Oct module gave 40%")

est = node.step(dict(img=base[200:600, 200:600].copy(), alt=ALT, yaw=0.0,
                     roll=0.0, pitch=0.0, gps=None, t=40 / HZ))
v = est["vel"]
print("\nthe state the recorder now carries")
for k in ("vn", "ve", "speed_mps", "valid", "sent_vn", "sent_ve", "deadbanded",
          "deadband_mps", "age_s", "flow_updates", "bias_n", "bias_frozen",
          "reason"):
    print(f"    {k:<14} {getattr(v, k)!r}")
check("flow_updates counted", v.flow_updates > 10, f"{v.flow_updates}")
check("bias stayed frozen through the run", v.bias_frozen is True, "")
check("deadband scaled with height", abs(v.deadband_mps - (0.35 + 0.005 * ALT)) < 1e-9,
      f"{v.deadband_mps:.3f} at {ALT:.0f} m")

node.stop()
print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
