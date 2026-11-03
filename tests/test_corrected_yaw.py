"""One corrected yaw, used by all three camera-frame consumers.

Having two was the bug. Until 2 Nov shift_to_ne took the raw FC yaw while
only the map matcher's submit added camera_yaw_offset_deg, and accel_to_ne
had the same fault for three hours longer. Nothing caught either, because no
test compared the call sites. This one does.

All three consume CAMERA-FRAME measurements - the flow's pixel shift, and
the acceleration after _to_cam has put the BNO086 into the camera's axes -
so all three need the same corrected heading.

Runtime evidence that _to_cam is real and not the no-op fallback: the node
logs "IMU->CAM_C rotation read from calibration (179.6 deg off the camera
frame)" on every start, 437 of them in vps_stdout.log, and the two failure
warnings have never been printed once.
"""
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "vps_vio"))

fails = []


def check(what, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + what
          + (f"   [{detail}]" if detail else ""))
    if not ok:
        fails.append(what)


from live_node import accel_to_ne  # noqa: E402
import config as C  # noqa: E402

src = open(os.path.join(ROOT, "vps_vio", "live_node.py"),
           encoding="utf-8").read()


class _Stub:
    """A VpsNode built the way the other tests build one: no __init__."""
    def __init__(self, off):
        self.cam_yaw_offset_deg = off
    from live_node import VpsNode as _V
    corrected_yaw = _V.corrected_yaw


print("1) the helper itself")
check("offset is added, in degrees", abs(_Stub(-4.42).corrected_yaw({"yaw": 10.0}) - 5.58) < 1e-9,
      str(_Stub(-4.42).corrected_yaw({"yaw": 10.0})))
check("a positive offset goes the other way",
      abs(_Stub(+4.42).corrected_yaw({"yaw": 10.0}) - 14.42) < 1e-9)
check("zero offset is the raw yaw",
      _Stub(0.0).corrected_yaw({"yaw": 123.4}) == 123.4)
check("no yaw gives None, not a wrong number",
      _Stub(-4.42).corrected_yaw({"yaw": None}) is None)


class _NoAttr:
    from live_node import VpsNode as _V
    corrected_yaw = _V.corrected_yaw


check("a stub without the attribute falls back to 0, not a crash",
      _NoAttr().corrected_yaw({"yaw": 7.0}) == 7.0)

print("\n2) all three call sites use it, and none keeps a raw yaw")
check("accel_to_ne", "accel_to_ne(ax, ay, self.corrected_yaw(f), az," in src)
check("shift_to_ne", "yaw_deg=self.corrected_yaw(f),\n"
      "                                 focal_y_px=self.focal_y)" in src)
check("the matcher's submit",
      "yaw_deg=self.corrected_yaw(f)," in src)
check("no camera-frame consumer is left on the raw yaw",
      'accel_to_ne(ax, ay, f["yaw"]' not in src
      and 'yaw_deg=f["yaw"],' not in src)
check("exactly three consumers", src.count("self.corrected_yaw(f)") == 3,
      str(src.count("self.corrected_yaw(f)")))

print("\n3) zero offset reproduces the old accel output exactly")
rng = np.random.default_rng(11)
worst = 0.0
for _ in range(3000):
    ax, ay, az = rng.normal(0, 3, 3)
    yaw, r, p = rng.uniform(-180, 180), rng.uniform(-30, 30), rng.uniform(-30, 30)
    a = accel_to_ne(ax, ay, yaw, az, r, p)
    b = accel_to_ne(ax, ay, _Stub(0.0).corrected_yaw({"yaw": yaw}), az, r, p)
    worst = max(worst, abs(a[0] - b[0]), abs(a[1] - b[1]))
check("3000 random samples are bit-identical", worst == 0.0, f"worst {worst:g}")

print("\n4) headings 0/90/180/270, positive and negative offset")
for off in (-4.42, +4.42):
    for yaw in (0.0, 90.0, 180.0, 270.0):
        n0, e0 = accel_to_ne(1.0, 0.0, yaw, 0.0, 0.0, 0.0)
        n1, e1 = accel_to_ne(1.0, 0.0, yaw + off, 0.0, 0.0, 0.0)
        d = (math.degrees(math.atan2(e1, n1) - math.atan2(e0, n0)) + 180) % 360 - 180
        check(f"offset {off:+.2f} at {yaw:3.0f} deg rotates by {off:+.2f}",
              abs(d - off) < 1e-6, f"{d:+.4f}")

print("\n5) the value stays configurable")
# The value must come from the config, never a literal in the node. Comments
# quote the measured number freely - that is the record of where it came from
# - so this looks at CODE, with comments stripped.
import ast as _ast
_code = "\n".join(l.split("#")[0] for l in src.splitlines())
check("the node assigns it only from the config",
      'float(_c.camera_yaw_offset_deg)' in _code)
check("and no numeric literal of it is written in the node's code",
      "4.42" not in _code, [l.strip() for l in _code.splitlines()
                            if "4.42" in l][:2])
check("and the config holds the measured value",
      abs(C.VpsConfig().camera_yaw_offset_deg + 4.42) < 1e-9)

print("\nALL PASS" if not fails else f"\n{len(fails)} FAILED: {', '.join(fails)}")
sys.exit(1 if fails else 0)
