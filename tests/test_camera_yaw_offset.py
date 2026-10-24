"""The camera's yaw on the airframe reaches the map query.

The mount was moved on 23 Oct. Measured 24 Oct two ways: a similarity fitted
to the translation inliers gives -2.03 deg on log 240 (spread 0.85) and
-1.87 deg on log 242 (spread 0.83), over different yaw ranges, with corr
against yaw of only -0.25 and +0.19 - a constant offset, not a heading-scale
error. Replaying locate() on real frames against the raw receiver decided the
SIGN, which the first attempt got wrong by testing +1.95 only:

    log 240:  -1.95 -> 23 fixes, 11.22 m | 0 -> 19, 11.38 | +1.95 -> 12, 12.58
    log 242:  -1.95 -> 16 fixes,  9.41 m | 0 -> 14,  9.70 | +1.95 -> 10,  8.19
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vps_vio"))
from config import VpsConfig  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        fails.append(name)


c = VpsConfig()
print("the offset is configured, and it is the measured sign")
check("the field exists", hasattr(c, "camera_yaw_offset_deg"))
check("it is the measured value", abs(c.camera_yaw_offset_deg + 1.95) < 1e-9,
      f"{c.camera_yaw_offset_deg} deg")
check("NEGATIVE - positive halved the fix count on both flights",
      c.camera_yaw_offset_deg < 0)
check("and it is a mount angle, not a large correction",
      abs(c.camera_yaw_offset_deg) < 10.0,
      "anything bigger is a mounting error to fix with a spanner")

print("\nit is added to the heading the map query is rotated by")
src = open(os.path.join(os.path.dirname(__file__), "..",
                        "vps_vio", "live_node.py")).read()
check("the submit call carries it",
      'yaw_deg=f["yaw"] + self.cam_yaw_offset_deg' in src)
check("it is read from the config, not written in the node",
      "self.cam_yaw_offset_deg = float(_c.camera_yaw_offset_deg)" in src)

print("\nthe flow's own yaw rotation is left alone, on purpose")
check("shift_to_ne still gets the raw heading",
      'yaw_deg=f["yaw"],' in src,
      "the same offset should physically apply there too, but measuring it "
      "from flow direction against the receiver gave +3.05/+6.21/-1.19/+5.43 "
      "deg with an IQR of 15-49 - the phantom swamps it")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
