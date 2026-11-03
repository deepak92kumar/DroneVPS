"""The camera's yaw on the airframe reaches the map query.

The mount was moved on 23 Oct. Measured 24 Oct two ways: a similarity fitted
to the translation inliers gives -2.03 deg on log 240 (spread 0.85) and
-1.87 deg on log 242 (spread 0.83), over different yaw ranges, with corr
against yaw of only -0.25 and +0.19 - a constant offset, not a heading-scale
error. Replaying locate() on real frames against the raw receiver decided the
SIGN, which the first attempt got wrong by testing +1.95 only:

    log 240:  -1.95 -> 23 fixes, 11.22 m | 0 -> 19, 11.38 | +1.95 -> 12, 12.58
    log 242:  -1.95 -> 16 fixes,  9.41 m | 0 -> 14,  9.70 | +1.95 -> 10,  8.19

1 NOV: BACK TO 0, at the pilot's instruction to put it right, and putting it
right meant taking it out. Both measurements above are a different site and a
different sheet. Flight 382 was swept for the offset directly - 81 frames,
window centred on GPS_RAW_INT so the drifting prior could not colour it, and
with locate()'s own +-6 deg yaw search disabled so the sweep measured the
offset instead of smearing it across it:

    offset  -16  -12   -8   -4    0   +4   +8  +12  +16
    err p50  32   32   36   32   35   33   32   34   30   m

Flat across 32 degrees. A real mounting offset digs a well at its own value
and climbs on both sides; there is no well anywhere here. The matcher's error
on this sheet does not depend on yaw at all, which is what every other test of
it has said - there is no match to align. 0 is the value that claims nothing
the data does not carry.

The mount has not moved since 23 Oct, so if a sheet ever matches, sweep it on
that sheet and put the measured number back.
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
print("the offset is configured, and it claims only what was measured")
check("the field exists", hasattr(c, "camera_yaw_offset_deg"))
# 2 NOV: back to -1.95, at the pilot's instruction, because that is what the
# 27 Oct flights - the ones whose matcher worked, 73 fixes with the two
# matchers 2.0 m apart - actually flew. It had been set to 0 on 1 Nov after a
# sweep found no well across 32 degrees.
#
# What the sweep on 27 Oct's own frames says, replayed through today's code:
# 0.00 gives 9 of 20 fixes within 15 m and -1.95 gives 8, so the position is
# a wash - but the two matchers' DISAGREEMENT is 7.0 m at -1.95 against
# 13.9 m at 0.00, and 2.0 m is what 27 Oct recorded. The offset is not proven
# to matter for position; it is being restored because the flight that worked
# had it and because flow derotation reads the same number.
check("it is the measured mount offset for the rebuilt mount",
      abs(c.camera_yaw_offset_deg + 4.42) < 1e-9,
      f"{c.camera_yaw_offset_deg} deg")
check("and it is a mount angle, not a large correction",
      abs(c.camera_yaw_offset_deg) < 10.0,
      "anything bigger is a mounting error to fix with a spanner")

print("\nit is added to the heading the map query is rotated by")
src = open(os.path.join(os.path.dirname(__file__), "..",
                        "vps_vio", "live_node.py")).read()
check("the submit call carries it, through the canonical helper",
      "yaw_deg=self.corrected_yaw(f)," in src)
check("it is read from the config, not written in the node",
      "self.cam_yaw_offset_deg = float(_c.camera_yaw_offset_deg)" in src)

print("\nthe flow path gets it too - 2 Nov, and why this check reversed")
# This file used to assert the OPPOSITE: that shift_to_ne keeps the raw
# heading, because measuring the offset from flow direction against the
# receiver gave +3.05/+6.21/-1.19/+5.43 deg with an IQR of 15-49 and the
# phantom swamped it. That reasoning was sound and the conclusion was wrong -
# the noise was in the METHOD, not in the offset.
#
# Sliding the aircraft along a known line on the ground removes every source
# of that noise: no GPS smoothing, no altitude, no phantom. North leg +3.04,
# south leg +5.29, weighted +4.42 - and the two whole-flight weighted means,
# unfiltered, had said +4.32 and +4.41 all along.
# 2 NOV: the offset has to reach BOTH paths. Until today shift_to_ne took the
# raw FC yaw and only the matcher's submit added camera_yaw_offset_deg, so
# the mount's rotation went straight into every metre of flow - and nothing
# caught it because no test compared the two call sites.
print("\nthe flow path gets it too, not only the matcher")
# getattr, like n_frame_stale: the gap-fill test builds a VpsNode without
# running __init__ so that nothing opens a camera, and a bare attribute read
# broke it - which is how this reached a push with one test red.
check("shift_to_ne is given it, through the same helper",
      "yaw_deg=self.corrected_yaw(f)," in src
      and "focal_y_px=self.focal_y)" in src)
# 2 NOV, later: all three camera-frame consumers were put behind one
# corrected_yaw() helper, so these check for the helper rather than for
# three separate spellings of the same addition. Having two spellings IS
# what let accel_to_ne keep the raw yaw while shift_to_ne was corrected.
check("the accelerometer path gets it too",
      "accel_to_ne(ax, ay, self.corrected_yaw(f), az," in src)
check("all three use the one helper",
      src.count("self.corrected_yaw(f)") == 3,
      str(src.count("self.corrected_yaw(f)")))
check("no call site is left on the raw yaw",
      'yaw_deg=f["yaw"],' not in src)
check("and the value is the measured one",
      abs(c.camera_yaw_offset_deg + 4.42) < 1e-9,
      f"{c.camera_yaw_offset_deg} deg")

print("\nALL PASS" if not fails else f"\n{len(fails)} FAILED: {', '.join(fails)}")
sys.exit(1 if fails else 0)
