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
check("it is the measured mount offset the working flights used",
      abs(c.camera_yaw_offset_deg + 1.95) < 1e-9,
      f"{c.camera_yaw_offset_deg} deg")
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
