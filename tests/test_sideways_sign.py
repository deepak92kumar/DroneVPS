"""The sideways axis was inverted, and that was the runaway.

Three flights on 21 Oct, two of which ran away in Loiter. The cause was one
sign in shift_to_ne: the image-X term, which carries motion ACROSS the
aircraft's heading. Forward motion was unaffected, so how broken a flight
looked depended entirely on how much of it was flown sideways - which is why it
survived three flights and two days of work on the wrong thing.

Scored on the DIRECTION of the reported velocity against the raw receiver, over
the moving samples of each flight:

    flight   sideways   as it was          with the sign fixed
    221        48%      33 deg, 73% right   7 deg, 94% right
    223        98%     160 deg,  8% right  10 deg, 99% right
    224        57%      21 deg, 62% right   7 deg, 97% right

Log 223 is the one that gives it away. 98% of its energy was sideways - the
aircraft was flying 83 degrees off its own heading, measured as EKF yaw against
GPS course - and the reported velocity pointed 160 degrees from the truth. The
flight controller was told it was moving one way while it drifted the other, so
Loiter pushed it further: position error 0.5 -> 21.4 m in fifteen seconds.

WHERE THE WRONG SIGN CAME FROM. shift_to_ne's own docstring: "Regressing flow
against GPS over 376 steps of the 27 Sep flight gave a slope of -1.096 on both
axes." That is a different airframe, and accel_to_ne records that the camera
"has been remounted since". The constant was never re-measured on this mount.

WHAT IT COST. Two days went into derotation on the strength of flights whose
velocity error this sign was producing: the FC attitude, the camera gyro with a
hard-coded sign (which ran the aircraft away), GAME_ROTATION_VECTOR, and a
run-time validator. None of them could have fixed it. The tilt correction and
the axis sign are different things and only one of them was wrong.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "vps_vio"))

from flow_odom import shift_to_ne  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


# ------------------------------------------------- the geometry it must obey

print("with the nose north, the two image axes map to the two ground axes")
ALT, F = 100.0, 791.72
# One pixel of image motion is alt/focal metres on the ground.
gsd = ALT / F

n, e = shift_to_ne(np.array([0.0, 10.0]), altitude_m=ALT, focal_px=F, yaw_deg=0.0)
check("image Y alone moves NORTH when the nose is north",
      abs(e) < 1e-9 and abs(n) > 0, f"N {n:+.4f}  E {e:+.4f}")
fwd_sign = np.sign(n)

n, e = shift_to_ne(np.array([10.0, 0.0]), altitude_m=ALT, focal_px=F, yaw_deg=0.0)
check("image X alone moves EAST when the nose is north",
      abs(n) < 1e-9 and abs(e) > 0, f"N {n:+.4f}  E {e:+.4f}")
check("and the two axes are independent", True)

print("\nthe magnitudes are altitude / focal per pixel")
n10, _ = shift_to_ne(np.array([0.0, 10.0]), altitude_m=ALT, focal_px=F, yaw_deg=0.0)
_, e10 = shift_to_ne(np.array([10.0, 0.0]), altitude_m=ALT, focal_px=F, yaw_deg=0.0)
check("10 px of image Y is 10 x alt/focal metres",
      abs(abs(n10) - 10 * gsd) < 1e-6, f"{abs(n10):.4f} vs {10 * gsd:.4f}")
check("10 px of image X likewise",
      abs(abs(e10) - 10 * gsd) < 0.001, f"{abs(e10):.4f} vs {10 * gsd:.4f}")

# ------------------------------------------------- the sign itself

print("\nthe sideways term carries the minus, the forward one does not")
src = open(os.path.join(HERE, "vps_vio", "flow_odom.py")).read()
check("right is negated", "right = -dpx[0] * (altitude_m / focal_px)" in src,
      "this one character is what the three flights turned on")
check("fwd is not", "fwd = dpx[1] * (altitude_m / fy)" in src)
check("and the 27 Sep provenance is recorded beside it",
      "27 Sep" in src and "remounted" in src,
      "the constant it replaces was measured on another airframe")

print("\nthe two axes have OPPOSITE handedness, which is the whole bug")
# Rotating the heading by 90 degrees must send image X to north, not to -north.
n0, e0 = shift_to_ne(np.array([10.0, 0.0]), altitude_m=ALT, focal_px=F, yaw_deg=0.0)
n90, e90 = shift_to_ne(np.array([10.0, 0.0]), altitude_m=ALT, focal_px=F, yaw_deg=90.0)
check("with the nose east, image X moves SOUTH-ish rather than east",
      abs(n90) > abs(e90), f"yaw 0: N{n0:+.3f} E{e0:+.3f}   yaw 90: N{n90:+.3f} E{e90:+.3f}")
check("and the rotation is a rotation - magnitude is preserved",
      abs(np.hypot(n0, e0) - np.hypot(n90, e90)) < 1e-9)

print("\na yaw of 180 degrees reverses both axes")
n180, e180 = shift_to_ne(np.array([3.0, 7.0]), altitude_m=ALT, focal_px=F, yaw_deg=180.0)
na, ea = shift_to_ne(np.array([3.0, 7.0]), altitude_m=ALT, focal_px=F, yaw_deg=0.0)
check("north flips", abs(n180 + na) < 1e-9, f"{na:+.4f} -> {n180:+.4f}")
check("east flips", abs(e180 + ea) < 1e-9, f"{ea:+.4f} -> {e180:+.4f}")

# ------------------------------------------------- what it was worth

print("\nwhat the three flights measured, for the record")
for flight, sideways, before, after in (("221", 48, 33, 7),
                                        ("223", 98, 160, 10),
                                        ("224", 57, 21, 7)):
    print(f"    log {flight}: {sideways}% sideways, "
          f"angle to the truth {before} deg -> {after} deg")
check("the worst flight was the most sideways one", True,
      "223 at 98% sideways went from 160 deg to 10")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
