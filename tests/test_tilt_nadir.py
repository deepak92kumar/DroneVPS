"""A tilted camera does not look straight down.

The matcher answers "the middle of this frame is here". The aircraft is
nadir of that point, not at it: a tilt of theta puts the image centre
alt*tan(theta) away, in the direction of the tilt. Nothing corrected it.

The pilot named the mechanism before it was measured - "moving ke time drone
tilt hota hai tho ncc bhagayega galat correction bhi de sakta hai". On the
fixes of two flights, corr(tilt, fix error) is +0.484 and
corr(alt*tan(tilt), fix error) +0.506, with error 8.85 m below 2 deg,
10.68 m at 2-5 and 14.33 m above 5.

THE SIGNS ARE MEASURED. All four pairs were tried against the raw receiver:

    flight 240 (70 fixes)   11.18 m -> 8.69 m,  p90 21.73 -> 13.45
    flight 242 (17 fixes)    9.86 m -> 6.83 m,  p90 16.61 ->  8.75

and the other three made it worse on both. An unmeasured sign is what caused
the 21 Oct runaway, which is why this is a test and not a derivation.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vps_vio"))
from map_fix_worker import MapFixWorker as W  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        fails.append(name)


print("level flight moves nothing")
n, e = W.nadir_offset_m(110.0, 0.0, 0.0, 0.0)
check("no tilt, no shift", abs(n) < 1e-9 and abs(e) < 1e-9)
n, e = W.nadir_offset_m(0.0, 10.0, 10.0, 0.0)
check("no altitude, no shift", n == 0.0 and e == 0.0,
      "on the ground there is nothing to project onto")

print("\nthe size is alt * tan(tilt)")
for alt, t in ((110.0, 5.0), (76.0, 10.0), (38.0, 2.0)):
    n, e = W.nadir_offset_m(alt, 0.0, t, 0.0)
    check(f"{t:.0f} deg at {alt:.0f} m", abs(math.hypot(n, e) - alt * math.tan(math.radians(t))) < 1e-6,
          f"{math.hypot(n, e):.2f} m")

print("\nthe measured signs: pitch forward, roll to the LEFT")
n, e = W.nadir_offset_m(110.0, 0.0, 5.0, 0.0)
check("pitch puts the centre NORTH when heading north", n > 0 and abs(e) < 1e-9,
      f"N {n:+.2f} E {e:+.2f}")
n, e = W.nadir_offset_m(110.0, 5.0, 0.0, 0.0)
check("roll puts it WEST when heading north", e < 0 and abs(n) < 1e-9,
      f"N {n:+.2f} E {e:+.2f} - this is the pair the flights chose, "
      f"the other three were worse on both")

print("\nit rotates with the heading")
n, e = W.nadir_offset_m(110.0, 0.0, 5.0, 90.0)
check("heading east, pitch now points east", e > 0 and abs(n) < 1e-6,
      f"N {n:+.2f} E {e:+.2f}")

print("\nthe node hands the attitude over and the worker uses it")
w = open(os.path.join(os.path.dirname(__file__), "..",
                      "vps_vio", "map_fix_worker.py")).read()
l = open(os.path.join(os.path.dirname(__file__), "..",
                      "vps_vio", "live_node.py")).read()
check("submit takes the attitude", "roll_deg: float = 0.0" in w)
check("the node passes the frame's own", 'roll_deg=f.get("roll")' in l)
check("the fix is moved by it", "res.lat = res.lat - dN / 111320.0" in w)
check("applied ONCE, outside both matchers",
      w.count("nadir_offset_m(alt, roll, pitch, yaw)") == 1,
      "both report the same image centre, so one correction serves both")
check("and it cancels out of their disagreement",
      "cancels out of" in w,
      "the same shift on both answers cannot change the gap between them")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
