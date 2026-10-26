"""The lidar corrects the barometer's offset. It never replaces it.

Velocity is pixels x altitude / focal, so an altitude error is a velocity
error of the same proportion. Measured on flight 245 against the lidar,
alt_m / lidar came out:

    0-5 m    2.86      30-50 m   1.17
    5-15 m   1.72      50-80 m   1.08
    15-30 m  1.29      80-120 m  1.07

A near-constant 5-7 m offset, the SAME climbing and descending (+6.59,
+6.82, +7.00 against +6.63, +6.02, +5.24), so an offset and not the descent
hysteresis remembered from an earlier airframe. At touchdown the lidar read
0.60 m and alt_m read 4.2. Six percent at 100 m is nothing; 72% at 10 m is
a phantom three quarters the size of real motion, and 35 m is where the
aircraft was circling.

THE PILOT'S CONDITION: a badly configured lidar must never stop the
aircraft flying. So nothing here is required. No lidar, no correction, and
the altitude is exactly what it always was.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vps_vio"))
from live_node import VpsNode  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        fails.append(name)


def node():
    n = VpsNode.__new__(VpsNode)
    n._alt_bias_m = 0.0
    n._alt_bias_t = None
    n.alt_bias_m = 0.0
    return n


print("no lidar, no correction")
n = node()
for f in ({"alt": 40.0, "t": 1.0}, {"alt": 40.0, "rngfnd": None, "t": 2.0}):
    check(f"{f} -> 0.0", n._alt_bias(f) == 0.0)

print("\nout of the lidar's range, no correction")
n = node()
check("0.8 m is the leg, not the ground", n._alt_bias({"alt": 7.0, "rngfnd": 0.8, "t": 1.0}) == 0.0)
check("90 m is past the TF03", n._alt_bias({"alt": 96.0, "rngfnd": 90.0, "t": 2.0}) == 0.0)

print("\nan absurd disagreement is not a bias")
n = node()
check("40 m apart is the lidar on a roof, not an offset",
      n._alt_bias({"alt": 60.0, "rngfnd": 20.0, "t": 1.0}) == 0.0,
      "clamped at ALT_BIAS_MAX_M")

print("\nit learns the offset, and it learns it SLOWLY")
n = node()
first = n._alt_bias({"alt": 36.0, "rngfnd": 30.0, "t": 0.0})
check("the first usable reading seeds it", abs(first - 6.0) < 1e-9, f"{first:.3f}")
after = n._alt_bias({"alt": 40.0, "rngfnd": 30.0, "t": 1.0})
check("one second moves it a little, not all the way",
      6.0 < after < 7.0, f"{after:.3f} toward 10.0 - tau is 20 s")
for i in range(2, 200):
    after = n._alt_bias({"alt": 40.0, "rngfnd": 30.0, "t": float(i)})
check("and it gets there eventually", abs(after - 10.0) < 0.1, f"{after:.3f}")

print("\nthe correction is subtracted, and never makes the aircraft underground")
src = open(os.path.join(os.path.dirname(__file__), "..",
                        "vps_vio", "live_node.py")).read()
check("alt is reduced by the bias",
      'f["alt"] = max(1.0, float(f["alt"]) - self.alt_bias_m)' in src)
check("and floored at 1 m", 'max(1.0,' in src,
      "an offset is not a licence to invent a negative height")
check("it is logged", '"alt_bias_m"' in src)
check("the lidar is still not a height source",
      'altitude_m=f["alt"]' in src,
      "everything downstream takes the CORRECTED barometer, not the lidar")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
