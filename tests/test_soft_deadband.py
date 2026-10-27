"""The deadband shrinks the speed; it does not zero everything below a line.

Flight 245 is why. The phantom finally shrank enough to sit ON the deadband
instead of well above it, and the hard version then chattered across it: the
sent speed crossed zero 307 times in 634 s - 0.48 a second - with 39.9% of
rows zeroed. Every crossing is a step from 0 to the full value and back, and
the flight controller answered with tilt commands saturating at
ATC_ANGLE_MAX and rates to 56.9 deg/s. That is the jerking the pilot
reported. Map fixes were ruled out: 16% of the >1 m/s jumps fall within
0.3 s of a fix against a 15% base rate.

Subtracting the threshold instead of zeroing below it is continuous - at the
threshold the output is 0 and it grows from there - so there is no step at
all, while a hover is suppressed exactly as before.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vps_vio"))
from velocity_bridge import VelocityBridge  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        fails.append(name)


def sent(speed, alt=82.0):
    """What the FC is told, for a speed due north at this height."""
    b = VelocityBridge()
    b.push_flow(0.0, speed, 0.0)
    st = b.value(0.0, alt)
    return st.sent_vn, st.deadband_mps


db = VelocityBridge().deadband_for(82.0)
print(f"the deadband at 82 m is {db:.3f} m/s")

print("\na hover is still suppressed")
for v in (0.0, 0.3, db * 0.5, db * 0.99):
    s, _ = sent(v)
    check(f"{v:.2f} m/s -> 0", abs(s) < 1e-9, f"{s:.4f}")

print("\nand there is NO STEP at the threshold")
below, _ = sent(db * 0.999)
above, _ = sent(db * 1.001)
check("just below and just above are the same to within a whisker",
      abs(above - below) < 0.01,
      f"{below:.4f} -> {above:.4f}; the hard version jumped to {db:.3f}")

print("\nbetween the threshold and 1.5x it, a straight ramp")
for v in (db * 1.1, db * 1.25, db * 1.4):
    s, _ = sent(v)
    want = v * (v - db) / (0.5 * db)
    check(f"{v:.2f} m/s -> {want:.2f}", abs(s - want) < 0.02, f"{s:.3f}")

print("\nand above 1.5x it, REAL MOTION GOES OUT IN FULL")
# The first attempt subtracted the threshold everywhere, so a genuine 3 m/s
# left as 2.34; the second ramped to 2x, which still swallowed 0.8 m/s at
# low level. Both are log 229's fault - an aircraft told it is slower than
# it is does not correct, and it ran 11.1 m in 24 s. The ramp is half a
# deadband wide and no wider.
for v in (db * 1.5, db * 1.8, 3.0, 5.0):
    s, _ = sent(v)
    check(f"{v:.2f} m/s -> {v:.2f}, untouched", abs(s - v) < 0.02, f"{s:.3f}")

print("\nit is monotonic - more real speed is never less reported speed")
prev = -1.0
mono = True
for i in range(400):
    v = i * 0.02
    s, _ = sent(v)
    if s < prev - 1e-9:
        mono = False
        break
    prev = s
check("monotonic across 0 to 8 m/s", mono)

print("\nthe deadband still scales with height, and still reports itself")
check("higher means a wider threshold",
      VelocityBridge().deadband_for(110.0) > VelocityBridge().deadband_for(38.0),
      f"{VelocityBridge().deadband_for(38.0):.2f} at 38 m, "
      f"{VelocityBridge().deadband_for(110.0):.2f} at 110")
b = VelocityBridge()
b.push_flow(0.0, 0.2, 0.0)
check("a suppressed hover still says so", b.value(0.0, 82.0).deadbanded is True)

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
