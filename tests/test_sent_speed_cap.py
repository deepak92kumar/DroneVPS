"""Nothing above max_sent_mps ever leaves the bridge, and the slew is not that bound.

The slew limits the rate of CHANGE. At slew_mps2 of 5 it will walk the output
to 25 m/s over five seconds and call every step legal, and that is what flew:

    sent to the FC      log 228      log 230
      p50                 1.03         1.39
      p99                15.52        15.10
      max                25.86        25.47
    bridge pre-slew max  26.25        25.48

So the slew removed essentially nothing. Against what the aircraft actually
did in those same VPS windows, from the receiver: p50 0.54-0.86, p95
1.39-2.22, p99 1.90-3.58, maximum 5.04 m/s.

LOIT_SPEED_MS is 3.0 and the VPS is only ever the source in LOITER, so the
position controller cannot command more; ATC_ANGLE_MAX of 15 degrees bounds
the acceleration near 2.6 m/s^2. Nothing above 6 m/s is a measurement - it is
the flow's pixel noise multiplied by height.

Those spikes are what tripped the EKF. XKF4.SV sat at p50 0.22-0.61 against a
1.2 gate and reached 13.77: the failsafe fires on spikes, not on a sustained
bad state, and ALT_HOLD followed four times on log 230. At 40-50 m, 48% of
sent samples were above 4 m/s.

Replayed against both flights' own sent-speed columns, a 6.0 cap takes p99
from 15.5 to 6.0 and the maximum from 25.9 to 6.0, while clipping 0.00% of
samples below 25 m on log 228 and 0.37% on log 230 - the altitudes where the
hold already works are untouched.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "vps_vio"))

from velocity_bridge import VelocityBridge

FAILED = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}   {detail}")
    if not ok:
        FAILED.append(label)


def feed(b, vn, ve, secs=20.0, hz=30.0, alt=31.0):
    """Hold one flow velocity long enough for the slew to be irrelevant."""
    t = 0.0
    nxt = 0.0
    st = None
    while t < secs:
        b.push_accel(t, 0.0, 0.0)
        if t >= nxt:
            b.push_flow(t, vn, ve)
            nxt += 1.0 / hz
        st = b.value(t, alt)
        t += 0.01
    return st


print(__doc__)

b = VelocityBridge()
CAP = b.max_sent_mps
print(f"max_sent_mps = {CAP}")
check("the cap is above LOIT_SPEED_MS of 3.0", CAP > 3.0, f"{CAP}")
check("and above the fastest the aircraft was measured doing on the VPS",
      CAP > 5.04, "5.04 m/s, log 230")
check("and well below the 25.5-25.9 m/s that actually went out", CAP < 10.0)

print("\nreal motion inside the cap passes through untouched")
for sp in (0.8, 2.0, 3.0, 5.0):
    st = feed(VelocityBridge(), sp, 0.0)
    check(f"a sustained {sp:.1f} m/s is sent as {st.sent_vn:.3f}",
          abs(st.sent_vn - sp) < 0.01 and not st.capped)

print("\nabove it, the length is cut and the DIRECTION is kept")
for vn, ve in ((25.0, 0.0), (0.0, -18.0), (12.0, 12.0)):
    st = feed(VelocityBridge(), vn, ve)
    mag = math.hypot(st.sent_vn, st.sent_ve)
    want = math.atan2(ve, vn)
    got = math.atan2(st.sent_ve, st.sent_vn)
    check(f"({vn:.0f},{ve:.0f}) -> |{mag:.2f}| at the cap",
          abs(mag - CAP) < 0.01, f"{mag:.3f}")
    check("   and pointing the same way",
          abs(math.atan2(math.sin(want - got), math.cos(want - got))) < 1e-6)

print("\nthe cap is counted, so a flight shows how often it bit")
bb = VelocityBridge()
st = feed(bb, 20.0, 0.0)
check("st.capped is set on a clipped step", st.capped is True)
check("and the run total is carried", st.cap_count > 0, f"{st.cap_count}")
st2 = feed(VelocityBridge(), 1.0, 0.0)
check("not set when nothing was clipped",
      st2.capped is False and st2.cap_count == 0)

print("\nthe slew alone does NOT bound the magnitude - this is why the cap exists")
# With the cap disabled, ramp the flow up and let the slew have its way.
nb = VelocityBridge(max_sent_mps=0.0)
t = 0.0
nxt = 0.0
target = 25.0
st = None
while t < 30.0:
    nb.push_accel(t, 0.0, 0.0)
    if t >= nxt:
        nb.push_flow(t, target, 0.0)
        nxt += 1.0 / 30.0
    st = nb.value(t, 31.0)
    t += 0.01
check("without a cap the slew lets 25 m/s out", st.sent_vn > 20.0,
      f"{st.sent_vn:.2f} m/s - the slew bounds d/dt, not |v|")
check("and a zero cap means no cap, deliberately", nb.max_sent_mps == 0.0)

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
    sys.exit(1)
print("ALL PASS")
