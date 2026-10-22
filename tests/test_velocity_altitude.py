"""The flow's error is constant in PIXELS, so in metres it grows with height.

Why this exists. On 20 Oct the aircraft hovered at 70 m and the VPS reported a
horizontal speed of up to 24.6 m/s. `EK3_SRC1_VELXY = 3` means the EKF fuses
what we send, so had the switch been on the VPS that number would have flown
the aircraft. Nothing reached the flight controller only because the source
gate held.

The cause is not what it looks like. Across the 668 samples where the reported
speed exceeded 5 m/s:

    flow age           0.014 s mean, and only 2.8% older than 0.1 s
    accel bias         -0.02 / -0.04 m/s2, frozen 100% of the time

So the accelerometer was not running away and the flow was not stale. What was
happening is arithmetic: velocity is pixels x altitude / focal length, so a
tracking artefact of a fixed size in pixels becomes a larger and larger speed
the higher the aircraft is. Measured, same three flights:

    alt band    sigma of reported speed    per metre of altitude
     0-10 m             0.126 m/s                 0.078
    10-20 m             0.811                     0.049
    20-30 m             1.826                     0.077
    30-45 m             4.569                     0.118
    45-60 m             4.131                     0.078
    60-80 m             5.713                     0.082

The per-metre column barely moves - which is the whole point. A straight line
through it is sigma = 0.0867 x altitude.

Two defences, and this file pins both:

  1. A PHYSICAL GATE. The airframe cannot change velocity faster than it can
     accelerate. ATC_ANGLE_MAX is 30 deg, so 9.81*tan(30) = 5.66 m/s2, which
     at 30 Hz is 0.189 m/s of change per frame. Measured frame-to-frame change
     in the reported speed: p99 1.696, max 22.74. Nine times the ceiling, and
     a hundred and twenty times.

  2. AN HONEST ACCURACY. The speed accuracy sent to the flight controller used
     to be a flat 0.4 m/s at every height. Telling the EKF that a 24.6 m/s
     reading is accurate to 0.4 m/s is how it gets believed.

RE-MEASURED 20 Oct, after the slew went back in. The table above is the RAW
flow, which is what the bridge is fed - not what it sends. sigma = 0.0867 x
altitude was fitted on it, and it over-states the real output by about three
times, because the 5 m/s^2 output slew removes most of that error before it
leaves. Sigma of what actually LEAVES the bridge, same flights, hovers only:

    altitude      5      15      25      38      52      70 m
    raw        0.07    0.87    2.16    4.57    4.30    5.46 m/s
    slewed     0.56    0.69    0.63    1.17    1.20    2.39 m/s

The slewed row is nearly flat - the slew bounds the rate of change, not the
magnitude, so it is not proportional to height the way the raw error is. Four
models against those six numbers:

                            rms error   worst under-statement
    0.0867 x alt  (was)         2.305        +0.128 m/s
    0.0305 x alt                0.233        +0.259
    flat 1.20                   0.636        +1.194
    0.6 + 0.025 x alt           0.432        +0.044

The last is chosen on the UNDER-STATEMENT column, not the rms: an accuracy
declared to an EKF is a promise, and the failure that matters is claiming to be
better than you are. 0.0305 x alt fits more tightly overall but under-states
somewhere by a quarter of a metre per second. At 70 m the chosen model declares
2.35 m/s against a measured 2.39.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "vps_vio"))

from velocity_bridge import VelocityBridge  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def feed(b, t0, n, vn, ve, dt=1.0 / 30.0):
    """n flow samples at a steady velocity."""
    t = t0
    for _ in range(n):
        t += dt
        b.push_flow(t, vn, ve)
    return t


# ------------------------------------------------- 1. the physical gate

print("a step the airframe could not have made is refused")
b = VelocityBridge()
t = feed(b, 0.0, 60, 0.0, 0.0)
before = b.rejected_jump
# 22.74 m/s in one frame was the worst really seen. 5.66 m/s2 at 30 Hz allows
# 0.189, and the gate's own ceiling of 8.0 allows 0.267.
ok = b.push_flow(t + 1.0 / 30.0, 22.74, 0.0)
check("a 22.74 m/s jump in one frame is refused", ok is False)
check("and it is counted, not swallowed", b.rejected_jump == before + 1,
      f"rejected_jump {b.rejected_jump}")
check("the estimate did not move", abs(b.vn) < 0.5, f"vn {b.vn:.3f}")

print("\nwhat the aircraft CAN do still passes")
b2 = VelocityBridge()
t = feed(b2, 0.0, 60, 0.0, 0.0)
step = 8.0 * (1.0 / 30.0) * 0.9        # just inside the gate
ok = b2.push_flow(t + 1.0 / 30.0, step, 0.0)
check("a step inside the acceleration limit is accepted", ok is True,
      f"{step:.3f} m/s in one frame")
check("nothing was counted as a jump", b2.rejected_jump == 0)

print("\nreal acceleration is not blocked - it arrives with the filter's own lag")
b3 = VelocityBridge()
t = feed(b3, 0.0, 60, 0.0, 0.0)
# Accelerate at 5 m/s2 for a second - well inside what the airframe can do.
A = 5.0
v = 0.0
for _ in range(30):
    v += A * (1.0 / 30.0)
    t += 1.0 / 30.0
    b3.push_flow(t, v, 0.0)
# The first version of this expected the estimate to land on v, and it does
# not - by design. A first-order filter chasing a ramp settles one time
# constant behind it, so the gap is A * tau_s = 5 * 0.35 = 1.75 m/s, and that
# is the complementary filter doing its job, not the gate refusing anything.
# (In flight the accelerometer fills most of that lag; this test pushes flow
# alone, so the full lag shows.)
expected_lag = A * b3.tau_s
check("nothing was refused on the way", b3.rejected_jump == 0,
      "this is what the gate must not do")
check("the estimate follows the ramp one time constant behind",
      abs((v - b3.vn) - expected_lag) < 0.5,
      f"gap {v - b3.vn:.2f} m/s against the expected {expected_lag:.2f}")
check("and it is genuinely tracking, not stuck", b3.vn > 0.6 * v,
      f"{b3.vn:.2f} of {v:.2f}")

# ------------------------------------------------- 2. the escape hatch

print("\na wrong estimate must not reject good flow for ever")
b4 = VelocityBridge(jump_escape_n=5)
t = feed(b4, 0.0, 60, 0.0, 0.0)
accepted = None
for i in range(12):
    t += 1.0 / 30.0
    if b4.push_flow(t, 15.0, 0.0):
        accepted = i
        break
check("after the escape count it believes the measurement", accepted is not None,
      f"accepted on refusal {accepted}")
check("exactly after jump_escape_n refusals", accepted == 5, str(accepted))

# ------------------------------------------------- 3. staleness still wins

print("\na run of refusals ends in 'do not send', not in a stale number")
b5 = VelocityBridge(jump_escape_n=1000)      # never escape, for this test
t = feed(b5, 0.0, 60, 0.0, 0.0)
for _ in range(40):                           # 40 frames = 1.33 s of refusals
    t += 1.0 / 30.0
    b5.push_flow(t, 20.0, 0.0)
st = b5.value(t, 50.0)
check("the estimate is declared invalid", st.valid is False, st.reason)
check("because the flow clock kept running", st.age_s > b5.flow_timeout_s,
      f"age {st.age_s:.2f}s > timeout {b5.flow_timeout_s}")

# ------------------------------------------------- 4. altitude is carried

print("\nthe estimate carries the height it was made at")
b6 = VelocityBridge()
t = feed(b6, 0.0, 60, 0.5, 0.0)
for alt in (0.0, 20.0, 70.0):
    st = b6.value(t, alt)
    check(f"altitude_m reports {alt:g}", abs(st.altitude_m - alt) < 1e-9)
check("and the jump count rides along too", b6.value(t, 10.0).rejected_jump == 0)

# ------------------------------------------------- 5. the accuracy formula

print("\nthe accuracy sent to the FC grows with height")
src = open(os.path.join(HERE, "vps_vio", "live_node.py")).read()
check("live_node no longer sends a flat 0.4 m/s",
      "max(0.4, 0.4 + 2.0 * vel.age_s)" not in src,
      "that line told the EKF a 24.6 m/s reading was good to 0.4 m/s")
check("it scales on the carried altitude",
      "self.VEL_ACC_PER_M * max(0.0, vel.altitude_m)" in src)
check("with a coefficient that covers the error at height",
      "VEL_ACC_PER_M = 0.048" in src,
      "0.0867 was fitted on the raw flow; 0.025 on hovers that sat low")
check("and a floor, which is pessimistic below 25 m - the safe direction",
      "VEL_ACC_FLOOR = 0.6" in src)

# THE PROMISE, and it is one-sided on purpose: declaring MORE error than
# there is costs weight in the EKF, declaring LESS is how a bad number gets
# flown. An accuracy handed to an EKF is a promise, and XKF4.SV is exactly
# error / declared.
#
# Two sets of measurements have to be covered, and 0.025 covered only the
# first. Thirteen recorded VPS flights, sigma of the bridge's OUTPUT at
# hover (first six rows), and log 226 of 22 Oct, which flew to 82.7 m with
# the bridge's inputs recorded, PER-AXIS against the raw receiver (last
# four). 0.025 under-stated the 226 rows by 0.6-1.5 m/s, SV sat at p50 1.26
# against FS_EKF_THRESH 1.2, and the aircraft went to ALT_HOLD twice.
FLOOR, PER_M = 0.6, 0.048
MEASURED = ((5.0, 0.561, "13 flights, hover"), (15.0, 0.686, "13 flights, hover"),
            (25.0, 0.627, "13 flights, hover"), (38.0, 1.166, "13 flights, hover"),
            (52.0, 1.198, "13 flights, hover"), (70.0, 2.394, "13 flights, hover"),
            (15.0, 0.41, "log 226, 5-25 m"), (35.0, 2.09, "log 226, 25-45 m"),
            (57.0, 2.05, "log 226, 45-70 m"), (77.0, 4.19, "log 226, 70-85 m"))
for alt, measured, whose in MEASURED:
    said = FLOOR + PER_M * alt
    check(f"at {alt:g} m it tells the FC {said:.2f} m/s, over {measured:.2f}",
          said >= measured - 0.05, whose)

# And the thing that was actually broken: the OLD coefficient does not keep
# the promise on the new rows. This is kept as a check so the number cannot
# quietly go back.
broke = [f"{alt:g} m" for alt, measured, whose in MEASURED
         if whose.startswith("log 226") and (0.6 + 0.025 * alt) < measured - 0.05]
check("the old 0.025 would still break the promise at height",
      broke == ["35 m", "77 m"], f"it under-states at {', '.join(broke)}")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
