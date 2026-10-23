"""The blend must REDUCE the flow velocity's noise, and must not attenuate real motion.

This test exists because of a mistake. On 21 Oct 2026, reading log 225 /
vps_00059, the accel blend was accused of inflating the reported speed -
flow-only p50 0.53 m/s against the module's 1.04, with a receiver truth of
0.17 - and tau_s was lowered from 0.35 to 0.05 on that basis. The comparison
was invalid: the flow-only figure was smoothed over 30 samples and the
module's output was read raw. Smoothed identically at every width:

    width   flow-only   output   truth    corr N flow / output
      1       2.13       1.04     n/a        -0.093 / +0.266
      3       1.69       1.04     n/a        +0.072 / +0.263
     10       0.90       1.00     0.15       +0.235 / +0.253
     30       0.53       0.75     0.17       +0.412 / +0.234

The blend HALVES the raw flow velocity. Lowering tau_s would have made the
output track that 2.13 m/s of noise more closely and pushed XKF4.SV - which
put the aircraft in ALT_HOLD at 166.4 s - further the wrong way.

So the two properties below are the ones worth holding onto, and they pull
against each other: filter the flow's noise, without losing the magnitude of
real motion. The 8 Oct module satisfied the first by failing the second -
40% of truth at sustained speed, with nothing aboard able to tell.

Neither check asserts a value of tau_s. A sweep against a recording is still
owed, and needs vel_flow_* / vel_accel_* from a flight that has them.
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vps_vio.velocity_bridge import VelocityBridge

FAILED = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}   {detail}")
    if not ok:
        FAILED.append(label)


def fly(*, flow_noise, truth_vn=0.0, truth_ve=0.0, vib_mps2=0.0,
        flow_hz=30.0, accel_hz=100.0, dur_s=60.0, seed=11, bridge=None):
    """Return (rms of the OUTPUT's error, rms of the FLOW's error), m/s."""
    rnd = random.Random(seed)
    b = bridge if bridge is not None else VelocityBridge()
    t = 0.0
    next_flow = 0.0
    out_err = []
    flow_err = []
    while t < dur_s:
        if vib_mps2:
            b.push_accel(t, rnd.gauss(0.0, vib_mps2), rnd.gauss(0.0, vib_mps2))
        else:
            b.push_accel(t, 0.0, 0.0)
        if t >= next_flow:
            fn = truth_vn + rnd.gauss(0.0, flow_noise)
            fe = truth_ve + rnd.gauss(0.0, flow_noise)
            b.push_flow(t, fn, fe)
            if t > 5.0:
                flow_err.append(math.hypot(fn - truth_vn, fe - truth_ve))
            next_flow += 1.0 / flow_hz
        st = b.value(t, 31.0)
        if t > 5.0:
            out_err.append(math.hypot(st.vn - truth_vn, st.ve - truth_ve))
        t += 1.0 / accel_hz

    def rms(v):
        return math.sqrt(sum(x * x for x in v) / len(v))

    return rms(out_err), rms(flow_err)


print(__doc__)

print("1. the flow's noise at the height log 225 flew")
# The module's own measured figure, a few lines above max_accel_mps2:
# sigma = 0.0867 x altitude. At 31 m that is 2.69 m/s, and the raw flow
# velocity on log 225 did read p50 2.13 m/s while the aircraft moved at 0.17.
sigma_31 = 0.0867 * 31.0
out, flow = fly(flow_noise=sigma_31)
print(f"        sigma at 31 m = {sigma_31:.2f} m/s")
print(f"        flow rms {flow:.3f}  ->  output rms {out:.3f}   ({out / flow:.2f}x)")
check("the blend reduces the flow's noise, not amplifies it", out < flow,
      f"{out:.3f} < {flow:.3f}")
check("and reduces it by at least a third", out < 0.67 * flow,
      f"{out / flow:.2f}x")

print("\n2. the same, at the 80-95 m the map matcher needs")
sigma_90 = 0.0867 * 90.0
out90, flow90 = fly(flow_noise=sigma_90)
print(f"        sigma at 90 m = {sigma_90:.2f} m/s")
print(f"        flow rms {flow90:.3f}  ->  output rms {out90:.3f}   ({out90 / flow90:.2f}x)")
# 23 Oct: this demanded the same one-third reduction at 90 m as at 31 m, and
# tau_s 0.15 gives 0.72x. It is now a floor of "reduces at all", because the
# flight data says the model behind the stricter threshold has the trade
# backwards. Replayed on log 232's own flow and accel, scored against the
# receiver with identical smoothing on both sides:
#
#     tau_s      below 45 m          above 45 m
#              p50     p90         p50     p90
#     0.35    0.270   0.830       0.775   2.939
#     0.15    0.253   0.736       0.671   2.735
#     0.08    0.251   0.712       0.661   2.669
#
# Faster is better at BOTH heights and MORE so at height. Smoothing harder
# removes noise and adds lag; against real truth the lag costs more than the
# noise it buys. The direction is still pinned below - reduce, never amplify.
check("the blend still reduces at height", out90 < flow90,
      f"{out90 / flow90:.2f}x - one third is no longer demanded, see above")

print("\n3. the property the 8 Oct module broke: no attenuation")
for speed in (1.0, 3.0, 6.0):
    b = VelocityBridge()
    t = 0.0
    next_flow = 0.0
    while t < 25.0:
        b.push_accel(t, 0.0, 0.0)
        if t >= next_flow:
            b.push_flow(t, speed, 0.0)
            next_flow += 1.0 / 30.0
        t += 0.01
    st = b.value(t, 31.0)
    check(f"a sustained {speed:.1f} m/s flow reads back as itself",
          abs(st.vn - speed) < 0.05, f"{st.vn:.3f} m/s")

print("\n4. filtering noise must not be confused with hiding motion")
# Noise and signal together: the mean must survive even while the scatter
# around it is cut. This is the pair that the old module got wrong.
out, flow = fly(flow_noise=sigma_31, truth_vn=2.0)
print(f"        truth 2.0 m/s north with {sigma_31:.2f} m/s of flow noise")
print(f"        flow rms err {flow:.3f}  ->  output rms err {out:.3f}")
check("the error shrinks while the motion is real too", out < flow,
      f"{out:.3f} < {flow:.3f}")

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
    sys.exit(1)
print("ALL PASS")
