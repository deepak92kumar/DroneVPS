"""Pin the 8 October velocity collapse so it cannot come back.

The fault in one line: flow put in 9.93 m/s against a 9.72 m/s GPS1 truth and
the bridge put out 3.90 - 40% of truth, 6.03 m/s lost inside one module - while
the send path lost only 0.18. The aircraft then flew 6.78 m/s on a 4.0 m/s
mission with the EKF believing 3.02.

The decisive test is test_sustained_speed_with_sparse_flow: the old module gave
about 9 m/s when its own logged inputs were replayed offline but 1.10 live, so
whatever collapsed depended on how often flow actually arrived. Here flow is
deliberately sparse and the output still has to equal the truth.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "vps_vio"))
from velocity_bridge import VelocityBridge  # noqa: E402

TRUTH = 9.93          # m/s, the flow reading at 6+ m/s on 8 Oct
OLD_OUTPUT = 3.90     # what the lost module produced from it
ALT = 110.0

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def run(*, flow_every: int = 1, steps: int = 600, hz: float = 25.0,
        truth: float = TRUTH, accel_bias: float = 0.0, bridge=None):
    """Cruise north at `truth` m/s. True acceleration is zero at constant
    velocity, so the accelerometer sees only its own bias."""
    b = bridge or VelocityBridge()
    dt = 1.0 / hz
    t = 0.0
    b.push_flow(t, truth, 0.0)
    for i in range(1, steps + 1):
        t = i * dt
        b.push_accel(t, accel_bias, 0.0)
        if i % flow_every == 0:
            b.push_flow(t, truth, 0.0)
    return b, b.value(t, ALT)


print("the 8 October case")
b, st = run(flow_every=1)
check("flow every step reproduces truth", abs(st.speed_mps - TRUTH) < 0.3,
      f"{st.speed_mps:.2f} m/s vs truth {TRUTH}")
check("not the old 40% output", st.speed_mps > 0.9 * TRUTH,
      f"{100*st.speed_mps/TRUTH:.0f}% of truth, old module gave "
      f"{100*OLD_OUTPUT/TRUTH:.0f}%")

# THE decisive one. Flow only every 4th step, as in flight, where the live run
# collapsed to 1.10 while an offline replay of the same inputs gave about 9.
print("\nsparse flow, which is what separated live from replay")
for every in (2, 4, 8, 12):
    b, st = run(flow_every=every)
    check(f"flow every {every:>2} steps still reads truth",
          abs(st.speed_mps - TRUTH) < 0.5,
          f"{st.speed_mps:.2f} m/s = {100*st.speed_mps/TRUTH:.0f}% of truth")

print("\nan accelerometer bias may offset, never attenuate")
for bias in (0.0, 0.2, -0.5, 1.0):
    b, st = run(flow_every=4, accel_bias=bias)
    check(f"bias {bias:+.1f} m/s^2 does not scale the output",
          st.speed_mps > 0.85 * TRUTH,
          f"{st.speed_mps:.2f} m/s = {100*st.speed_mps/TRUTH:.0f}% of truth")

print("\nsustained motion must not be learned as bias")
b, st = run(flow_every=2, steps=1500)          # 60 s at 25 Hz
check("bias stayed put over 60 s of cruise",
      abs(b.bias_n) < 1e-6 and abs(b.bias_e) < 1e-6,
      f"bias_n {b.bias_n:.2e}")
check("bias reported as frozen", st.bias_frozen, "")
check("still reads truth after 60 s", abs(st.speed_mps - TRUTH) < 0.5,
      f"{st.speed_mps:.2f} m/s")

print("\nflow dropout holds the value and stops sending; it does NOT decay")
b = VelocityBridge()
dt = 1.0 / 25.0
b.push_flow(0.0, TRUTH, 0.0)
t = 0.0
for i in range(1, 51):                          # 2 s of accel only
    t = i * dt
    b.push_accel(t, 0.0, 0.0)
st = b.value(t, ALT)
check("value held, not decayed toward zero", abs(st.speed_mps - TRUTH) < 0.2,
      f"{st.speed_mps:.2f} m/s after {t:.1f}s without flow")
check("refused to send once stale", st.valid is False, st.reason)
check("the estimate is still readable for the log", st.vn > 0.9 * TRUTH,
      f"vn {st.vn:.2f}")

print("\na tracking collapse must not define the estimate")
b, _ = run(flow_every=4)
ok = b.push_flow(100.0, 630.0, 0.0)             # what the old flow returned on failure
check("630 m/s refused", ok is False, f"rejected_flow {b.rejected_flow}")
st = b.value(100.0, ALT)
check("estimate unchanged by it", st.vn < 2 * TRUTH, f"vn {st.vn:.2f}")

print("\nbias does adapt when genuinely stationary")
b = VelocityBridge()
t = 0.0
b.push_flow(t, 0.0, 0.0)
for i in range(1, 400):                         # 16 s at rest, biased accel
    t = i * dt
    b.push_accel(t, 0.3, 0.0)
    b.push_flow(t, 0.0, 0.0)
check("bias moved while at rest", abs(b.bias_n) > 0.01, f"bias_n {b.bias_n:.3f}")
check("bias reported unfrozen", b.value(t, ALT).bias_frozen is False, "")

print("\nthe deadband zeroes what is sent but reports that it did")
b = VelocityBridge()
b.push_flow(0.0, 0.20, 0.0)                     # under 0.35 + 0.005*110 = 0.90
st = b.value(0.0, ALT)
check("sent value is zero", st.sent_vn == 0.0 and st.sent_ve == 0.0, "")
check("deadbanded flag set", st.deadbanded is True, "")
check("real speed still reported", abs(st.speed_mps - 0.20) < 1e-6,
      f"speed_mps {st.speed_mps:.3f}, deadband {st.deadband_mps:.3f}")
b2 = VelocityBridge()
b2.push_flow(0.0, 5.0, 0.0)
st2 = b2.value(0.0, ALT)
check("above the deadband it passes through", abs(st2.sent_vn - 5.0) < 1e-6,
      f"sent_vn {st2.sent_vn:.2f}")

print("\nbad timestamps must not inject a step")
b = VelocityBridge()
b.push_flow(0.0, TRUTH, 0.0)
b.push_accel(0.04, 0.0, 0.0)
before = b.vn
b.push_accel(0.04, 50.0, 0.0)                   # dt = 0
b.push_accel(-1.0, 50.0, 0.0)                   # dt negative
b.push_accel(99.0, 50.0, 0.0)                   # dt absurd
check("zero, negative and absurd dt ignored", abs(b.vn - before) < 1e-9,
      f"vn {b.vn:.4f}")

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
