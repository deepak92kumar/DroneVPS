"""Nothing from before a GPS->VPS switch crosses it.

THE PILOT'S INSTRUCTION, 23 October: "gps se vps par switch ke waqt purana
koi bhi data le kar ... nahi ... jaha gps wahi se vps suru kiya" - where the
GPS is, that is where the VPS starts.

Half of this was already true and measured. live_node re-takes the origin from
the flight controller's own position and calls MapFuser.reset_displacement(),
which zeroes the accumulated displacement and puts the variance back to
init_sigma_m. Over the ten switches of flight 00000233 the displacement went
from as much as 74.5 m down to 0.4-2.0 m every single time, so POSITION was
never what crossed the handover.

The VELOCITY bridge was. It had no reset method at all, so the integrated
velocity, the learned accelerometer bias, the last flow and accel samples, the
slew reference and every timer ran continuously from node start, through
however long the aircraft sat on GPS, and straight into the first GPS_INPUT
packet after the switch. With EK3_SRC1_VELXY = 3 the flight controller takes
that velocity directly.
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "vps_vio"))

from map_fuse import MapFuser          # noqa: E402
from velocity_bridge import VelocityBridge  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


print("the velocity bridge keeps nothing across a reset")

b = VelocityBridge()
# Fly it for a while so every piece of state is dirty, the way eighty seconds
# of sitting on GPS leaves it.
t = 100.0
for i in range(200):
    b.push_accel(t, 0.3, -0.2)
    if i % 6 == 0:
        b.push_flow(t, 1.4, -0.9)
    b.value(t, 40.0)
    t += 0.03

dirty = {k: getattr(b, k) for k in
         ("vn", "ve", "bias_n", "bias_e", "_vn_flow", "_ve_flow",
          "_an", "_ae", "_sent_n", "_sent_e", "_t_sent", "_t_flow",
          "_t_accel", "_slew_since", "_slow_since", "_jump_run",
          "_started", "_last_alt_m")}
moved = [k for k, v in dirty.items()
         if v not in (None, 0.0, 0, False, -1e9)]
check("the run left real state behind to clear", len(moved) >= 4,
      ", ".join(sorted(moved)))

counters_before = {k: getattr(b, k) for k in
                   ("flow_updates", "accel_steps", "rejected_flow",
                    "rejected_jump", "slewed", "capped")}

b.reset_state("handover to VPS")

ZERO = {"vn": 0.0, "ve": 0.0, "bias_n": 0.0, "bias_e": 0.0,
        "_vn_flow": None, "_ve_flow": None, "_an": 0.0, "_ae": 0.0,
        "_sent_n": 0.0, "_sent_e": 0.0, "_t_sent": None, "_t_flow": None,
        "_t_accel": None, "_slew_since": -1e9, "_slow_since": None,
        "_jump_run": 0, "_started": False, "_last_alt_m": 0.0}
bad = [f"{k}={getattr(b, k)!r}" for k, want in ZERO.items()
       if getattr(b, k) != want]
check("every integrated and learned value is cleared", not bad,
      "; ".join(bad) if bad else f"{len(ZERO)} fields")

check("the learned accelerometer bias specifically",
      b.bias_n == 0.0 and b.bias_e == 0.0,
      "it is learned against the flow while the aircraft is slow, so eighty "
      "seconds of GPS loiter is eighty seconds of learning against an "
      "estimate nothing was correcting")

check("but the cumulative counters survive",
      all(getattr(b, k) == v for k, v in counters_before.items()),
      "a flight is audited with these afterwards; zeroing them mid-flight "
      "would hide the first half of it")
check("and the reset is itself counted and reasoned",
      b.n_resets == 1 and b.reset_why == "handover to VPS")

# A second reset must be idempotent - two switches in half a second happened
# ten times on flight 00000233.
b.reset_state("again")
check("a second reset is harmless", b.vn == 0.0 and b.n_resets == 2)

print("\nthe configuration is NOT thrown away with it")
for k, want in (("tau_s", 0.05), ("max_sent_mps", 6.0),
                ("deadband_base_mps", 0.35), ("deadband_per_m", 0.010),
                ("bias_tau_s", 20.0)):
    check(f"{k} is still {want}", abs(getattr(b, k) - want) < 1e-9,
          f"{getattr(b, k)}")

print("\nthe position half, which already worked")
f = MapFuser()
for _ in range(400):
    f.predict(1.0, 0.5)
check("dead reckoning really did accumulate",
      float(abs(f.pos[0])) > 100.0, f"{f.pos[0]:.0f} m north")
grown = f.sigma_m
f.reset_displacement()
check("reset_displacement zeroes the displacement",
      float(abs(f.pos[0])) < 1e-9 and float(abs(f.pos[1])) < 1e-9)
check("and puts the variance back to init, not just the position",
      f.sigma_m < grown, f"{grown:.1f} m -> {f.sigma_m:.1f} m")

print("\nboth are called at the handover, and at the arm re-anchor")
src = open(os.path.join(HERE, "vps_vio", "live_node.py")).read()
hand = src.index('node.vel.reset_state("handover to VPS")')
disp = src.index("node.fuser.reset_displacement()", src.index("carried = float"))
check("the handover resets the bridge", hand > 0)
check("beside the displacement, not instead of it", abs(hand - disp) < 2000)
check("the arm re-anchor does the same",
      'node.vel.reset_state("origin re-taken at arm")' in src,
      "the origin moves there too, so the same reasoning applies")
check("the origin is re-taken from the FC, not from SCR_USER",
      'node.set_origin(*f["gps"], sigma_m=f.get("gps1_hacc"))' in src,
      "where the GPS is, that is where the VPS starts")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
