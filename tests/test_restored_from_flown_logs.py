"""Values the rebuild changed, restored from what the aircraft actually sent.

The NVMe took the code with it. What came back was rewritten from the status
reports, and on 20 Oct the pilot asked the obvious question: the flight
controller's own logs from before the loss are still here, so read what the
Jetson was really sending then and compare it with what it sends now.

Read off GPA and GPS instance 1 in logs 00000191 (12 Oct, the 1077 m
autonomous mission) and 00000194 (13 Oct, the last good VPS flight), both in
DRONE_fc_logs_backup/2026-10-19/APM/LOGS/:

                              12 Oct 191      13 Oct 194      the rebuild
    GPS_INPUT rate            4.28 Hz         4.31 Hz         30 Hz
    horizontal accuracy       p50 21.28 m     p50 21.36 m     the fuser's sigma
    vertical accuracy         3.00 constant   3.00 constant   sigma x 2 (~42 m)
    speed accuracy            p50 1.55 m/s    p50 0.93 m/s    flat 0.4 m/s
    speed sent, max           7.10 m/s        8.30 m/s        24.6 (20 Oct CSV)
    fix / satellites          3 / 12          3 / 12          3 / 12

Every row of this file pins one of those back, and names where it came from.
The aircraft is the authority here, not the reasoning - these are numbers it
flew on.
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "vps_vio"))

from map_fuse import MapFuser  # noqa: E402
from velocity_bridge import VelocityBridge  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


src = open(os.path.join(HERE, "vps_vio", "live_node.py")).read()

# ------------------------------------------------- the GPS_INPUT rate

print("GPS_INPUT goes out at a receiver's cadence, not the flow loop's")
check("there is a rate gate at all", "self._next_send" in src,
      "without one it sends on every frame, 30 Hz")
check("the period is the 4.3 Hz the working flights carried",
      "self.send_period_s = 1.0 / 4.3" in src,
      "logs 00000191 and 00000194 measured 4.28 and 4.31 Hz")
check("skipped sends are counted, not silent", "self.skipped_rate" in src)
check("and the gate returns BEFORE building the message",
      src.index("self.skipped_rate += 1") < src.index("gps_input_send"),
      "a gate after the work still does the work")

# Seven correlated samples where the EKF expects one independent measurement
# is not a cosmetic difference: it inflates the effective weight by about the
# square root of the over-sampling.
over = 30.0 / 4.3
check("the over-sampling this removes is about 7x", 6.5 < over < 7.5,
      f"{over:.1f}x, which weighs the VPS ~{over ** 0.5:.1f}x too heavily")

# ------------------------------------------------- the accuracies

print("\nwhat we declare about our own error")
check("vertical accuracy is the constant the flights sent",
      "VERT_ACC_M = 3.0" in src and "self.VERT_ACC_M," in src,
      "GPA.VAcc is exactly 3.00 through both logs")
check("it is no longer sigma x 2", "float(sigma_m * 2.0)" not in src,
      "at a typical sigma of 21 m that declared 42 m")
check("horizontal accuracy is still the fuser's own sigma",
      "float(sigma_m), self.VERT_ACC_M" in src,
      "p50 21.28 and 21.36 m in the two logs - the fuser's sigma, as designed")
check("speed accuracy scales with height",
      "self.VEL_ACC_PER_M * max(0.0, vel.altitude_m)" in src,
      "the flights sent 0.93-1.55 m/s from a measured window, never a flat value")

# ------------------------------------------------- the output slew

print("\nthe output slew, from the 9 Oct report")
b = VelocityBridge()
check("the bridge has a slew limit", hasattr(b, "slew_mps2"))
check("at the 5 m/s2 that flew", abs(b.slew_mps2 - 5.0) < 1e-9,
      '"input gate only -> 5 m/s^2 slew, 2 s hold"')
check("with the 2 s hold", abs(b.slew_hold_s - 2.0) < 1e-9)

# It has to actually bite. Feed a step far beyond the airframe and watch what
# leaves, not what the estimate holds.
t = 0.0
for _ in range(60):
    t += 1.0 / 30.0
    b.push_flow(t, 0.0, 0.0)
    b.value(t, 10.0)
b2 = VelocityBridge(max_accel_mps2=1e9)      # let the INPUT gate pass it
t2 = 0.0
for _ in range(60):
    t2 += 1.0 / 30.0
    b2.push_flow(t2, 0.0, 0.0)
    b2.value(t2, 10.0)
worst = 0.0
for _ in range(10):
    t2 += 1.0 / 30.0
    b2.push_flow(t2, 25.0, 0.0)
    st = b2.value(t2, 10.0)
    worst = max(worst, (st.sent_vn ** 2 + st.sent_ve ** 2) ** 0.5)
# 10 frames at 30 Hz is a third of a second; 5 m/s2 allows about 1.7 m/s.
check("a 25 m/s step cannot leave as 25 m/s", worst < 3.0,
      f"most that got out in a third of a second: {worst:.2f} m/s")
check("and the slew says it bit", b2.slewed > 0, f"{b2.slewed} times")
check("the input gate alone would not have done this",
      b2.rejected_jump == 0,
      "max_accel was disabled for this case on purpose - the 9 Oct report's "
      "first three words are 'input gate only'")

# ------------------------------------------------- the fuser's trust in a fix

print("\nhow much the filter trusts a map fix")
f = MapFuser()
check("fix_sigma_m is 20, not 30 and not the 8 of 23 Oct morning",
      abs(f.fix_var ** 0.5 - 20.0) < 1e-6,
      "this line guarded 20 against a 30 that reappeared with the rebuild, "
      "and on 23 Oct it also had to be guarded against an 8 that came from a "
      "sweep scored against the map's own fixes. Against the raw receiver "
      "that 8 was four times worse above 100 m. A number arriving here needs "
      "to say which truth it was measured against")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
