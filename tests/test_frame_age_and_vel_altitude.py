"""Two measurements that were wrong, and one gate that is new.

FRAME AGE. The old form anchored the host and device clocks on the first
frame and reported (now - t0) - (ts - ts0). The two clocks run at about 0.05%
different rates, so that walks away from zero no matter how fresh the frames
are: run 418 climbed 0.0000 -> 0.3492 s over 495 s in a straight line and run
406 FELL 0.2467 -> 0.0000 over 815 s, with 11% of rows sitting at exactly
0.0000 where the max(0.0, ...) clamp hid the negative half. frame_age_max_s
gates the flow on this number. It now uses the rolling minimum of (now - ts)
over a window, which is the only estimator of the offset that follows drift.

THE COUNTERS. n_frame_stale and n_flow_vetoed were both counted and written
nowhere - not the CSV, not the summary. Finding that 8339 of run 418's 12160
frames were over the age limit took a day because of it.

VELOCITY BY ALTITUDE. Measured on runs 238/406/411/418, the sent velocity's
per-axis p90 against gps1 is 0.63 m/s below 15 m and 1.12 by 25 m, against
EKF3's ~1.0 m/s GPS-quality threshold. So velocity is good near the ground
and noise at height, and the altitude decides rather than a flight-wide
switch.
"""
import ast
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "vps_vio"))

fails = []


def check(what, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + what
          + (f"   [{detail}]" if detail else ""))
    if not ok:
        fails.append(what)


src = open(os.path.join(ROOT, "vps_vio", "live_node.py"),
           encoding="utf-8").read()
import config as C  # noqa: E402

print("1) frame age is a latency, not a clock-drift reading")
check("the first-frame anchor is gone from the age calculation",
      "(now - self._dev_t0) - (ts - self._dev_ts0)" not in src)
check("a rolling offset window exists", "self._clk_off" in src
      and "collections.deque(maxlen=250)" in src)
check("the age is measured against the window's MINIMUM",
      "d - min(self._clk_off)" in src)

# The estimator itself, on synthetic input. SYNTHETIC: it models a drifting
# device clock, which is what the real data showed, and checks that a fresh
# frame reads ~0 and a late one reads its real lateness.
print("\n2) the estimator, on a synthetic drifting clock (SYNTHETIC)")
import collections  # noqa: E402
win = collections.deque(maxlen=250)
DRIFT = 0.0005          # s of host time per s, the measured rate
ages = []
for i in range(400):
    host = i * 0.04
    dev = host - DRIFT * host          # device clock runs slow
    late = 0.18 if i == 300 else 0.0   # one genuinely late frame
    d = (host + late) - dev
    win.append(d)
    ages.append(max(0.0, d - min(win)))
fresh = [a for i, a in enumerate(ages[100:]) if i + 100 != 300]
check("a fresh frame reads about zero even after 16 s of drift",
      max(fresh) < 0.02, f"max {max(fresh):.4f} s")
check("the one late frame reads its real lateness",
      abs(ages[300] - 0.18) < 0.02, f"{ages[300]:.4f} s vs 0.18")

print("\n3) both gate counters reach the CSV and the summary")
from live_node import CSV_COLUMNS  # noqa: E402
for c in ("n_frame_stale", "n_flow_vetoed"):
    check(f"{c} is a column", c in CSV_COLUMNS)
check("both are printed in the end-of-run summary",
      "frames too old" in src and "flow vetoed" in src)

print("\n4) velocity is gated by altitude")
cfg = C.VpsConfig()
check("vel_send_max_alt_m exists and is in the measured-good band",
      2.0 <= cfg.vel_send_max_alt_m <= 20.0, str(cfg.vel_send_max_alt_m))
check("the node reads it, not a hard-coded number",
      'getattr(_vps_config(),\n                                                "vel_send_max_alt_m", 15.0)' in src
      or '"vel_send_max_alt_m", 15.0' in src)
tree = ast.parse(src)
vf = next(n for n in ast.walk(tree)
          if isinstance(n, ast.FunctionDef) and n.name == "velocity_fields")
ifs = [n for n in vf.body if isinstance(n, ast.If)]
# The gate reads the bound through getattr into _max_alt, the same idiom
# n_frame_stale uses, because the velocity tests build a GpsInputSender
# without running __init__ so that nothing opens a serial port.
check("the altitude refusal comes before any velocity is computed",
      len(ifs) >= 2 and "_max_alt" in ast.dump(ifs[1].test)
      and "_alt" in ast.dump(ifs[1].test),
      "second branch must be the altitude gate")
check("the bound is read with getattr so a stubbed sender still works",
      '"vel_send_max_alt_m", 15.0' in src)
# 3 NOV: the gate now bounds the DECLARATION, not whether a velocity is
# sent, because IGN_VEL_HORIZ freezes the last velocity instead of
# withholding one (see test_velocity_off for the AP_GPS_MAV source and the
# 118 measurement). Above the gate the real measurement still goes out,
# carrying the full honest accuracy.
check("the gate no longer sets the ignore flags",
      src.count("IGN_VEL_HORIZ | self.IGN_SPEED_ACC") == 0)
check("above the gate a real measurement is still remembered and sent",
      "_remember_real_vel(vel)" in src and "_hold(f\"above" in src)

print("\nALL PASS" if not fails else f"\n{len(fails)} FAILED: {', '.join(fails)}")
sys.exit(1 if fails else 0)
