"""The height this node divides by is the barometer's, at every altitude.

The lidar is logged and flown on by nothing. Two comments claimed otherwise
until 25 Oct - config.py said "Height stays TF03 below lidar_max_alt_m,
barometer above" and live_node repeated it - and lidar_max_alt_m is read by
no code at all. A comment that describes a path the code does not take is
worse than no comment, because the next person plans around it.

WHY IT MATTERS. Velocity is pixels x altitude / focal, so an altitude error
is a velocity error of the same proportion. Measured on flight 245 against
the lidar:

    lidar 0-5 m      alt_m  1.80    ratio 2.86
    lidar 5-15 m     alt_m 16.40    ratio 1.72
    lidar 15-30 m    alt_m 29.00    ratio 1.29
    lidar 50-80 m    alt_m 82.00    ratio 1.08
    lidar 80-120 m   alt_m 107.7    ratio 1.07

A near-constant 5-7 m offset. The SAME climbing and descending - +6.59,
+6.82, +7.00 against +6.63, +6.02, +5.24 - so it is an offset and not the
descent hysteresis remembered from last time. At touchdown the lidar read
0.60 m and alt_m read 4.2.

Six percent at 100 m is nothing. Seventy-two percent at 10 m is a phantom
three quarters the size of real motion, and low level is where the aircraft
was circling.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vps_vio"))
from config import VpsConfig  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        fails.append(name)


HERE = os.path.join(os.path.dirname(__file__), "..", "vps_vio")
src = {n: open(os.path.join(HERE, n)).read()
       for n in ("live_node.py", "config.py", "velocity_bridge.py",
                 "flow_odom.py", "map_fix_worker.py")}

print("lidar_max_alt_m is a setting nothing reads")
# A READ, not a mention: live_node names it in the comment that says it is
# dead, and that comment is the point of this file.
import re
_use = re.compile(r'(\.lidar_max_alt_m\b|["\']lidar_max_alt_m["\'])')
readers = [n for n, s in src.items()
           if n != "config.py" and _use.search(s)]
check("no module outside config actually reads it", not readers, str(readers))
check("and config still carries it, with the truth beside it",
      hasattr(VpsConfig(), "lidar_max_alt_m")
      and "read by nothing" in src["config.py"],
      "kept so the next attempt is a setting and not a rewrite")

print("\nthe comments no longer claim the lidar carries the height")
check("live_node says the barometer does",
      "BAROMETER AND THE EKF AT EVERY" in src["live_node.py"])
check("and does not still say the lidar takes over below 60 m",
      "Height comes from the lidar below 60 m" not in src["live_node.py"])
check("config records what the offset costs",
      "ratio 1.72" in src["config.py"],
      "the number is the point - 72% at 10 m")

print("\nthe lidar is still logged, because that is how the next decision gets made")
check("rngfnd_m is a column", '"rngfnd_m"' in src["live_node.py"])
check("and it still guards the pre-arm zero", '"lidarzero"' in src["live_node.py"],
      "on and logged, flown on by nothing - which is what the pilot asked for")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
