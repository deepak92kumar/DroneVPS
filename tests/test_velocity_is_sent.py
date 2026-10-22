"""The velocity must reach the flight controller. Withholding it is not a fallback.

The archived pre-SSD reports tried it and measured the result:

    "send-veloff -> ON. Without it the EKF differentiates our position and
     believed 0.21 m/s while the aircraft did 1.09."

A fifth of the true speed, with nothing aboard able to tell - the 8 Oct
failure mode reached by another route. The EKF's own differentiation of a
GPS_INPUT position is heavily filtered, so it lags and attenuates.

This test exists because the obvious response to a high XKF4.SV is to stop
sending velocity and let the position carry the flight, and on 22 Oct 2026
that was about to be proposed. The position IS the good signal (SP never left
0.56 against a 1.2 gate on log 226, and the GPS->VPS handover lands within a
metre), which makes the idea look safe. It is not.

The answer to a noisy velocity is to measure it better - VpsConfig
.flow_vel_frames - and to declare its accuracy honestly - VEL_ACC_PER_M.
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "vps_vio"))

FAILED = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}   {detail}")
    if not ok:
        FAILED.append(label)


print(__doc__)

src = open(os.path.join(HERE, "vps_vio", "live_node.py")).read()

print("the sender is built to send velocity by default")
check("GpsInputSender takes send_velocity=True", "send_velocity: bool = True" in src)
# The IGN_VEL_HORIZ bit is how a GPS_INPUT message says "no velocity here". It
# has to exist - a stale or invalid estimate must be withheld - but it must be
# driven by the estimate's own validity, never by a blanket setting.
check("the ignore bit exists, for a stale or invalid estimate",
      "IGN_VEL_HORIZ" in src)
i = src.index("def velocity_fields")
j = src.index("\n    def ", i + 10)
body = src[i:j]
check("velocity_fields decides per sample, on the bridge's own validity",
      "vel.valid" in body or "not vel" in body,
      "it must refuse a stale estimate, not all of them")

print("\nand the reason is written down where someone would look")
vb = open(os.path.join(HERE, "vps_vio", "velocity_bridge.py")).read()
check("velocity_bridge records the measured cost of withholding it",
      "0.21 m/s while the aircraft did 1.09" in vb)
check("and names what to do instead", "flow_vel_frames" in vb
      and "VEL_ACC_PER_M" in vb)

print("\nthe two things that ARE the answer are in place")
from config import VpsConfig
cfg = VpsConfig()
check("a multi-frame velocity baseline", cfg.flow_vel_frames > 1,
      f"flow_vel_frames = {cfg.flow_vel_frames}")
check("and an altitude-scaled declared accuracy",
      "VEL_ACC_PER_M = 0.048" in src,
      "0.6 + 0.048 x alt, fitted on log 226")

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
    sys.exit(1)
print("ALL PASS")
