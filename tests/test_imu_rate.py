"""All three camera IMU sensors take their rate from config, and 100 is not a ceiling.

The three enabled reports - GYROSCOPE_CALIBRATED, LINEAR_ACCELERATION and
GAME_ROTATION_VECTOR - share one depthai packet, so they must be asked for at
one rate. Each used to be a literal 100 with a comment calling that "not a
choice, it is the ceiling", on Luxonis' table giving the calibrated gyro a
maximum of 100 Hz on the BNO086.

Measured on this device 21 Oct 2026, IMU alone: a request for 400 delivered
357.9 Hz on the calibrated gyro and on LINEAR_ACCELERATION alike. And on the
full pipeline, both mono streams and stereo depth up, 800p at 30 fps:

    requested   frames    IMU packets   accel/gyro/quat
      100 Hz    30.0 Hz       22006         99.9 Hz
      400 Hz    30.0 Hz       36601        165.9 Hz

The frame rate does not move, which is what 3 Oct's "dragged the whole stream
down" had been taken to rule out.

Why it is worth the link: log 225's in-flight camera accelerometer put 94% of
its energy above 10 Hz with every top peak at 49.8 Hz - hard against Nyquist
for a 100 Hz sample - while the same channel at rest spreads its energy and
reads sd 0.030 m/s2 against 1.30 in flight. At rest the fold is visible
directly: peaks at 36.0-48.7 Hz when sampled at 100, and the same content at
59.0-66.8 Hz when sampled at 400.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "vps_vio"))

from config import VpsConfig

FAILED = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}   {detail}")
    if not ok:
        FAILED.append(label)


print(__doc__)

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "vps_vio", "live_node.py")
src = open(src_path).read()

calls = re.findall(r"enableIMUSensor\(\s*dai\.IMUSensor\.(\w+)\s*,\s*([^)]+)\)", src)
print("the enableIMUSensor calls as written")
for name, rate in calls:
    print(f"        {name:24s} <- {rate.strip()}")

check("all three sensors are still enabled", len(calls) == 3, f"{len(calls)} found")
check("every one takes the configured rate, none a literal",
      all(c[1].strip() == "imu_rate" for c in calls),
      ", ".join(c[1].strip() for c in calls))
names = {c[0] for c in calls}
# GAME, not plain ROTATION_VECTOR: ROTATION_VECTOR fuses the magnetometer and
# shares the rotation slot with it, and the magnetometer sits beside the motors
# here - it drifted 18.9 deg/min with the BNO's own accuracy estimate at 79
# degrees. LINEAR_ACCELERATION, not ACCELEROMETER_RAW: push_accel is specified
# to take acceleration with gravity already removed.
check("GAME_ROTATION_VECTOR, not ROTATION_VECTOR",
      "GAME_ROTATION_VECTOR" in names and "ROTATION_VECTOR" not in names)
check("LINEAR_ACCELERATION, not ACCELEROMETER_RAW",
      "LINEAR_ACCELERATION" in names and "ACCELEROMETER_RAW" not in names)

print("\nthe configured rate")
cfg = VpsConfig()
print(f"        VpsConfig.imu_rate = {cfg.imu_rate}")
check("it is above 100, so Nyquist clears the aliased band", cfg.imu_rate > 100,
      f"{cfg.imu_rate}")
# Luxonis recommend the BNO's raw gyro at or below 400 - above that the part
# jitters - and the device rounds a request UP to its next supported rate, so
# asking beyond what it supports buys a slower packet, not a faster one.
check("and not above the 400 Hz the part is recommended at", cfg.imu_rate <= 400,
      f"{cfg.imu_rate}")

print("\nthe rate reaches LiveSource from config, and can be overridden")
check("LiveSource takes an imu_rate keyword", "imu_rate: int | None = None" in src)
check("it falls back to the config value",
      "cfg.imu_rate if imu_rate is None else int(imu_rate)" in src)
check("and main exposes it on the command line", '"--imu-rate"' in src)
check("which is passed to LiveSource", "imu_rate=a.imu_rate" in src)

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
    sys.exit(1)
print("ALL PASS")
