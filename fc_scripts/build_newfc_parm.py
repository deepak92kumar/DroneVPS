#!/usr/bin/env python3
"""Everything to set on a REPLACEMENT flight controller, same airframe.

Built from the flight 238 snapshot - a configuration that flew with no
failsafe - with every deliberate change measured since folded in.

What is left out is as important as what is in. A parameter that describes
the OLD board's sensors is worse than useless on a new one: it hides whether
the new board was ever calibrated, and it reads as a fault that is not there.
So all calibration, every device id, the counters and the learned hover
throttle are dropped, and the new board gets them from its own calibration.

    python3 build_newfc_parm.py
"""
import re

SRC = "/home/abc/drone_param_backups/fc_params_238_vs_286.txt"
OUT = "/data/DRONE/reports/new_fc_setup.parm"

old = {}
for line in open(SRC):
    m = re.match(r"^([A-Z][A-Z0-9_]*)\s+(\S+)\s+(\S+)", line)
    if m and m.group(2) != "ABSENT":
        old[m.group(1)] = float(m.group(2))

# Measured after the snapshot was taken. Each one replaces the 238 value.
OVERRIDE = {
    # 30 Oct, at the pilot's instruction
    "WP_SPD": 3.0, "LOIT_SPEED_MS": 3.0, "WP_SPD_UP": 1.5,
    "WP_SPD_DN": 1.1, "PILOT_SPD_UP": 1.5,
    # 31 Oct, motor_spin_cal.py with the props off: all six start at PWM 1145
    # and run steadily at 1160. The snapshot holds 0.140/0.150, which came
    # across from the quadcopter this flight controller used to fly and was
    # never measured on 4008s with 16 inch props.
    "MOT_SPIN_ARM": 0.145, "MOT_SPIN_MIN": 0.160,
    # The demo site, typed on the RC. The snapshot holds the old one, 37 km
    # away, and a wrong origin is a wrong position from the first frame.
    "SCR_USER1": 30.555281, "SCR_USER2": 74.104889,
}

# Each pattern says why, and the why is printed into the file. Anything that
# belongs to a sensor, a board or a flight history stays with the old board.
DROP = [
 (r"^INS_(ACC\d*(OFFS|SCAL)|GYR\d*OFFS|(ACC|GYR)\d*_(ID|CALTEMP)|ACC_BODYFIX|GYR_CAL)",
  "accelerometer and gyro calibration - the new board calibrates itself"),
 (r"^COMPASS_(OFS|DIA|ODI|MOT|DEV_ID|PRIO|SCALE|CAL_FIT|DEC)",
  "compass calibration and device ids"),
 (r"^AHRS_TRIM_", "level calibration"),
 (r"^BARO\d*_(DEVID|GND_PRESS)", "barometer id and its per-boot reference"),
 (r"^RC\d+_(MIN|MAX|TRIM|DZ|REVERSED)",
  "RC calibration - do it on the new board, but keep RC6 at 1045/1945"),
 (r"^STAT_", "flight counters from the old board"),
 (r"^MOT_THST_HOVER$", "the flight controller learns this in the air"),
 (r"^(SYSID_SW|FORMAT_VERSION|INS_(ACC|GYR)\d*_ID)", "firmware and hardware ids"),
 (r"^COMPASS_USE[23]?$",
  "set after calibrating the new board's compasses, not before"),
]
DROP = [(re.compile(p), w) for p, w in DROP]

# Same family, but not the same board. Flagged rather than silently copied.
CHECK = {
    "BATT_VOLT_PIN": "X7 vs X7+ analog pin numbering may differ",
    "BATT_CURR_PIN": "X7 vs X7+ analog pin numbering may differ",
    "BATT_VOLT_MULT": "depends on the power module AND the pin",
    "BATT_AMP_PERVLT": "depends on the power module AND the pin",
}

keep, dropped = [], {}
for k in sorted(old):
    for rx, why in DROP:
        if rx.match(k):
            dropped.setdefault(why, []).append(k)
            break
    else:
        keep.append((k, OVERRIDE.get(k, old[k]), k in OVERRIDE, CHECK.get(k)))

lines = [
 "# DroneVPS - what to set on a replacement flight controller.",
 "# Same airframe, CUAV X7 -> X7+.",
 "#",
 "# Base: the flight 238 snapshot (27 Oct), a configuration that flew with",
 "# no failsafe, with every measured change since folded in and marked.",
 "#",
 "# LOAD THIS FIRST, THEN CALIBRATE. Not the other way round: FRAME_CLASS",
 "# and the SERIAL ports have to be right before anything else makes sense.",
 "#",
 "# AFTER loading, on the aircraft:",
 "#   1. accelerometer calibration (6 point)",
 "#   2. compass calibration - then decide COMPASS_USE2 from what it finds",
 "#   3. RC calibration - RC6 must end at MIN 1045 / MAX 1945 or the",
 "#      GPS/VPS switch stops meaning anything",
 "#   4. check the four BATT_ lines marked CHECK below against the X7+ pinout",
 "#   5. python3 /data/DRONE/vps_vio/preflight.py",
 "#",
]
for why, names in dropped.items():
    lines.append(f"# NOT HERE ({len(names)}) - {why}")
lines.append("#")
lines.append(f"# {len(keep)} parameters follow.")
lines.append("")
for k, v, is_new, chk in keep:
    note = ""
    if is_new:
        note = "      # MEASURED after 238, not the snapshot value"
    elif chk:
        note = f"      # CHECK - {chk}"
    lines.append(f"{k},{v:.9g}{note}")
open(OUT, "w").write("\n".join(lines) + "\n")
print(OUT)
print(f"  lagane hai : {len(keep)}")
print(f"  chhode     : {sum(len(v) for v in dropped.values())}")
for why, names in dropped.items():
    print(f"     {len(names):4d}  {why}")
print()
print("  MEASURED (snapshot se alag):")
for k, v, is_new, _ in keep:
    if is_new:
        print(f"     {k:<16} {v:g}")
print()
print("  CHECK karne wale (X7+ par alag ho sakte hai):")
for k, v, _, chk in keep:
    if chk:
        print(f"     {k:<16} {v:g}   - {chk}")
