#!/usr/bin/env python3
"""Build the restore list from the MEASURED diff, not from my judgement.

The first list was 131 parameters I picked by hand and it missed about forty,
including the whole of the attitude tuning. This one starts from every
parameter that differs between the FC as it stands and the configuration that
flew as 238, and takes things OUT for stated reasons rather than guessing what
to put in.
"""
import json
import re

NOW = "/tmp/claude-1000/-data-DRONE/86eaaeb5-803c-484e-beba-357c25eb6503/scratchpad/fc_now.json"
OLD = "/home/abc/drone_param_backups/fc_params_238_vs_286.txt"
OUT = "/data/DRONE/reports/restore_from_diff.parm"

now = json.load(open(NOW))
old = {}
for line in open(OLD):
    m = re.match(r"^([A-Z][A-Z0-9_]*)\s+(\S+)\s+(\S+)", line)
    if m and m.group(2) != "ABSENT":
        old[m.group(1)] = float(m.group(2))

# WHAT IS LEFT OUT, and why. Each of these is a deliberate decision, not an
# oversight, which is the difference between this list and the last one.
EXCLUDE = [
 # Narrow on purpose. The first version matched ^(INS_ACC|INS_GYR) and
 # swallowed INS_ACCEL_FILTER and INS_GYRO_FILTER with it - two filter
 # cutoffs that feed the control loop and are not calibration at all. They
 # sat at 20/20 against the 10/28 that flew, and nothing would have said so.
 (re.compile(r"^INS_(ACC\d*(OFFS|SCAL)|GYR\d*OFFS|(ACC|GYR)\d*_(ID|CALTEMP)"
             r"|ACC_BODYFIX|GYR_CAL)"),
  "today's accelerometer and gyro calibration"),
 (re.compile(r"^COMPASS_(OFS|DIA|ODI|MOT|DEV_ID|PRIO|SCALE|CAL_FIT|DEC)"),
  "today's compass calibration"),
 (re.compile(r"^AHRS_TRIM_"),
  "today's level calibration"),
 (re.compile(r"^BARO\d*_DEVID$"),
  "barometer hardware id"),
 (re.compile(r"^BARO\d*_GND_PRESS$"),
  "ground pressure, set at every boot - an old one makes altitude wrong"),
 (re.compile(r"^RC\d+_(MIN|MAX|TRIM|DZ|REVERSED)$"),
  "today's RC calibration - an old one breaks the sticks and the switch"),
 (re.compile(r"^STAT_"),
  "counters - boot count, flight time"),
 (re.compile(r"^MOT_THST_HOVER$"),
  "the FC learns this in flight"),
 (re.compile(r"^(WP_SPD|WP_SPD_UP|WP_SPD_DN|LOIT_SPEED_MS|PILOT_SPD_UP)$"),
  "the speeds changed on 30 Oct, on purpose"),
 # Measured on 31 Oct with motor_spin_cal.py, props off: all six start at
 # PWM 1145 and run steadily at 1160. The 0.140/0.150 in the snapshot came
 # across from the quadcopter this flight controller used to fly and had
 # never been measured on 4008s with 16 inch props. Restoring them would
 # put a guess back over a measurement.
 (re.compile(r"^MOT_SPIN_(ARM|MIN)$"),
  "measured 31 Oct by spinning the motors, not the quad's inherited guess"),
 (re.compile(r"^SCR_USER[12]$"),
  "the demo site, typed on the RC"),
 (re.compile(r"^COMPASS_(USE2|USE3|ORIENT3)$"),
  "internal compass stays off, at the pilot's word"),
 (re.compile(r"^MIS_TOTAL$"),
  "a different mission"),
 (re.compile(r"^GPS_PRIMARY$"),
  "the RC switch owns this, not a file"),
 (re.compile(r"^ARMING_SKIPCHK$"),
  "it SKIPS arming checks. The GPS pre-arm has to be answered by "
  "injecting, not by switching the check off"),
]

keep, dropped = [], {}
for k in sorted(set(old) & set(now)):
    if abs(old[k] - now[k]) <= max(abs(old[k]) * 1e-6, 1e-6):
        continue
    for rx, why in EXCLUDE:
        if rx.match(k):
            dropped.setdefault(why, []).append(k)
            break
    else:
        keep.append((k, now[k], old[k]))

lines = ["# DroneVPS - back to the flight 238 configuration.",
         "#",
         "# Every line here is a parameter MEASURED to differ between the FC",
         "# as it stands and the configuration that flew as flight 238 with no",
         "# failsafe. Nothing is in here because I thought it mattered.",
         "#"]
for why, names in dropped.items():
    lines.append(f"# LEFT OUT ({len(names)}) - {why}:")
    for i in range(0, len(names), 4):
        lines.append("#     " + "  ".join(names[i:i + 4]))
lines.append("")
for k, cur, want in keep:
    lines.append(f"{k},{want:g}      # abhi {cur:g}")
open(OUT, "w").write("\n".join(lines) + "\n")

print(f"{OUT}")
print(f"  lagane hai : {len(keep)}")
print(f"  chhode     : {sum(len(v) for v in dropped.values())}")
print()
print("  %-22s %14s %14s" % ("param", "ab FC par", "238 (lagana hai)"))
print("  " + "-" * 54)
for k, cur, want in keep:
    print("  %-22s %14.6g %14.6g" % (k, cur, want))
