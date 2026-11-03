#!/usr/bin/env python3
"""Every parameter back to what flew on 27 October, measured from the logs.

The pilot asked for the aircraft to be put back exactly as it was on 27 Oct,
battery aside, because that is the configuration the working flight and the
WP mission flew. This builds the list from the two FC logs rather than from
anyone's memory of what was touched:

    27 Oct   flight_logs/00000258.BIN   (15:42, the board that flew it)
    3 Nov    flight_logs/00000118.BIN   (today)

THE BOARD IS NOT THE SAME ONE. The X7+ went in on 31 Oct, and the logs prove
it: STAT_BOOTCNT 817 -> 63, STAT_FLTCNT 296 -> 14, INS_ACC2_ID 2622002 ->
2621962, and INS_ACC3_ID 0 -> 2622002 for a third IMU that did not exist
before. A board cannot lose boots. So the per-board calibration is NOT
restored - writing the old board's accelerometer offsets and compass
calibration onto different physical chips would fly this aircraft on another
board's numbers. Everything that is a decision rather than a calibration does
go back.
"""
import re
from pymavlink import mavutil

OLD = "flight_logs/00000258.BIN"
NEW = "flight_logs/00000118.BIN"
OUT = "reports/restore_to_27oct.parm"

EXCLUDE = [
 # Narrow on purpose. ^(INS_ACC|INS_GYR) would swallow INS_ACCEL_FILTER and
 # INS_GYRO_FILTER, two cutoffs that feed the control loop and are not
 # calibration at all. They happen to match today (10 and 28) but the regex
 # must not be the thing that decides that.
 (re.compile(r"^INS_(ACC\d*(OFFS|SCAL)|GYR\d*OFFS|(ACC|GYR)\d*_(ID|CALTEMP)"
             r"|ACC_BODYFIX|GYR_CAL)"), "this board's accel/gyro calibration"),
 (re.compile(r"^COMPASS_(OFS|DIA|ODI|MOT|DEV_ID|PRIO|SCALE|CAL_FIT|DEC)"),
  "this board's compass calibration"),
 (re.compile(r"^AHRS_TRIM_"), "this board's level calibration"),
 (re.compile(r"^BARO\d*_DEVID$"), "barometer hardware id"),
 (re.compile(r"^BARO\d*_GND_PRESS$"),
  "ground pressure, re-set every boot - an old one makes altitude wrong"),
 (re.compile(r"^RC\d+_(MIN|MAX|TRIM|DZ|REVERSED)$"),
  "today's RC calibration - an old one breaks the sticks and the switch"),
 (re.compile(r"^STAT_"), "counters: boot count, flight time, distance"),
 (re.compile(r"^MOT_THST_HOVER$"), "the FC learns this in flight"),
 (re.compile(r"^SCR_USER[12]$"), "the site, typed on the RC by the pilot"),
 # The pilot's own instruction: the power module is a different type and is
 # already set correctly, so nothing in the battery group is touched.
 (re.compile(r"^(BATT\d*_|MOT_BAT_VOLT_)"),
  "PILOT: different power module, already correct"),
 # MIS_TOTAL is maintained by the mission upload, not written by hand. Setting
 # it to 15 with one item stored would have AUTO fly to whatever is in those
 # slots. The 27 Oct mission has to be re-uploaded instead.
 (re.compile(r"^MIS_TOTAL$"),
  "set by uploading the mission, not by hand - re-upload the 27 Oct WP file"),
]

def params(path):
    m = mavutil.mavlink_connection(path)
    out = {}
    while True:
        msg = m.recv_match(type=["PARM"], blocking=False)
        if msg is None:
            break
        d = msg.to_dict()
        out[d["Name"]] = d["Value"]
    return out

old, new = params(OLD), params(NEW)
# A relative threshold, because logged floats are single precision and
# LOIT_ACC_MAX_M reads 2.627684 against 2.62768 - the same number twice.
def differs(a, b):
    return abs(a - b) > max(1e-6, 1e-5 * max(abs(a), abs(b)))

diff = {k: (old[k], new[k]) for k in set(old) & set(new)
        if differs(old[k], new[k])}
# Present on 27 Oct and gone today: a parent switch is off, so the children
# are hidden. Restoring the parent brings them back, so they are listed too.
gone = {k: old[k] for k in sorted(set(old) - set(new))}

restore, skipped = {}, []
for k in sorted(list(diff) + list(gone)):
    why = next((w for rx, w in EXCLUDE if rx.match(k)), None)
    if why:
        skipped.append((k, old[k], new.get(k), why))
    else:
        restore[k] = old[k]

GROUPS = [
 ("NAVIGATION AND THE EKF - what the estimate is built from",
  re.compile(r"^(EK3_|AHRS_|GPS\d*_|INS_USE)")),
 ("COMPASS - which ones are used, at the pilot's word",
  re.compile(r"^COMPASS_")),
 ("HEIGHT - rangefinder, surface tracking, terrain",
  re.compile(r"^(RNGFND|SURFTRAK|TERRAIN)")),
 ("SPEED AND LOITER",
  re.compile(r"^(LOIT_|WP_SPD|PILOT_SPD)")),
 ("MOTORS",
  re.compile(r"^MOT_")),
 ("FLIGHT MODES, MISSION, FENCE",
  re.compile(r"^(FLTMODE|INITIAL_MODE|MIS_|FENCE_)")),
 ("PRECISION LANDING - it was enabled on 27 Oct",
  re.compile(r"^PLND_")),
 ("SERIAL PORTS AND BOARD",
  re.compile(r"^(SERIAL|BRD_)")),
 ("LOGGING",
  re.compile(r"^(LOG_|INS_LOG)")),
]

lines = [
 "# DroneVPS - every parameter back to the 27 October configuration.",
 "#",
 "# Built by fc_scripts/build_27oct_restore.py from the two FC logs:",
 "#   27 Oct  flight_logs/00000258.BIN  (15:42)",
 "#   3 Nov   flight_logs/00000118.BIN",
 "#",
 "# The flight controller is NOT the same board - the X7+ went in on 31 Oct,",
 "# and the logs prove it (STAT_BOOTCNT 817 -> 63, INS_ACC2_ID 2622002 ->",
 "# 2621962, a third IMU that did not exist before). So this file carries the",
 "# DECISIONS and leaves this board's own CALIBRATION alone. What was left out",
 "# is listed at the bottom, with the reason for each.",
 "#",
 f"# {len(restore)} parameters to write.",
 "#",
 "# Load in Mission Planner or QGroundControl, then reboot the FC. Read the",
 "# groups before writing - nothing here is urgent enough to load unread.",
 "",
]
done = set()
for title, rx in GROUPS:
    rows = [(k, v) for k, v in restore.items() if rx.match(k) and k not in done]
    if not rows:
        continue
    lines.append(f"# ---- {title}")
    for k, v in rows:
        done.add(k)
        was = new.get(k)
        note = "absent today" if was is None else f"today {was:g}"
        lines.append(f"{k},{v:.6f}".rstrip("0").rstrip(".") + f"    # {note}")
    lines.append("")
rest = [(k, v) for k, v in restore.items() if k not in done]
if rest:
    lines.append("# ---- everything else that differs")
    for k, v in rest:
        was = new.get(k)
        note = "absent today" if was is None else f"today {was:g}"
        lines.append(f"{k},{v:.6f}".rstrip("0").rstrip(".") + f"    # {note}")
    lines.append("")

lines.append("# ======== DELIBERATELY NOT RESTORED ========")
by_why = {}
for k, o, n, why in skipped:
    by_why.setdefault(why, []).append((k, o, n))
for why, rows in sorted(by_why.items()):
    lines.append(f"#")
    lines.append(f"# {why}  ({len(rows)})")
    for k, o, n in rows:
        lines.append("#   %-22s 27 Oct %12.4f   today %s"
                     % (k, o, "absent" if n is None else "%.4f" % n))
lines.append("")

open(OUT, "w").write("\n".join(lines))
print(f"{OUT}: {len(restore)} to write, {len(skipped)} left out")
