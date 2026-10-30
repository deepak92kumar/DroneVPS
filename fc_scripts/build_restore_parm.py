#!/usr/bin/env python3
"""Pull everything this project set out of the 238 backup, in the order it
has to go back on after a hard reset."""
import re

SRC = "/home/abc/drone_param_backups/fc_params_238_vs_286.txt"
OUT = "/data/DRONE/reports/restore_after_hard_reset.parm"

vals = {}
for line in open(SRC):
    m = re.match(r"^([A-Z][A-Z0-9_]*)\s+(\S+)\s+(\S+)", line)
    if m and m.group(2) != "ABSENT":
        vals[m.group(1)] = float(m.group(2))

GROUPS = [
 ("1. THE LINK - this one first, over USB, or telem2 stays dead",
  ["SERIAL2_PROTOCOL", "SERIAL2_BAUD", "SERIAL2_OPTIONS",
   "SERIAL1_PROTOCOL", "SERIAL1_BAUD",
   "SERIAL3_PROTOCOL", "SERIAL3_BAUD",
   "SERIAL4_PROTOCOL", "SERIAL4_BAUD",
   "SERIAL5_PROTOCOL", "SERIAL6_PROTOCOL",
   "SERIAL7_PROTOCOL", "SERIAL7_BAUD"]),
 ("2. THE AIRFRAME",
  ["FRAME_CLASS", "FRAME_TYPE"]),
 ("3. BATTERY - the pins and multipliers are this board's, not defaults",
  ["BATT_MONITOR", "BATT_VOLT_PIN", "BATT_CURR_PIN", "BATT_VOLT_MULT",
   "BATT_AMP_PERVLT", "BATT_CAPACITY", "BATT_ARM_VOLT", "BATT_LOW_VOLT",
   "BATT_CRT_VOLT", "BATT_FS_LOW_ACT", "BATT_FS_CRT_ACT"]),
 ("4. HARMONIC NOTCH - tuned for this airframe, lost in the reset",
  ["INS_HNTCH_ENABLE", "INS_HNTCH_MODE", "INS_HNTCH_FREQ", "INS_HNTCH_BW",
   "INS_HNTCH_REF", "INS_HNTCH_HMNCS", "INS_HNTCH_ATT", "INS_HNTCH_FM_RAT",
   "INS_HNTCH_OPTS"]),
 ("5. GPS AND THE VPS INJECTION - GPS2 is where our estimate arrives",
  ["GPS_PRIMARY", "GPS2_TYPE", "GPS_AUTO_SWITCH", "GPS_AUTO_CONFIG",
   "GPS1_DELAY_MS", "GPS2_DELAY_MS", "GPS1_RATE_MS", "GPS2_RATE_MS",
   "GPS2_POS_X", "GPS2_POS_Y", "GPS2_POS_Z"]),
 ("6. EKF SOURCES - what the estimate is actually built from",
  ["AHRS_EKF_TYPE", "EK3_ENABLE", "EK3_SRC1_POSXY", "EK3_SRC1_VELXY",
   "EK3_SRC1_POSZ", "EK3_SRC1_VELZ", "EK3_SRC1_YAW",
   "EK3_SRC2_POSXY", "EK3_SRC2_VELXY", "EK3_SRC_OPTIONS",
   "EK3_GLITCH_RAD", "EK3_POSNE_M_NSE", "EK3_VELNE_M_NSE", "EK3_VELD_M_NSE",
   "EK3_GPS_CHECK"]),
 ("7. FAILSAFE - FS_OPTIONS bit 0 is what lets a WP mission continue "
  "through an RC failsafe",
  ["FS_THR_ENABLE", "FS_THR_VALUE", "FS_OPTIONS", "FS_GCS_ENABLE",
   "FS_GCS_TIMEOUT", "FS_EKF_ACTION", "FS_EKF_THRESH", "FS_EKF_FILT",
   "FS_CRASH_CHECK", "FS_VIBE_ENABLE", "FS_DR_ENABLE", "FS_DR_TIMEOUT",
   "RC_FS_TIMEOUT"]),
 ("8. SPEEDS - as set on 30 Oct, not as flown on 27 Oct",
  ["WP_SPD", "WP_SPD_UP", "WP_SPD_DN", "WP_ACC", "WP_ACC_Z", "WP_JERK",
   "WP_RADIUS_M", "WP_YAW_BEHAVIOR", "WP_RFND_USE", "WP_NAVALT_MIN",
   "LOIT_SPEED_MS", "LOIT_ACC_MAX_M",
   "PILOT_SPD_UP", "PILOT_SPD_DN", "PILOT_ACC_Z",
   "PILOT_Y_RATE", "ACRO_Y_RATE",
   "LAND_SPD_MS", "LAND_SPD_HIGH_MS", "LAND_ALT_LOW_M", "LAND_REPOSITION",
   "RTL_ALT_M", "RTL_ALT_FINAL_M", "RTL_ALT_TYPE", "RTL_SPEED_MS"]),
 ("9. RANGEFINDER - off, and it has to stay off until the lidar is back",
  ["RNGFND1_TYPE"]),
 ("10. THE ORIGIN THE VPS FLIES FROM - retype SCR_USER1/2 at the new site",
  ["SCR_ENABLE", "SCR_USER1", "SCR_USER2", "SCR_USER3"]),
 ("11. RC AND FLIGHT MODES - RC6 1045/1945 is the GPS/VPS switch",
  ["RC_PROTOCOLS", "RC_OPTIONS", "FLTMODE_CH",
   "FLTMODE1", "FLTMODE2", "FLTMODE3", "FLTMODE4", "FLTMODE5", "FLTMODE6",
   "RC6_OPTION", "RC6_MIN", "RC6_MAX", "RC6_TRIM", "RC6_DZ", "RC6_REVERSED"]),
 ("12. PRECISION LANDING - enabled, but nothing feeds it since 26 Sep",
  ["PLND_ENABLED", "PLND_TYPE", "PLND_EST_TYPE", "PLND_ALT_MIN",
   "PLND_ALT_MAX", "PLND_XY_DIST_MAX", "PLND_STRICT", "PLND_TIMEOUT",
   "PLND_RET_MAX", "PLND_LAG", "PLND_ACC_P_NSE", "PLND_BUS"]),
]

# 30 Oct: the speeds were changed after the backup was taken.
# The demo site, read back off the FC on 29 Oct after the pilot typed it
# on the RC. The 238 snapshot holds the OLD site, 37 km away.
NEWER = {"SCR_USER1": 30.555281, "SCR_USER2": 74.104889,
         "WP_SPD": 3.0, "LOIT_SPEED_MS": 3.0, "WP_SPD_DN": 1.1,
         "WP_SPD_UP": 1.5, "PILOT_SPD_UP": 1.5}

lines, n, missing = [], 0, []
lines.append("# DroneVPS - what to put back after the FC hard reset.")
lines.append("# Built from the 238/286 parameter snapshot, which is a")
lines.append("# configuration that flew with no failsafe, plus the five")
lines.append("# speeds changed on 30 Oct.")
lines.append("#")
lines.append("# NOT in here, on purpose: accelerometer, compass and RC")
lines.append("# calibration. Those are physical and belong to a sensor on a")
lines.append("# day, not to a file - do them on the aircraft.")
lines.append("")
for title, names in GROUPS:
    lines.append("")
    lines.append("# " + title)
    for k in names:
        if k in NEWER:
            fmt = "{:.9f}" if k.startswith("SCR_USER") else "{:g}"
            note = ("# the DEMO site, not the old one"
                    if k.startswith("SCR_USER") else
                    "# 30 Oct, not the 27 Oct value")
            lines.append(f"{k}," + fmt.format(NEWER[k]) + "      " + note)
            n += 1
        elif k in vals:
            v = vals[k]
            # %g keeps six significant digits, which on a latitude is ten
            # metres. Anything geographic gets written in full.
            fmt = "{:.9f}" if k in ("SCR_USER1", "SCR_USER2") else "{:g}"
            lines.append(f"{k}," + fmt.format(v))
            n += 1
        else:
            missing.append(k)
open(OUT, "w").write("\n".join(lines) + "\n")
print(f"{OUT}\n{n} parameters")
if missing:
    print("backup me nahi mile:", ", ".join(missing))
