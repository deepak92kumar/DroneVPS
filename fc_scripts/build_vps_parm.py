#!/usr/bin/env python3
"""Only what the VPS needs. The aircraft already flies on GPS.

Everything here exists for one of six reasons: the Jetson has to reach the
flight controller, our estimate has to arrive as GPS2, the EKF has to be told
to use it, the origin has to come from the point typed on the RC, the pilot's
switch has to own the source, and a VPS that stops has to give ALT_HOLD and
never GPS. Nothing about tuning, the frame, the battery or the radio is in
here - the aircraft has those already.

    python3 build_vps_parm.py
"""
import re

SRC = "/home/abc/drone_param_backups/fc_params_238_vs_286.txt"
OUT = "/data/DRONE/reports/vps_only.parm"

old = {}
for line in open(SRC):
    m = re.match(r"^([A-Z][A-Z0-9_]*)\s+(\S+)\s+(\S+)", line)
    if m and m.group(2) != "ABSENT":
        old[m.group(1)] = float(m.group(2))
# The demo site, typed on the RC. The snapshot holds the old one, 37 km away.
old["SCR_USER1"], old["SCR_USER2"] = 30.555281, 74.104889

GROUPS = [
 ("1. THE LINK - Jetson talks to the FC on telem2. Without this nothing "
  "else here matters, and a hard reset puts BAUD back to 57600 while the "
  "router runs at 921600, which looks exactly like a dead Jetson.",
  [("SERIAL2_PROTOCOL", "2 = MAVLink2"),
   ("SERIAL2_BAUD", "921 = 921600, what the router opens"),
   ("SERIAL2_OPTIONS", "")]),

 ("2. OUR ESTIMATE ARRIVES AS GPS2. The node sends GPS_INPUT with "
  "--gps-id 1, which the FC files as its second GPS.",
  [("GPS2_TYPE", "14 = MAV. 0 here and the injection has nowhere to land"),
   ("GPS_AUTO_SWITCH", "0 = the FC never picks a GPS by itself"),
   ("GPS2_DELAY_MS", "0"),
   ("GPS2_RATE_MS", "200"),
   ("GPS2_POS_X", "camera offset from the FC, metres"),
   ("GPS2_POS_Y", ""),
   ("GPS2_POS_Z", "0.22 = camera below the FC")]),

 ("3. THE EKF IS TOLD TO USE IT. SRC1 is the set the aircraft flies on; "
  "POSXY/VELXY 3 means GPS, and which GPS is GPS_PRIMARY, which the RC "
  "switch owns.",
  [("AHRS_EKF_TYPE", "3 = EKF3"),
   ("EK3_ENABLE", "1"),
   ("EK3_SRC1_POSXY", "3 = GPS"),
   ("EK3_SRC1_VELXY", "3 = GPS"),
   ("EK3_SRC1_POSZ", "1 = barometer"),
   ("EK3_SRC1_VELZ", "0 = none"),
   ("EK3_SRC1_YAW", "1 = compass"),
   ("EK3_SRC2_POSXY", "0"),
   ("EK3_SRC2_VELXY", "0"),
   ("EK3_SRC_OPTIONS", "0"),
   ("EK3_GLITCH_RAD", "25 m - beyond this our position reads as a glitch"),
   ("EK3_POSNE_M_NSE", "0.5"),
   ("EK3_VELNE_M_NSE", "0.3"),
   ("EK3_VELD_M_NSE", "0.5"),
   ("EK3_GPS_CHECK", "0 - the receiver's own checks, off")]),

 ("4. THE ORIGIN. Every VPS position is measured from this point, so a "
  "wrong one is wrong from the first frame. SCR_USER1/2 is RETYPED at each "
  "new site and the node reads it only at startup, so power-cycle after.",
  [("SCR_ENABLE", "1 - without this SCR_USER is never read"),
   ("SCR_USER1", "latitude of the take-off point"),
   ("SCR_USER2", "longitude"),
   ("SCR_USER3", "")]),

 ("5. THE PILOT'S SWITCH. ch6 at 1045 asks for the M9N, 1945 for the VPS, "
  "and the middle holds. drone-rc-switch writes GPS_PRIMARY from it, so "
  "nothing else may be listening on that channel.",
  [("RC6_OPTION", "0 - no ArduPilot function on ch6, or flipping it would "
                  "do two things"),
   ("FLTMODE_CH", "5 - modes are on ch5, which keeps ch6 free"),
   ("GPS_PRIMARY", "the switch owns this. 1 = VPS, 0 = M9N")]),

 ("6. WHAT HAPPENS WHEN THE VPS STOPS. The pilot's rule: ALT_HOLD, never a "
  "quiet fall back to GPS. Under jamming the receiver is the attack, not "
  "the refuge.",
  [("FS_EKF_ACTION", "2 = ALT_HOLD. 1 would LAND"),
   ("FS_EKF_THRESH", "1.2"),
   ("FS_EKF_FILT", "5"),
   ("FS_OPTIONS", "17 - bit 0 lets a WP mission continue through an RC "
                  "failsafe"),
   ("FS_THR_ENABLE", "1"),
   ("RC_FS_TIMEOUT", "1")]),

 ("7. ARMING. Our honest HDOP is sigma/1.5, which is 13 at the 20 m the "
  "origin declares, against a GPS_HDOP_GOOD of 2.50 built for a satellite "
  "receiver. ARMING_SKIPCHK 6156 is how the aircraft has flown 290 times; "
  "it skips compass, GPS lock, safety switch and GPS config. On a NEW "
  "board leave it 0 for the first arms, then set 6156.",
  [("GPS_HDOP_GOOD", "250 = 2.50"),
   ("ARMING_SKIPCHK", "6156 - see above, start at 0 on a new board")]),

 ("8. RANGEFINDER - off. The TF03 is not on the aircraft; 1 here with no "
  "sensor is a pre-arm failure.",
  [("RNGFND1_TYPE", "0")]),
]

lines = ["# DroneVPS - only the parameters the VPS needs.",
         "# The aircraft already flies on GPS; nothing here touches tuning,",
         "# the frame, the battery or the radio.",
         "#",
         "# After loading: power-cycle, then check the node's own startup says",
         "#     origin from SCR_USER1/2 (typed on the RC): <your point>",
         "#       sheet covers this point: YES",
         "# If it says 'live receiver' or 'NO', do not fly on the VPS.",
         ""]
n = 0
for title, items in GROUPS:
    lines.append("")
    for i, chunk in enumerate(re.findall(r".{1,72}(?:\s|$)", title)):
        lines.append("# " + chunk.rstrip())
    for k, why in items:
        if k not in old:
            lines.append(f"# {k} - snapshot me nahi mila")
            continue
        v = old[k]
        s = f"{k},{v:.9g}"
        lines.append(f"{s:<28}# {why}" if why else s)
        n += 1
open(OUT, "w").write("\n".join(lines) + "\n")
print(f"{OUT}\n  {n} parameters")
