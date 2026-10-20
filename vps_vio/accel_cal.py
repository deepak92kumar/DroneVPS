#!/usr/bin/env python3
"""The full 6-position accelerometer calibration. PROPS OFF. GROUND ONLY.

Why this exists. Read off this aircraft on 20 Oct, at full float precision:

    INS_ACCSCAL_X/Y/Z    1.000000000  bit-identical, both IMUs, all six axes
    INS_ACCOFFS_*        set, and different per IMU
    INS_ACC1/2_CALTEMP   36.1 / 39.1 C   - so a calibration HAD run
    AHRS_TRIM_X/Y/Z      0.000000        - all three exactly zero

That is the fingerprint of ArduPilot's `simple_accel_cal()`, which sets the
scale to (1,1,1) and saves it, saves the offsets, saves caltemp, and zeroes
the AHRS trim. The 6-position path (`_acal_save_calibrations`) instead saves a
FITTED gain as the scale and computes a trim from the level sample. So what
had been run was "Calibrate Level", not "Calibrate Accel" - and note that
running the level one AFTER a full one silently puts the scales back to 1.0,
so both can be true at once.

How much it was costing, measured before this was written: |a| on the ground
9.8114 m/s2 on IMU0 (+0.05% of g) and 9.8224 on IMU1 (+0.16%), EKF-learned
accel bias +0.0029 / -0.0001 / -0.0127, ground attitude roll -0.36 pitch
-0.30 deg. Small - a level calibration is accurate at the attitude it was
taken at and drifts off with lean angle, and this aircraft is limited to 15
degrees by PSC_ANGLE_MAX. This is unfinished setup, not a fault.

    python3 accel_cal.py            # run it
    python3 accel_cal.py --dry-run  # show what is there now and stop

Everything goes through the ROUTER, never /dev/ttyTHS1. It refuses while
armed. It prints the old values before touching anything, and
`DRONE_fc_logs_backup/2026-10-20/params/accel_cal_before.txt` holds them too.

The six positions, in the order ArduPilot asks for them. Hold each one STILL
until this says it accepted it - the FC averages samples and a moving aircraft
is what makes a calibration fail:

    level      on its legs, as it sits
    LEFT       rolled 90 deg onto its left side
    RIGHT      rolled 90 deg onto its right side
    nose DOWN  pitched nose to the ground
    nose UP    pitched nose to the sky
    BACK       upside down

A reboot is needed afterwards; the FC asks for it itself.
"""
from __future__ import annotations
import argparse
import sys
import time

from pymavlink import mavutil

ROUTER = "udpout:127.0.0.1:14650"

# MAV_CMD_ACCELCAL_VEHICLE_POS, and the order ArduPilot asks for them in.
POS = [(1, "LEVEL", "on its legs, exactly as it sits"),
       (2, "LEFT", "rolled 90 degrees onto its LEFT side"),
       (3, "RIGHT", "rolled 90 degrees onto its RIGHT side"),
       (4, "NOSE DOWN", "pitched nose-down toward the ground"),
       (5, "NOSE UP", "pitched nose-up toward the sky"),
       (6, "BACK", "upside down, on its back")]

AXES = "XYZ"


class FC:
    def __init__(self, port):
        self.m = mavutil.mavlink_connection(port, source_system=200)
        self.m.mav.heartbeat_send(6, 8, 0, 0, 0)
        if self.m.wait_heartbeat(timeout=15) is None:
            sys.exit(f"no heartbeat on {port} - is the FC powered and "
                     "drone-router running?")
        if bool(self.m.motors_armed()):
            sys.exit("*** ARMED - ABORT. Calibrate disarmed, props off. ***")

    def beat(self):
        self.m.mav.heartbeat_send(6, 8, 0, 0, 0)

    def get(self, name, timeout=6.0):
        self.m.mav.param_request_read_send(
            self.m.target_system, self.m.target_component, name.encode(), -1)
        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = self.m.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
            if msg is not None and msg.param_id.strip("\x00") == name:
                return float(msg.param_value)
            self.beat()
        return None

    def drain_text(self, seconds, want=None):
        """Print the FC's own words. They are the instructions, not ours."""
        out = []
        t0 = time.time()
        while time.time() - t0 < seconds:
            msg = self.m.recv_match(type="STATUSTEXT", blocking=True, timeout=0.5)
            if msg is None:
                self.beat()
                continue
            txt = msg.text.strip()
            if txt.startswith("SRC=") or "VPS" in txt:
                continue          # our own node talking to the FC
            out.append(txt)
            print(f"      FC: {txt}")
            if want and any(w.lower() in txt.lower() for w in want):
                break
        return out


def show(fc, title):
    print(f"\n  {title}")
    for imu, (s, o, i) in ((1, ("INS_ACCSCAL_%s", "INS_ACCOFFS_%s", "INS_ACC_ID")),
                           (2, ("INS_ACC2SCAL_%s", "INS_ACC2OFFS_%s", "INS_ACC2_ID"))):
        if not fc.get(i):
            continue
        sc = [fc.get(s % a) for a in AXES]
        of = [fc.get(o % a) for a in AXES]
        print("    IMU%d scale %s   offset %s" % (
            imu,
            " ".join("%.6f" % v if v is not None else "--" for v in sc),
            " ".join("%+.4f" % v if v is not None else "--" for v in of)))
    for t in ("AHRS_TRIM_X", "AHRS_TRIM_Y"):
        print("    %-14s %s" % (t, fc.get(t)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=ROUTER)
    ap.add_argument("--dry-run", action="store_true",
                    help="print what is there now and change nothing")
    a = ap.parse_args()

    fc = FC(a.port)
    print("=" * 66)
    print("6-POSITION ACCELEROMETER CALIBRATION")
    print("=" * 66)
    show(fc, "before:")
    scale_now = fc.get("INS_ACCSCAL_X")
    if scale_now == 1.0:
        print("\n  scale is bit-identical 1.0 -> only a LEVEL calibration has "
              "ever been saved")

    if a.dry_run:
        print("\n  --dry-run, nothing sent.")
        return 0

    print("\n  PROPS OFF. The aircraft is about to be turned on its sides and")
    print("  its back. Nothing here arms it and no motor is commanded, but a")
    print("  propeller on a disarmed motor is still a propeller.")
    ans = input("\n  Props off and ready? [yes/no] ").strip().lower()
    if ans not in ("y", "yes"):
        print("  stopped, nothing sent.")
        return 1

    fc.m.mav.command_long_send(
        fc.m.target_system, fc.m.target_component,
        mavutil.mavlink.MAV_CMD_PREFLIGHT_CALIBRATION, 0,
        0, 0, 0, 0, 1, 0, 0)        # param5 = 1 -> accelerometer calibration
    print("\n  calibration started. The FC drives this, not us.")

    # THE FC ASKS, WE ANSWER. The first version of this walked its own list of
    # six positions and sent each one as soon as the operator pressed Enter.
    # That is wrong: ArduPilot prompts, then SAMPLES, then prompts again, and a
    # position command that arrives while it is still sampling the previous one
    # aborts the whole calibration - which is exactly what happened on 20 Oct,
    # and a failed calibration zeroes the offsets on its way out. So the loop
    # now waits for the FC's own "Place vehicle ..." line and answers that.
    WORDS = [("level", 1, "on its legs, exactly as it sits"),
             ("left", 2, "rolled 90 degrees onto its LEFT side"),
             ("right", 3, "rolled 90 degrees onto its RIGHT side"),
             ("nose down", 4, "pitched nose-down toward the ground"),
             ("nosedown", 4, "pitched nose-down toward the ground"),
             ("nose up", 5, "pitched nose-up toward the sky"),
             ("noseup", 5, "pitched nose-up toward the sky"),
             ("back", 6, "upside down, on its back")]
    done = 0
    deadline = time.time() + 600
    while time.time() < deadline:
        msg = fc.m.recv_match(type="STATUSTEXT", blocking=True, timeout=1)
        if msg is None:
            fc.beat()
            continue
        txt = msg.text.strip()
        if txt.startswith("SRC=") or "VPS" in txt or "PreArm" in txt:
            continue
        low = txt.lower()
        if "fail" in low:
            print(f"      FC: {txt}")
            print("\n  *** FAILED. ArduPilot zeroes the accel offsets when a "
                  "calibration fails,\n      so the aircraft now has NO accel "
                  "calibration and will refuse to arm.\n      Put the old one "
                  "back from\n      DRONE_fc_logs_backup/2026-10-20/params/"
                  "accel_cal_before.txt\n      before flying, then try again "
                  "holding each position dead still. ***")
            return 1
        if "success" in low or "calibration successful" in low:
            print(f"      FC: {txt}")
            break
        if "place" in low or "vehicle" in low:
            code = how = None
            for w, c, h in WORDS:
                if w in low:
                    code, how = c, h
                    break
            if code is None:
                print(f"      FC: {txt}   (position not recognised - skipping)")
                continue
            done += 1
            print(f"\n  --- {done}/6  the FC asks for: {txt}")
            print(f"      hold it {how}, and keep it STILL")
            input("      press Enter once it is in position and steady ")
            fc.m.mav.command_long_send(
                fc.m.target_system, fc.m.target_component,
                mavutil.mavlink.MAV_CMD_ACCELCAL_VEHICLE_POS, 0,
                code, 0, 0, 0, 0, 0, 0)
            print("      sent - hold it there until the FC asks for the next one")

    # The loop above already saw "Calibration successful" - that is how it
    # left. Do not go looking for it a second time: the first version of this
    # did, found nothing because the message had already been consumed, and
    # printed "No success message was seen" over a calibration that had in
    # fact worked. A tool that misreports success is a tool nobody can use.
    ok = True
    print("\n  the FC will also ask for a reboot; that is normal.")
    fc.drain_text(6.0, want=["reboot"])

    time.sleep(2.0)
    show(fc, "after:")
    new = fc.get("INS_ACCSCAL_X")
    print()
    if ok and new is not None and new != 1.0:
        print("  DONE - the scales are no longer exactly 1.0, which is what a "
              "6-position\n  calibration looks like. REBOOT the flight "
              "controller before flying.")
        return 0
    if ok:
        print("  The FC said it succeeded, but the scale is still exactly 1.0. "
              "Check it\n  in Mission Planner before trusting it.")
        return 1
    print("  No success message was seen. The old calibration is still in "
          "place unless\n  the values above changed. Check them before flying.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
