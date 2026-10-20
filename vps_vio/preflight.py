#!/usr/bin/env python3
"""One command, before the props go on: is everything that should record, recording?

Why this exists. On 20 Oct the aircraft flew, and afterwards:

  * there were no images, because the service never passed --save-frames
  * two CSVs had NUL tails, because nothing had been fsynced
  * it would not auto-disarm, because the lidar's zero had drifted to 2.53 m
  * the VPS origin was 150 m from the aircraft, because SCR_USER was not
    retyped after driving to the launch spot

Every one of those was invisible until the flight was over and the data was
being read. None of them needed a flight to find. So this checks them on the
ground, in one pass, and says GO or NO-GO.

It writes nothing and arms nothing. The one thing it will tell you to fix by
hand is the lidar zero, because that needs `lidar_zero.py --apply` and a
deliberate decision that the aircraft is standing on its legs.

    python3 preflight.py
    python3 preflight.py --port udpout:127.0.0.1:14650

Exit code 0 = GO, 1 = something will cost you the flight's data or the disarm.
"""
from __future__ import annotations
import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
import time

from pymavlink import mavutil

OUT = "/data/DRONE/flight_logs/vps"
ALIVE = "/run/drone/vps.alive"
SERVICES = ("drone-router", "drone-vps", "drone-rc-switch")
# The land detector needs the rangefinder under 2 m in absolute terms before it
# will declare the aircraft landed. 1.5 is that threshold with margin for the
# sensor's own drift during one flight.
LIDAR_MAX_M = 1.5
# Expected after 20 Oct. A power cycle keeps these (they are in flash), but a
# parameter reset or someone else's session does not.
EXPECT = {
    "RNGFND1_RMETRIC": (0.0, "a current loop does not scale with the 5 V rail"),
    "INS_HNTCH_FREQ": (47.0, "the measured prop fundamental"),
    "INS_HNTCH_REF": (0.331, "the measured hover throttle"),
    "INS_HNTCH_BW": (24.0, "FREQ/2"),
    "MOT_THST_HOVER": (0.33, "measured"),
    "ATC_RAT_RLL_I": (0.07, "= ATC_RAT_RLL_P"),
    "ATC_RAT_PIT_I": (0.09, "= ATC_RAT_PIT_P"),
    "GPS2_TYPE": (14.0, "MAV - our injection arrives as GPS2"),
    "EK3_SRC1_POSXY": (3.0, "GPS"),
    "DISARM_DELAY": (2.0, "seconds after landing"),
}

blockers, warnings = [], []


def ok(name, detail=""):
    print(f"  \033[32mOK\033[0m    {name}" + (f"   {detail}" if detail else ""))


def bad(name, detail=""):
    print(f"  \033[31mNO-GO\033[0m {name}" + (f"   {detail}" if detail else ""))
    blockers.append(name)


def warn(name, detail=""):
    print(f"  \033[33mWARN\033[0m  {name}" + (f"   {detail}" if detail else ""))
    warnings.append(name)


def check(cond, name, detail="", soft=False):
    (ok if cond else (warn if soft else bad))(name, detail)
    return cond


# ------------------------------------------------------------------ companion

def companion():
    print("\nTHE COMPANION")
    for s in SERVICES:
        r = subprocess.run(["systemctl", "is-active", s],
                           capture_output=True, text=True)
        check(r.stdout.strip() == "active", f"{s} running", r.stdout.strip())

    try:
        age = time.time() - os.path.getmtime(ALIVE)
        txt = open(ALIVE).read().strip()
        check(age < 5, "the node's loop is turning",
              f"{age:.1f}s old: {txt[:60]}")
    except OSError:
        bad("the node's loop is turning", f"no {ALIVE}")
        txt = ""

    runs = sorted(glob.glob(os.path.join(OUT, "vps_[0-9]*")))
    cur = runs[-1] if runs else None
    if not cur:
        bad("this run has somewhere to record", f"nothing in {OUT}")
        return
    if os.path.isdir(cur):
        ok("the run has a FOLDER, so frames can go in it",
           os.path.basename(cur))
        est = [p for p in glob.glob(os.path.join(cur, "vps_*.csv"))
               if not p.endswith("_imu.csv")]
        check(bool(est), "the estimate CSV exists")
        check(os.path.exists(os.path.join(cur, "camera.txt")),
              "camera.txt is written", "fx/fy/cx/cy/crop_left for offline matching")
        imu = glob.glob(os.path.join(cur, "*_imu.csv"))
        if check(bool(imu), "the camera IMU log exists"):
            n = sum(1 for _ in open(imu[0])) - 1
            check(n > 50, "the camera IMU is actually reporting",
                  f"{n} samples so far")
        njpg = len(glob.glob(os.path.join(cur, "*.jpg")))
    else:
        bad("the run has a FOLDER, so frames can go in it",
            f"{os.path.basename(cur)} is a bare CSV - the unit is not passing "
            f"--save-frames")
        est, njpg = [cur], 0

    # fsync: the newest row on disk must be seconds old, not minutes
    if est:
        try:
            last = open(est[0]).read().strip().rsplit("\n", 1)[-1].split(",")[0]
            lag = time.time() - os.path.getmtime(est[0])
            check(lag < 4, "rows are reaching the DISK, not just the buffer",
                  f"file touched {lag:.1f}s ago, last row t={last}s")
        except (OSError, IndexError):
            warn("rows are reaching the disk", "could not read the CSV tail")

    free = shutil.disk_usage("/data").free / 2**30
    check(free > 5, "disk space", f"{free:.0f} GB free "
          f"(~1.5 GB an hour of frames at 4 Hz)")
    return njpg


# ------------------------------------------------------------------------ FC

class FC:
    def __init__(self, port):
        self.m = mavutil.mavlink_connection(port, source_system=205)
        self.m.mav.heartbeat_send(6, 8, 0, 0, 0)
        self.up = self.m.wait_heartbeat(timeout=12) is not None
        if self.up:
            self.m.mav.request_data_stream_send(
                self.m.target_system, self.m.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1)

    def beat(self):
        self.m.mav.heartbeat_send(6, 8, 0, 0, 0)

    def get(self, name, timeout=5.0):
        self.m.mav.param_request_read_send(
            self.m.target_system, self.m.target_component, name.encode(), -1)
        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = self.m.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
            if msg is not None and msg.param_id.strip("\x00") == name:
                return float(msg.param_value)
            self.beat()
        return None

    def gather(self, seconds=8.0):
        got = {}
        t0 = time.time()
        while time.time() - t0 < seconds:
            msg = self.m.recv_match(blocking=True, timeout=1)
            if msg is None:
                self.beat()
                continue
            t = msg.get_type()
            if t == "RANGEFINDER":
                got.setdefault("rng", []).append(msg.distance)
            elif t == "DISTANCE_SENSOR":
                # This FC sends DISTANCE_SENSOR and NOT RANGEFINDER. Listening
                # for only one of them reported "lidar is reporting: NO-GO" on
                # a perfectly healthy sensor - a check that cries wolf is worse
                # than no check, because the next real NO-GO gets ignored.
                got.setdefault("rng", []).append(msg.current_distance / 100.0)
            elif t == "GPS_RAW_INT":
                got["sats"] = msg.satellites_visible
                got["hdop"] = msg.eph / 100.0
                got["fix"] = msg.fix_type
            elif t == "SYS_STATUS":
                got["volt"] = msg.voltage_battery / 1000.0
            elif t == "RC_CHANNELS":
                got["ch6"] = msg.chan6_raw
            elif t == "STATUSTEXT":
                got.setdefault("text", []).append(msg.text)
            elif t == "VIBRATION":
                got["vibe"] = (msg.vibration_x, msg.vibration_y, msg.vibration_z)
                got["clip"] = (msg.clipping_0, msg.clipping_1, msg.clipping_2)
        return got


def flight_controller(port):
    print("\nTHE FLIGHT CONTROLLER")
    fc = FC(port)
    if not check(fc.up, "the FC answers through the router",
                 "nothing else can be checked without it"):
        return None
    armed = bool(fc.m.motors_armed())
    if armed:
        bad("DISARMED", "*** the aircraft is ARMED - stop here ***")
        return fc
    ok("disarmed")

    g = fc.gather()

    # The one that cost 20 Oct its disarm.
    rng = g.get("rng") or []
    if rng:
        mean = sum(rng) / len(rng)
        if mean < LIDAR_MAX_M:
            ok("lidar zero - the aircraft will be able to auto-disarm",
               f"{mean:.2f} m on the ground (needs < 2.00)")
        else:
            bad("lidar zero - IT WILL NOT AUTO-DISARM",
                f"{mean:.2f} m on the ground. Run: python3 "
                f"vps_vio/lidar_zero.py --apply")
        # Is it STILL moving? Zeroing a sensor that is mid-warm-up writes an
        # offset that is wrong by the time the aircraft takes off: measured
        # 20 Oct, the reading walked 0.83 -> 0.66 m in about fifteen seconds
        # just after a power cycle.
        if len(rng) >= 12:
            k = len(rng) // 3
            drift = (sum(rng[-k:]) / k) - (sum(rng[:k]) / k)
            check(abs(drift) < 0.10, "lidar is settled, not still warming up",
                  f"moved {drift:+.3f} m across the sample window" +
                  ("" if abs(drift) < 0.10 else
                   " - give it a few minutes of power, THEN re-zero"),
                  soft=True)
        # Below RNGFND1_MIN the FC calls it out of range and stops using it for
        # height. That still lets the land detector pass, but EK3 loses its
        # height source near the ground.
        rmin = fc.get("RNGFND1_MIN") or 0.1
        check(mean > rmin + 0.1, "lidar is inside its own usable range",
              f"{mean:.2f} m against RNGFND1_MIN {rmin:.2f}" +
              ("" if mean > rmin + 0.1 else
               " - too low; EK3 will drop it for height near the ground"),
              soft=True)
    else:
        bad("lidar is reporting",
            "no RANGEFINDER or DISTANCE_SENSOR messages in the window")

    # Parameters survive a power cycle, but check rather than assume.
    for name, (want, why) in EXPECT.items():
        v = fc.get(name)
        if v is None:
            warn(f"{name} readable", "no answer")
        else:
            check(abs(v - want) < 2e-3, f"{name} = {want:g}",
                  f"{why}" if abs(v - want) < 2e-3 else f"found {v:g}, not {want:g}")

    if "sats" in g:
        good_gps = g["sats"] >= 10 and g.get("hdop", 99) < 1.5
        check(good_gps, "GPS",
              f"{g['sats']} sats, HDop {g.get('hdop', 0):.2f}, fix type "
              f"{g.get('fix')}" + ("" if good_gps else
              " - LOITER needs a fix (EK3_SRC1_POSXY=3). Indoors this is "
              "normal; it must be good before take-off. Note ARMING_SKIPCHK "
              "skips the GPS-lock pre-arm check, so the FC will NOT stop you"),
              soft=True)
    if "volt" in g:
        arm_v = fc.get("BATT_ARM_VOLT") or 0.0
        if g["volt"] < arm_v:
            bad("battery - ARMING WILL BE REFUSED",
                f"{g['volt']:.2f} V is under BATT_ARM_VOLT {arm_v:.2f}. "
                f"Fit a charged pack.")
        else:
            check(g["volt"] > arm_v + 0.5, "battery at rest",
                  f"{g['volt']:.2f} V, arm threshold {arm_v:.2f} (it sags "
                  f"~1.7 V in hover and warns at 22.8)", soft=True)
    if "ch6" in g:
        c = g["ch6"]
        where = ("real GPS" if c < 1200 else "centre - holds whatever is set"
                 if c < 1700 else "VPS")
        ok("RC channel 6", f"{c} = {where}")
    if "vibe" in g:
        vx, vy, vz = g["vibe"]
        check(max(g["clip"]) == 0, "no accelerometer clipping", str(g["clip"]),
              soft=True)
        ok("vibration at rest", f"X {vx:.1f} Y {vy:.1f} Z {vz:.1f} m/s2 "
                                f"(in hover on 20 Oct: 0.8 / 1.6 / 2.3)")

    # The origin, and whether the sheet covers it.
    lat, lon = fc.get("SCR_USER1"), fc.get("SCR_USER2")
    if lat and lon:
        print(f"\nTHE ORIGIN  SCR_USER1/2 = {lat:.6f}, {lon:.6f}")
        covered = sheet_covers(lon, lat)
        if covered is None:
            warn("the sheet covers the take-off point", "could not read bhh.tif")
        else:
            check(covered, "the sheet covers the take-off point",
                  "bhh.tif" if covered else
                  "OFF-SHEET - there will be no map fixes for the whole flight")
        # Is the aircraft actually AT the typed point? 20 Oct: 150 m away.
        d = distance_to_gps(fc, lat, lon)
        if d is None:
            warn("the aircraft is at the typed origin", "no GPS position yet")
        else:
            check(d < 30, "the aircraft is at the typed origin",
                  f"{d:.0f} m from SCR_USER1/2" +
                  ("" if d < 30 else " - RETYPE IT at this take-off point"))
    else:
        bad("SCR_USER1/2 is set", "the node has no origin to work from")
    return fc


def sheet_covers(lon, lat, path="/data/DRONE/bhh.tif"):
    try:
        from osgeo import gdal
        ds = gdal.Open(path)
        gt = ds.GetGeoTransform()
        w, h = ds.RasterXSize, ds.RasterYSize
        x0, y0 = gt[0], gt[3]
        x1, y1 = x0 + gt[1] * w, y0 + gt[5] * h
        return min(x0, x1) <= lon <= max(x0, x1) and min(y0, y1) <= lat <= max(y0, y1)
    except Exception:
        return None


def distance_to_gps(fc, lat, lon):
    t0 = time.time()
    while time.time() - t0 < 6:
        msg = fc.m.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1)
        if msg is None:
            fc.beat()
            continue
        if msg.lat == 0 and msg.lon == 0:
            continue
        import math
        dn = (msg.lat / 1e7 - lat) * 111320.0
        de = (msg.lon / 1e7 - lon) * 111320.0 * math.cos(math.radians(lat))
        return math.hypot(dn, de)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="udpout:127.0.0.1:14650")
    a = ap.parse_args()

    print("=" * 68)
    print("PRE-FLIGHT - nothing here is written or armed")
    print("=" * 68)
    njpg = companion()
    flight_controller(a.port)

    print("\nWHAT WILL BE RECORDED")
    print(f"  images    4 Hz, ONLY while armed  (on the ground now: "
          f"{njpg if njpg is not None else '?'} saved so far this run)")
    print("  estimate  40 columns at 30 Hz, flushed to disk every second")
    print("  camera    every IMU sample, 100 Hz gyro + 100 Hz accel")
    print("  the FC's own dataflash, as always")
    print("\n  After the flight: copy the new .BIN off the SD card, then")
    print("    python3 vps_vio/hover_balance.py <log>.BIN      # did the trim improve")
    print("    python3 vps_vio/match_flight.py <run folder>    # needs 60 m+ to have fixes")

    print("\n" + "=" * 68)
    if blockers:
        print(f"NO-GO - {len(blockers)} thing(s) will cost you the flight:")
        for b in blockers:
            print(f"   - {b}")
    else:
        print("GO" + (f" - with {len(warnings)} warning(s): {', '.join(warnings)}"
                      if warnings else " - everything checks out"))
    print("=" * 68)
    return 1 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())
