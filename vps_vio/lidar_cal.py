#!/usr/bin/env python3
"""Calibrate the analog rangefinder against known heights. GROUND ONLY.

The analog lidar reads distance = (voltage - offset) * scaling, so a wrong
scaling makes the aircraft believe it is at the wrong height. Before the wire
was resoldered this one read 296 cm at a true 110 cm — 2.7x out — which is why
it was taken out of the height solution entirely.

Method: hold the drone at two or more measured heights, record what the lidar
reports, and fit a straight line. Two points give both scaling and offset; one
point only gives scaling and assumes the offset is right.

Safety: refuses to run while armed. It temporarily enables the rangefinder to
read it, and restores whatever RNGFND1_TYPE was set before it started.

    python3 lidar_cal.py            # interactive
    python3 lidar_cal.py --apply    # also write the computed scaling
"""
from __future__ import annotations
import argparse, sys, time
import numpy as np
from pymavlink import mavutil

# The ROUTER, not the serial device. This file used to open /dev/ttyTHS1
# directly, which is the one thing the project forbids: drone-router is the
# sole owner of that port and a second opener splits the stream with it, so
# running this while the services were up corrupted both sides. Found 20 Oct.
PORT, BAUD = "udpout:127.0.0.1:14650", 115200


class FC:
    def __init__(self):
        self.m = mavutil.mavlink_connection(PORT, baud=BAUD, source_system=200)
        self.m.mav.heartbeat_send(6, 8, 0, 0, 0)
        if self.m.wait_heartbeat(timeout=15) is None:
            sys.exit("no heartbeat from the flight controller")
        if bool(self.m.motors_armed()):
            sys.exit("*** ARMED — ABORT. Calibrate on the ground, disarmed. ***")
        for sid in (mavutil.mavlink.MAV_DATA_STREAM_ALL,):
            self.m.mav.request_data_stream_send(self.m.target_system,
                                                self.m.target_component, sid, 5, 1)

    def get(self, name, timeout=3.0):
        self.m.mav.param_request_read_send(self.m.target_system,
                                           self.m.target_component, name.encode(), -1)
        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = self.m.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
            if msg and msg.param_id == name:
                return msg.param_value
        return None

    def set(self, name, val):
        self.m.mav.param_set_send(self.m.target_system, self.m.target_component,
                                  name.encode(), float(val),
                                  mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        time.sleep(0.4)
        return self.get(name)

    def read_distance(self, seconds=3.0):
        """Median lidar distance in metres over a few seconds, plus spread."""
        vals = []
        t0 = time.time()
        while time.time() - t0 < seconds:
            msg = self.m.recv_match(type=["DISTANCE_SENSOR", "RANGEFINDER"],
                                    blocking=True, timeout=1)
            if msg is None:
                continue
            if msg.get_type() == "RANGEFINDER":
                vals.append(msg.distance)
            else:
                vals.append(msg.current_distance / 100.0)
        if not vals:
            return None, None, 0
        a = np.array(vals)
        return float(np.median(a)), float(a.std()), len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the computed scaling")
    ap.add_argument("--keep-enabled", action="store_true",
                    help="leave the rangefinder enabled afterwards")
    a = ap.parse_args()

    fc = FC()
    print("DISARMED confirmed\n")
    orig_type = fc.get("RNGFND1_TYPE")
    scaling = fc.get("RNGFND1_SCALING")
    offset = fc.get("RNGFND1_OFFSET")
    gndclr = fc.get("RNGFND1_GNDCLR")
    print(f"  RNGFND1_TYPE    = {orig_type}")
    print(f"  RNGFND1_SCALING = {scaling}")
    print(f"  RNGFND1_OFFSET  = {offset}")
    print(f"  RNGFND1_GNDCLR  = {gndclr}\n")

    if orig_type is not None and abs(orig_type) < 0.5:
        print("rangefinder is disabled — enabling temporarily for the measurement")
        if fc.set("RNGFND1_TYPE", 1.0) is None:
            sys.exit("could not enable the rangefinder")
        time.sleep(2.0)

    pts = []
    try:
        print("Hold the drone at a MEASURED height, type the height in cm, press Enter.")
        print("Do at least two clearly different heights (e.g. 50 and 200 cm).")
        print("Blank line when done.\n")
        while True:
            s = input("true height (cm), blank to finish: ").strip()
            if not s:
                break
            try:
                truth_m = float(s) / 100.0
            except ValueError:
                print("  a number please"); continue
            d, sd, n = fc.read_distance(3.0)
            if d is None:
                print("  no rangefinder data — check the wiring"); continue
            print(f"  true {truth_m*100:6.1f} cm   lidar {d*100:7.1f} cm  "
                  f"(sd {sd*100:.1f} cm over {n} samples)  ratio {d/max(truth_m,1e-6):.3f}")
            pts.append((truth_m, d))
    except (EOFError, KeyboardInterrupt):
        print()

    if len(pts) < 1:
        print("no measurements taken")
    else:
        truth = np.array([p[0] for p in pts])
        meas = np.array([p[1] for p in pts])
        print(f"\n--- {len(pts)} point(s) ---")
        if len(pts) >= 2:
            # meas = m*truth + c  ->  we want the reading to equal the truth
            m_fit, c_fit = np.polyfit(truth, meas, 1)
            print(f"  fit: lidar = {m_fit:.4f} * true + {c_fit*100:.1f} cm")
            print(f"  slope error {m_fit:.3f}x, offset error {c_fit*100:.1f} cm")
            new_scaling = (scaling / m_fit) if scaling else None
            resid = meas - (m_fit * truth + c_fit)
            print(f"  residual: max {np.abs(resid).max()*100:.1f} cm")
        else:
            m_fit = meas[0] / truth[0]
            print(f"  single point: reading is {m_fit:.3f}x the truth")
            new_scaling = (scaling / m_fit) if scaling else None
        if new_scaling is not None:
            print(f"\n  RNGFND1_SCALING  {scaling:.3f}  ->  {new_scaling:.3f}")
            if a.apply:
                got = fc.set("RNGFND1_SCALING", new_scaling)
                print(f"  written, reads back {got}")
                print("  re-run without --apply to verify the corrected readings")
            else:
                print("  (not written — re-run with --apply to write it)")

    if not a.keep_enabled and orig_type is not None and abs(orig_type) < 0.5:
        fc.set("RNGFND1_TYPE", 0.0)
        print("\nrangefinder disabled again (as it was found)")
        print("when it is trusted: RNGFND1_TYPE=1, EK3_RNG_USE_HGT=70, SURFTRAK_MODE=1")


if __name__ == "__main__":
    main()
