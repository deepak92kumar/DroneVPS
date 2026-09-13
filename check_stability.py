#!/usr/bin/env python3
"""
Live VIO Stability & Hand-Carry Testing Monitor
Run this script to watch real-time Earth NED coordinates, altitude, compass heading,
and EKF status while moving the drone by hand.
"""

import sys
import os
import time
import glob
import re

LOG_DIR = "/home/abc/Desktop/DRONE/flight_logs"

def get_latest_log():
    logs = glob.glob(os.path.join(LOG_DIR, "flight_*.log"))
    if not logs:
        return None
    logs.sort(key=os.path.getmtime, reverse=True)
    return logs[0]

def clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()

def main():
    latest_log = get_latest_log()
    if not latest_log:
        print("[!] No flight log found in", LOG_DIR)
        return

    print(f"[*] Monitoring live telemetry from: {latest_log}")
    print("[*] Press Ctrl+C to stop.")
    time.sleep(1)

    log_pattern = re.compile(
        r"Pos:\s*N:([+-]?\d+\.\d+)m\s*E:([+-]?\d+\.\d+)m\s*Z:([+-]?\d+\.\d+)m\s*Yaw:([+-]?\d+\.\d+)°.*?"
        r"PosVar:\s*([0-9\.]+)"
    )

    with open(latest_log, "r") as f:
        # Go to near end of file
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue

            match = log_pattern.search(line)
            if match:
                north = float(match.group(1))
                east = float(match.group(2))
                alt = float(match.group(3))
                yaw = float(match.group(4))
                posvar = float(match.group(5))

                status_badge = "PERFECT (LOCKED)" if posvar < 0.05 else ("CONVERGING" if posvar < 0.20 else "UNSTABLE")

                clear_screen()
                print("=================================================================")
                print("     AUTONOMOUS DRONE GNSS-DENIED VIO STABILITY LIVE MONITOR")
                print("=================================================================")
                print(f" Log File          : {os.path.basename(latest_log)}")
                print("-----------------------------------------------------------------")
                print(f" North Position (N): {north:+06.2f} m  (Forward/Backward Earth Axis)")
                print(f" East Position  (E): {east:+06.2f} m  (Right/Left Earth Axis)")
                print(f" Ground Clearance  : {alt:05.2f} m  (OAK-D Active IR Stereo Depth)")
                print(f" Compass Yaw       : {yaw:05.1f}°  (CUAV Calibrated Compass)")
                print("-----------------------------------------------------------------")
                print(f" Autopilot EKF Var : {posvar:.3f}   [{status_badge}]")
                print("=================================================================")
                print("\n>>> HOW TO TEST (HAND-CARRY BENCHTOP TEST):")
                print(" 1. STATIONARY CHECK: Drone table par rakhein. N aur E numbers static hone chahiye.")
                print(" 2. TILT CHECK      : Drone ko haath me utha kar aage-piche tilt karein.")
                print("                      Position drift nahi honi chahiye (Tilt derotation active).")
                print(" 3. HEADING CHECK   : Drone ko 90° ghuma kar aage badhayein.")
                print("                      Coordinates sahi Earth directions (N vs E) me badhenge.")
                print("\n[Press Ctrl+C to exit monitor]")
                sys.stdout.flush()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[*] Monitor closed.")
