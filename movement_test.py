#!/usr/bin/env python3
"""
Interactive Movement & Odometer Testing Tool
Shows real-time physical displacement in Centimeters (cm) and Meters (m)
as you move the drone by hand.
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

    log_pattern = re.compile(
        r"Pos:\s*N:([+-]?\d+\.\d+)m\s*E:([+-]?\d+\.\d+)m\s*Z:([+-]?\d+\.\d+)m\s*Yaw:([+-]?\d+\.\d+)°.*?"
        r"CUAV:\s*(\S+)\s*\((.*?)\).*?"
        r"PosVar:\s*([0-9\.]+)"
    )

    print(f"[*] Connecting to live stream: {os.path.basename(latest_log)}")
    time.sleep(1)

    start_n = None
    start_e = None
    start_z = None

    with open(latest_log, "r") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.08)
                continue

            match = log_pattern.search(line)
            if match:
                north = float(match.group(1))
                east = float(match.group(2))
                alt = float(match.group(3))
                yaw = float(match.group(4))
                port = match.group(5)
                flight_mode = match.group(6)
                posvar = float(match.group(7))

                if start_n is None:
                    start_n = north
                    start_e = east
                    start_z = alt

                # Displacements from starting spot
                dn_cm = (north - start_n) * 100.0
                de_cm = (east - start_e) * 100.0
                dz_cm = (alt - start_z) * 100.0
                total_disp_cm = (dn_cm**2 + de_cm**2)**0.5

                # Direction hint
                dir_n = "NORTH (Aage)" if dn_cm > 5 else ("SOUTH (Piche)" if dn_cm < -5 else "CENTER")
                dir_e = "EAST (Right)" if de_cm > 5 else ("WEST (Left)" if de_cm < -5 else "CENTER")

                clear_screen()
                print("=================================================================")
                print("      DRONE LIVE MOVEMENT & ODOMETER VERIFICATION TOOL")
                print("=================================================================")
                print(f" Flight Mode : {flight_mode}  |  CUAV Port: {port}")
                print(f" Compass Yaw : {yaw:05.1f}°  |  EKF Var: {posvar:.3f} (LOCKED)")
                print("-----------------------------------------------------------------")
                print(f" [CURRENT SPOT DISPLACEMENT FROM TEST START]")
                print(f"   * North / South Movement : {dn_cm:+06.1f} cm   --> {dir_n}")
                print(f"   * East  / West  Movement : {de_cm:+06.1f} cm   --> {dir_e}")
                print(f"   * Height / Altitude      : {alt:04.2f} m ({dz_cm:+05.1f} cm delta)")
                print(f"   * Total Ground Distance  : {total_disp_cm:05.1f} cm ({total_disp_cm/100.0:04.2f} m)")
                print("-----------------------------------------------------------------")
                print(f" Raw World Coordinates      : N: {north:+05.2f}m , E: {east:+05.2f}m")
                print("=================================================================")
                print(">>> STEP-BY-STEP MOVEMENT TEST:")
                print(" 1. Drone ko haath me pakad kar 50 cm AAGE (North) le jayein:")
                print("    -> North Movement +50 cm badhna chahiye.")
                print(" 2. Wapas starting spot par layein:")
                print("    -> Displacement 0 cm ke paas wapas aana chahiye.")
                print(" 3. Drone ko RIGHT (East) le jayein:")
                print("    -> East Movement +50 cm badhna chahiye.")
                print(" 4. Drone ko UP (upar uthayein):")
                print("    -> Height badhegi, lekin North/East static rahenge.")
                print("\n[Press Ctrl+C to exit]")
                sys.stdout.flush()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[*] Movement test closed.")
