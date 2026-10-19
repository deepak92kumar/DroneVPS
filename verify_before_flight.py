#!/usr/bin/env python3
"""
Pre-Flight Ground & Bench Verification Dashboard
Antigravity Autonomous Drone Navigation Systems

Run this tool to verify all optical flow, tilt immunity, and EKF stability fixes
BEFORE putting propellers on the drone.
"""

import sys
import os
import time
import glob
import re

LOG_DIR = "/data/DRONE/flight_logs"

def get_latest_log():
    logs = glob.glob(os.path.join(LOG_DIR, "flight_*.log"))
    if not logs:
        return None
    logs.sort(key=os.path.getmtime, reverse=True)
    return logs[0]

def colorize(text, color):
    codes = {
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "cyan": "\033[96m",
        "bold": "\033[1m",
        "reset": "\033[0m"
    }
    return f"{codes.get(color, '')}{text}{codes['reset']}"

def clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()

def main():
    latest_log = get_latest_log()
    if not latest_log:
        print("[!] No flight log found in", LOG_DIR)
        return

    print(f"[*] Connecting to live VIO telemetry feed: {latest_log}...")
    time.sleep(0.5)

    line_re = re.compile(
        r"Pos:\s*N:([+-]?\d+\.\d+)m\s*E:([+-]?\d+\.\d+)m\s*Z:([+-]?\d+\.\d+)m\s*\|\s*"
        r"\[(ARUCO[^\]]*)\]\s*\|\s*"
        r"TF03_H=([0-9\.]+)m.*?"
        r"CUAV:\s*([^\s]+)\s*\(([^)]+)\)\s*\|\s*"
        r"PosVar:\s*([0-9\.]+)\s*\|\s*"
        r"VelVar:\s*([0-9\.]+)"
    )

    with open(latest_log, "r") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 2000))
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue

            match = line_re.search(line)
            if match:
                pos_n = float(match.group(1))
                pos_e = float(match.group(2))
                pos_z = float(match.group(3))
                aruco_str = match.group(4)
                tf03_h = float(match.group(5))
                cuav_port = match.group(6)
                flight_mode = match.group(7)
                pos_var = float(match.group(8))
                vel_var = float(match.group(9))

                if pos_var < 0.05:
                    var_status = colorize(f"EXCELLENT ({pos_var:.3f})", "green")
                elif pos_var < 0.20:
                    var_status = colorize(f"ACCEPTABLE ({pos_var:.3f})", "yellow")
                else:
                    var_status = colorize(f"POOR / DRIFTING ({pos_var:.3f})", "red")

                if "LOCKED" in aruco_str:
                    aruco_disp = colorize(f"PAD LOCKED ({aruco_str})", "green")
                else:
                    aruco_disp = colorize("SEARCHING (Optical Flow Active)", "cyan")

                clear_screen()
                print("=" * 72)
                print(colorize("      PRE-FLIGHT BENCH VERIFICATION DASHBOARD (CUAV + OAK-D)", "bold"))
                print("=" * 72)
                print(f" Flight Controller : {cuav_port} | Mode: {colorize(flight_mode, 'bold')}")
                print(f" Autopilot EKF3    : PosVar: {var_status} | VelVar: {vel_var:.3f}")
                print(f" Target Pad Status : {aruco_disp}")
                print("-" * 72)
                print(colorize(" [LIVE METRICS]", "bold"))
                print(f"  North (Forward)  : {pos_n:+0.2f} m")
                print(f"  East  (Right)    : {pos_e:+0.2f} m")
                print(f"  Altitude (Z AGL) : {pos_z:0.2f} m  (LiDAR TF03: {tf03_h:0.2f} m)")
                print("-" * 72)
                print(colorize(" [4 BENCH TESTS - VERIFY WITH YOUR OWN EYES BEFORE FLIGHT]:", "bold"))
                print("  1. " + colorize("TILT IMMUNITY TEST", "yellow") + ": Drone ko haath mein pakad kar 15°-20° tilt karo.")
                print("     -> North/East bhagna nahi chahiye! PosVar green (0.000) rehna chahiye.")
                print("  2. " + colorize("TRANSLATION TEST", "yellow") + ": Drone ko haath se 1 meter aage le jao.")
                print("     -> North +1.0m badhna chahiye. Right le jao to East +1.0m badhna chahiye.")
                print("  3. " + colorize("2M+ HEIGHT TEST", "yellow") + ": Drone ko 2 meter se upar uthao.")
                print("     -> TF03 aur VIO Z bina jhatke ke smoothly sath chalne chahiye.")
                print("  4. " + colorize("ARUCO HANDOVER TEST", "yellow") + ": Drone ko pad ke upar se hatakar door le jao.")
                print("     -> State 'SEARCHING' par aayegi par coordinate jump nahi karega!")
                print("=" * 72)
                print(" Press Ctrl+C to exit dashboard.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[*] Dashboard stopped.")
