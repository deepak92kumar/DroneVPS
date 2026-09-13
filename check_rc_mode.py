#!/usr/bin/env python3
"""
Live RC Flight Mode & Switch Verification Tool
Turn ON your RC transmitter and flip the flight mode switch.
This tool shows exactly which mode the flight controller is receiving!
"""

import sys
import os
import time
sys.path.insert(0, '/home/abc/Desktop/DRONE')
from cuav_mavlink_bridge import CuavMavlinkClient

def main():
    print("=================================================================")
    print("      LIVE RC FLIGHT MODE & SWITCH VERIFIER")
    print("=================================================================")
    print("[*] Connecting to CUAV X7 Pro...")
    client = CuavMavlinkClient()
    time.sleep(2.0)

    if not client.is_connected:
        print("[-] CUAV X7 Pro not connected. Check USB cable.")
        return

    print(f"[+] Connected on {client.port}!")
    print(f"[*] Current Flight Mode: {client.flight_mode}")
    print("[*] Turn ON your RC transmitter and flip the Flight Mode switch.")
    print("[*] Watching mode changes in real-time... (Press Ctrl+C to exit)\n")

    last_mode = client.flight_mode
    try:
        while True:
            curr_mode = client.flight_mode
            if curr_mode != last_mode:
                badge = "🟢 POSITION HOLD ACTIVE (CAMERA ON)" if curr_mode == "LOITER" else ("🔴 MANUAL (NO CAMERA BRAKE)" if curr_mode == "STABILIZE" else "🟡 ALTITUDE HOLD")
                print(f"  [SWITCH CHANGED] New Mode: {curr_mode:<12} --> {badge}")
                last_mode = curr_mode
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[*] Tool closed.")
    finally:
        client.close()

if __name__ == "__main__":
    main()
