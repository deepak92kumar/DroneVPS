#!/usr/bin/env python3
"""
Autonomous Drone Mode Switcher
==============================
Effortlessly switches CUAV X7 Pro between:
1. GPS / Hybrid Mode ('gps'): Physical GPS active, live Google Maps tracking, OAK-D Pro AI Anti-Jamming backup.
2. Pure VIO Mode ('vio'): Physical GPS disabled, indoor GNSS-denied, instant LOITER arming without GPS.
3. Set Home Position ('home lat lon [alt]'): Manually set EKF origin + ArduPilot home when GPS is absent.
   Example: python3 set_mode.py home 28.6139 77.2090 216.0
   Use this when GPS is disconnected and UniRC GCS map point is known.
"""

import sys
import time
from pymavlink import mavutil
from configure_cuav_full import find_cuav_port, set_parameter, read_parameter

def switch_to_gps(m):
    print("=" * 60)
    print(" [*] CONFIGURING DRONE FOR: GPS / HYBRID MODE (OUTDOOR)")
    print("=" * 60)
    
    # Enable physical GPS module
    set_parameter(m, 'GPS1_TYPE', 1.0)
    time.sleep(0.15)
    
    # Enable External Compass in GPS puck
    set_parameter(m, 'COMPASS_USE', 1.0)
    set_parameter(m, 'COMPASS_USE2', 1.0)
    set_parameter(m, 'COMPASS_PRIO1_ID', 658953.0) # External puck as priority 1
    time.sleep(0.15)
    
    # Configure Primary EKF source to GPS, Secondary to OAK-D Pro VIO
    set_parameter(m, 'EK3_SRC1_POSXY', 3.0) # Primary: GPS
    set_parameter(m, 'EK3_SRC1_VELXY', 3.0)
    set_parameter(m, 'EK3_SRC1_POSZ', 1.0)  # Baro
    set_parameter(m, 'EK3_SRC1_VELZ', 3.0)  # GPS climb rate
    set_parameter(m, 'EK3_SRC1_YAW', 1.0)   # Compass
    time.sleep(0.15)
    
    # Secondary Backup: OAK-D Pro VIO
    set_parameter(m, 'EK3_SRC2_POSXY', 6.0)
    set_parameter(m, 'EK3_SRC2_VELXY', 6.0)
    set_parameter(m, 'EK3_SRC2_POSZ', 1.0)
    set_parameter(m, 'EK3_SRC2_VELZ', 0.0)
    set_parameter(m, 'EK3_SRC2_YAW', 1.0)
    time.sleep(0.15)
    
    set_parameter(m, 'AHRS_GPS_USE', 1.0)
    set_parameter(m, 'EK3_SRC_OPTIONS', 0.0)
    set_parameter(m, 'ARMING_SKIPCHK', 2048.0) # Standard safety checks
    
    print("\n[+] SUCCESS: GPS / Hybrid Mode Configured!")
    print("    - Real-time Google Maps location: ACTIVE")
    print("    - OAK-D Pro AI Shield: ACTIVE as seamless backup if GPS is jammed")
    print("    - NOTE: If you just plugged the GPS cable, reboot CUAV to detect hardware.\n")

def switch_to_vio(m):
    print("=" * 60)
    print(" [*] CONFIGURING DRONE FOR: PURE VIO MODE (GNSS-DENIED INDOOR)")
    print("=" * 60)
    
    # Disable physical GPS check
    set_parameter(m, 'GPS1_TYPE', 0.0)
    time.sleep(0.15)
    
    # Disable external compass (use internal CUAV compass)
    set_parameter(m, 'COMPASS_USE', 0.0)
    set_parameter(m, 'COMPASS_USE2', 1.0)
    set_parameter(m, 'COMPASS_PRIO1_ID', 1114642.0)
    time.sleep(0.15)
    
    # Primary Source: OAK-D Pro VIO
    set_parameter(m, 'EK3_SRC1_POSXY', 6.0)
    set_parameter(m, 'EK3_SRC1_VELXY', 6.0) # ExternalNav Velocity Active (30Hz)
    set_parameter(m, 'EK3_SRC1_POSZ', 2.0)  # 2 = Rangefinder / TF03 LiDAR (calibrated ADC)
    set_parameter(m, 'EK3_SRC1_VELZ', 0.0)  # None
    set_parameter(m, 'EK3_SRC1_YAW', 1.0)   # Compass
    time.sleep(0.15)
    
    # Clear secondary sources
    set_parameter(m, 'EK3_SRC2_POSXY', 0.0)
    set_parameter(m, 'EK3_SRC2_VELXY', 0.0)
    set_parameter(m, 'EK3_SRC2_POSZ', 0.0)
    set_parameter(m, 'EK3_SRC2_VELZ', 0.0)
    time.sleep(0.15)
    
    set_parameter(m, 'AHRS_GPS_USE', 0.0)
    set_parameter(m, 'EK3_SRC_OPTIONS', 0.0)
    set_parameter(m, 'ARMING_SKIPCHK', 6156.0) # Bypass GPS prearm locks
    
    print("\n[+] SUCCESS: Pure VIO GNSS-Denied Mode Configured!")
    print("    - OAK-D Pro 3D Visual Odometry: PRIMARY SENSOR")
    print("    - Indoor/Tunnel/Bunker Freeze Hover: ACTIVE")
    print("    - No GPS Lock Needed to Arm.\n")

def set_home(m, lat, lon, alt_m=0.0):
    """
    GPS disconnect hone par manually EKF origin + ArduPilot home set karta hai.

    Kab use karein:
    - GPS module physically absent/disconnected hai
    - UniRC GCS map par point select karke home set karna ho
    - flight_daemon start se pehle correct NED origin chahiye

    Do messages bhejta hai:
      1. SET_GPS_GLOBAL_ORIGIN  — EKF3 NED frame ka (0,0,0) anchor
      2. MAV_CMD_DO_SET_HOME    — ArduPilot home (RTL / failsafe reference)
    """
    # If altitude is not explicitly provided (or 0.0), auto-read current Baro MSL altitude
    # This guarantees that Relative Altitude (AGL) starts at exactly 0.0m on the ground!
    if alt_m is None or float(alt_m) == 0.0:
        print("[*] Detecting ground elevation from FC Barometer...")
        t0 = time.time()
        while time.time() - t0 < 2.0:
            msg = m.recv_match(type='VFR_HUD', blocking=True, timeout=0.5)
            if msg and getattr(msg, 'alt', 0.0) != 0.0:
                alt_m = float(msg.alt)
                break
        if alt_m == 0.0:
            print("[!] Could not read Baro MSL altitude, defaulting to 0.0m")
        else:
            print(f"[+] Ground elevation auto-detected: {alt_m:.1f} m MSL (AGL will be 0.0m on ground)")

    print("=" * 60)
    print(f" [*] HOME POSITION SET (GPS-DENIED MODE)")
    print(f"     Lat : {lat:.7f} deg")
    print(f"     Lon : {lon:.7f} deg")
    print(f"     Alt : {alt_m:.1f} m MSL")
    print("=" * 60)

    target_sys  = m.target_system   if getattr(m, 'target_system',  0) > 0 else 1
    target_comp = m.target_component if getattr(m, 'target_component', 0) > 0 else 1

    # Step 1 — EKF3 NED origin anchor
    m.mav.set_gps_global_origin_send(
        target_sys,
        int(lat   * 1e7),
        int(lon   * 1e7),
        int(alt_m * 1000)   # mm
    )
    time.sleep(0.1)
    print("[+] Step 1: SET_GPS_GLOBAL_ORIGIN sent  -> EKF3 NED origin set")

    # Step 2 — ArduPilot home position (param1=0 = use lat/lon/alt fields)
    m.mav.command_long_send(
        target_sys, target_comp,
        mavutil.mavlink.MAV_CMD_DO_SET_HOME,
        0,
        0,                  # param1=0 : use specified coordinates
        0, 0, 0,
        float(lat),
        float(lon),
        float(alt_m)
    )
    time.sleep(0.15)

    ack = m.recv_match(type='COMMAND_ACK', blocking=True, timeout=2.0)
    if ack and getattr(ack, 'result', -1) == mavutil.mavlink.MAV_RESULT_ACCEPTED:
        print("[+] Step 2: MAV_CMD_DO_SET_HOME ACK'd  -> Home locked")
        print(f"\n[+] SUCCESS: Home = ({lat:.7f}, {lon:.7f}, {alt_m:.1f}m)")
        print("    VIO NED frame ka (0,0,0) yahi point hai.")
        print("    UniRC GCS map par drone is point ke relative position dikhega.\n")
    else:
        code = getattr(ack, 'result', 'timeout') if ack else 'timeout'
        print(f"[!] Step 2: DO_SET_HOME result: {code}")
        print("    Home set hua ho sakta hai — UniRC GCS map se verify karein.\n")


if __name__ == '__main__':
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else 'status'

    m, port = find_cuav_port()
    if not m:
        print("[-] Error: CUAV Autopilot not found on serial ports.")
        sys.exit(1)

    print(f"[+] Connected to CUAV on {port}")

    if mode in ['gps', 'outdoor', 'hybrid']:
        switch_to_gps(m)

    elif mode in ['vio', 'indoor', 'nogps']:
        switch_to_vio(m)

    elif mode == 'home':
        # Usage: python3 set_mode.py home <lat> <lon> [alt_m]
        # Example: python3 set_mode.py home 30.895560 74.613858
        if len(sys.argv) < 4:
            print("\n[!] Home set karne ke liye lat aur lon dena zaroori hai.")
            print("\nUsage:")
            print("  python3 set_mode.py home <latitude> <longitude> [altitude_m]")
            print("\nExamples:")
            print("  python3 set_mode.py home 30.895560 74.613858           # Local Field (Auto Baro Alt)")
            print("  python3 set_mode.py home 30.895560 74.613858 177.0     # Explicit MSL Alt")
            print("\nNote: UniRC GCS map par right-click -> 'Set Home' se bhi yahi kaam")
            print("      hoga. flight_daemon chal raha ho to wo HOME_POSITION message")
            print("      automatically sun'ta hai aur EKF origin khud update kar leta hai.\n")
            sys.exit(1)
        try:
            h_lat = float(sys.argv[2])
            h_lon = float(sys.argv[3])
            h_alt = float(sys.argv[4]) if len(sys.argv) >= 5 else 0.0
            if not (-90.0 <= h_lat <= 90.0):
                raise ValueError(f"Latitude {h_lat} out of range (-90 to +90)")
            if not (-180.0 <= h_lon <= 180.0):
                raise ValueError(f"Longitude {h_lon} out of range (-180 to +180)")
            set_home(m, h_lat, h_lon, h_alt)
        except ValueError as e:
            print(f"[-] Invalid coordinates: {e}")
            sys.exit(1)

    else:
        print("\nCurrent Configuration:")
        print(" - GPS1_TYPE      :", read_parameter(m, 'GPS1_TYPE'))
        print(" - EK3_SRC1_POSXY :", read_parameter(m, 'EK3_SRC1_POSXY'))
        print(" - EK3_SRC1_VELZ  :", read_parameter(m, 'EK3_SRC1_VELZ'))
        print(" - COMPASS_USE    :", read_parameter(m, 'COMPASS_USE'))
        print(" - COMPASS_USE2   :", read_parameter(m, 'COMPASS_USE2'))
        print(" - ARMING_SKIPCHK :", read_parameter(m, 'ARMING_SKIPCHK'))
        print("\nUsage:")
        print("  python3 set_mode.py vio                       # Indoor GPS-denied")
        print("  python3 set_mode.py gps                       # Outdoor GPS")
        print("  python3 set_mode.py home <lat> <lon> [alt_m]  # Manual home (no GPS)")
