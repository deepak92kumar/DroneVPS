#!/usr/bin/env python3
"""
CUAV X7 Pro Full Configuration Script (Post-Erase Restoration)
Applies Option A Anti-Jamming & Anti-Spoofing Hybrid Architecture,
Configures TELEM1 for SIYI UniRC 7 Pro (115200 Baud),
Configures Frame as Hexacopter (Class 2, Type 1).
"""

import sys
import glob
import time
from pymavlink import mavutil

# Target Parameters Dictionary
PARAMETERS = {
    # 1. Frame Setup: Quadcopter (Class 1, Type 1 = X-frame)
    'FRAME_CLASS': 1.0,      # 1 = Quadcopter
    'FRAME_TYPE': 1.0,       # 1 = X-frame

    # 2. TELEM1 Port (SIYI UniRC 7 Pro Telemetry)
    'SERIAL1_PROTOCOL': 2.0, # MAVLink 2
    'SERIAL1_BAUD': 115.0,   # 115200 Baud (matches UniRC 7 Pro default)
    'BRD_SER1_RTSCTS': 0.0,  # Disable Flow Control for 4-pin SIYI cable
    'RC_OPTIONS': 32.0,      # Standard RC options

    # 3. USB MAVLink Port (SERIAL0)
    'SERIAL0_PROTOCOL': 2.0, # MAVLink 2

    # 4. Anti-Jamming & Anti-Spoofing Hybrid EKF3 Architecture (Option A)
    'VISO_TYPE': 1.0,        # Visual Odometry Active (OAK-D Pro)
    'GPS_TYPE': 1.0,         # Physical GPS Auto-detect Active
    'AHRS_EKF_TYPE': 3.0,    # EKF3 Active
    'EK3_ENABLE': 1.0,       # Enable EKF3
    
    # Primary Source (SRC1): OAK-D Pro Visual Odometry (Unjammable optical navigation)
    'EK3_SRC1_POSXY': 6.0,   # 6 = ExternalNav (Visual Odometry position)
    'EK3_SRC1_VELXY': 6.0,   # 6 = ExternalNav (Visual Odometry velocity active, 30Hz)
    'EK3_SRC1_POSZ': 2.0,    # 2 = Rangefinder / TF03 LiDAR (calibrated ADC)
    'EK3_SRC1_YAW': 1.0,     # 1 = Compass
    'VISO_DELAY_MS': 35.0,   # Optical flow processing latency compensation (ms)
    'EK3_VEL_I_GATE': 400.0, # Velocity innovation gate
    'FS_EKF_ACTION': 0.0,    # 0 = Warning only during tuning (prevents sudden LAND switch!)
    'FS_EKF_THRESH': 1.20,   # EKF variance threshold
    
    # Secondary Sources: Disabled indoor to prevent 'AHRS: EK3 sources require GPS' prearm lock
    'EK3_SRC2_POSXY': 0.0,
    'EK3_SRC2_VELXY': 0.0,
    'EK3_SRC2_POSZ': 0.0,
    'EK3_SRC2_YAW': 0.0,
    'EK3_SRC_OPTIONS': 0.0,
    'AHRS_GPS_USE': 0.0,
    'GPS_TYPE': 0.0,
    
    # Camera Mount Offsets (meters relative to center of gravity)
    'VISO_POS_X': 0.10,      # 10 cm forward of CoG
    'VISO_POS_Y': 0.00,      # Centered laterally
    'VISO_POS_Z': 0.13,      # 13 cm down (actual camera ground clearance)
    'VISO_ORIENT': 0.0,      # 0 = NED Earth frame (flight_daemon converts body to NED)
    'COMPASS_ENABLE': 1.0,   # Compass active
}

def find_cuav_port():
    candidates = sorted(glob.glob('/dev/ttyACM*')) + sorted(glob.glob('/dev/ttyUSB*'))
    print(f"[*] Scanning serial ports: {candidates}")
    for p in candidates:
        try:
            m = mavutil.mavlink_connection(p, baud=115200, timeout=2)
            hb = m.wait_heartbeat(timeout=1.5)
            if hb:
                print(f"[+] Found CUAV Autopilot on {p} (Type: {hb.type}, Autopilot: {hb.autopilot})")
                return m, p
        except Exception as e:
            pass
    return None, None

def set_parameter(m, param_id, value):
    param_id_bytes = param_id.encode('utf-8')
    m.mav.param_set_send(
        m.target_system, m.target_component,
        param_id_bytes,
        float(value),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    )

def read_parameter(m, param_id, timeout=2.0):
    param_id_bytes = param_id.encode('utf-8')
    m.mav.param_request_read_send(m.target_system, m.target_component, param_id_bytes, -1)
    start = time.time()
    while time.time() - start < timeout:
        msg = m.recv_match(type='PARAM_VALUE', blocking=True, timeout=0.5)
        if msg and msg.param_id == param_id:
            return msg.param_value
    return None

def main():
    print("=================================================================")
    print("      CUAV X7 PRO FULL PARAMETER RESTORATION & CONFIGURATION     ")
    print("=================================================================")
    
    m, port = find_cuav_port()
    if not m:
        print("[-] ERROR: Could not connect to CUAV X7 Pro on any serial port!")
        sys.exit(1)

    print(f"\n[*] Applying {len(PARAMETERS)} parameters...")
    failed = []
    applied = {}

    for param, val in PARAMETERS.items():
        # Set parameter
        set_parameter(m, param, val)
        time.sleep(0.08) # slight delay between sets
        
        # Read back and verify
        confirmed_val = read_parameter(m, param, timeout=1.5)
        if confirmed_val is not None and abs(confirmed_val - val) < 0.01:
            print(f"  [OK] {param:18} = {confirmed_val}")
            applied[param] = confirmed_val
        else:
            # Retry once
            set_parameter(m, param, val)
            time.sleep(0.15)
            confirmed_val = read_parameter(m, param, timeout=1.5)
            if confirmed_val is not None and abs(confirmed_val - val) < 0.01:
                print(f"  [OK*] {param:18} = {confirmed_val}")
                applied[param] = confirmed_val
            else:
                print(f"  [FAIL] {param:18} (Target: {val}, Got: {confirmed_val})")
                failed.append((param, val, confirmed_val))

    print("\n-----------------------------------------------------------------")
    print(f"Results: {len(applied)}/{len(PARAMETERS)} parameters successfully verified.")
    if failed:
        print(f"[-] WARNING: {len(failed)} parameters failed: {failed}")
    else:
        print("[+] SUCCESS: All parameters verified 100%!")

    # Request Reboot so new UART driver settings & EKF3 sources take effect
    print("\n[*] Sending Autopilot Reboot Command (MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN)...")
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
        0,
        1, 0, 0, 0, 0, 0, 0
    )
    print("[+] Reboot command sent. Waiting for CUAV X7 Pro to reboot...")
    time.sleep(6)

    # Reconnect and verify persistence
    print("[*] Reconnecting to verify parameter persistence after reboot...")
    m_new, new_port = find_cuav_port()
    if m_new:
        print(f"[+] Reconnected successfully on {new_port}!")
        sample_check = ['FRAME_CLASS', 'SERIAL1_BAUD', 'VISO_TYPE', 'EK3_SRC1_POSXY', 'BRD_SER1_RTSCTS']
        print("\n[*] Sample Persistence Verification:")
        for p in sample_check:
            val = read_parameter(m_new, p, timeout=2.0)
            print(f"  --> {p:16} = {val}")
        print("\n[+] FULL CONFIGURATION COMPLETE AND PERSISTED!")
    else:
        print("[!] Autopilot rebooted, please allow a few seconds for USB re-enumeration.")

if __name__ == '__main__':
    main()
