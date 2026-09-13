#!/usr/bin/env python3
"""
Configure CUAV X7 Pro for Downward TF03 LiDAR (Analog ADC) and Clean Up UART4
Antigravity Autonomous Flight Systems
"""

import sys
import time
import glob
from pymavlink import mavutil

print("=" * 65)
print(" CUAV X7 Pro - Downward Rangefinder & UART4 Reset")
print("=" * 65)

# Find port
ports = glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*')
if not ports:
    print("[-] ERROR: Flight controller not detected on Jetson USB!")
    print("    Kripya CUAV Flight Controller ki USB cable Jetson mein lagayein.")
    sys.exit(1)

port = ports[0]
print(f"[+] Connecting to CUAV on: {port}...")

try:
    m = mavutil.mavlink_connection(port, baud=115200)
    m.wait_heartbeat(timeout=10)
    print("[+] MAVLink Heartbeat Received!")
except Exception as e:
    print(f"[-] Connection failed: {e}")
    sys.exit(1)

def set_param(name, val):
    print(f"  Setting {name} = {val}...", end="", flush=True)
    m.mav.param_set_send(
        m.target_system, m.target_component,
        name.encode('ascii'),
        float(val),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    )
    # Wait for confirmation
    t0 = time.time()
    confirmed = False
    while time.time() - t0 < 1.0:
        msg = m.recv_match(type='PARAM_VALUE', blocking=True, timeout=0.5)
        if msg:
            p_id = msg.param_id
            if isinstance(p_id, bytes): p_id = p_id.decode('utf-8')
            p_id = p_id.strip(chr(0))
            if p_id == name:
                confirmed = True
                break
    if confirmed:
        print(" [OK]")
    else:
        print(" [SENT]")

print("\n--- 1. Configuring TF03 as Downward Rangefinder (Height) ---")
set_param('RNGFND1_TYPE', 1)        # Analog Mode
set_param('RNGFND1_PIN', 4)         # CUAV Port 09 ADC Pin
# F-09: Correct scaling for TF03 4-20mA over 150Ω:
#   V_min=0.600V (4mA=0m), V_max=3.000V (20mA=180m)
#   Scaling = 180 / (3.000 - 0.600) = 75.0
set_param('RNGFND1_SCALING', 75.0)  # F-09: Was 105.0 (wrong) → 75.0 (correct for 150Ω / 180m range)
set_param('RNGFND1_OFFSET', 0.600)  # F-09: 0.600V baseline at 0m (was 0.60, now explicit)
set_param('RNGFND1_MIN_CM', 10)     # 10 cm minimum
set_param('RNGFND1_MAX_CM', 18000)  # 180 m maximum
set_param('RNGFND1_GNDCLEAR', 11)   # F-09: 11 cm — TF03 lens height from ground on landing gear
set_param('RNGFND1_ORIENT', 25)     # 25 = Downward Facing (Height)
set_param('EK3_SRC1_POSZ', 2)       # Use Rangefinder as primary EKF3 height source

print("\n--- 2. Disabling Proximity & Obstacle Avoidance for Rangefinder ---")
set_param('PRX1_TYPE', 0)          # Disabled (No forward obstacle radar)

print("\n--- 3. Restoring UART4 & GPS2 Ports to Clean Default ---")
set_param('SERIAL6_PROTOCOL', -1)  # Restore UART4 (SERIAL6) to None/Default
set_param('SERIAL4_PROTOCOL', 5)   # Restore GPS2 (SERIAL4) to GPS

print("\n--- 4. Saving and Rebooting CUAV X7 Pro ---")
m.reboot_autopilot()
print("[+] Reboot command sent successfully!")
print("    Flight controller is rebooting. Please wait ~10 seconds.")
print("=" * 65)
