#!/usr/bin/env python3
"""
Script to set TF03 LiDAR scaling and offset on CUAV X7 Pro.
Runs without waiting for PARAM_VALUE acknowledgements to avoid hangs.

F-09 CALIBRATION NOTE:
  TF03 4-20mA over 150Ω load resistor:
    V_min = 4mA  × 150Ω = 0.600 V  → 0 m
    V_max = 20mA × 150Ω = 3.000 V  → 180 m
    Scaling = 180 / (3.000 - 0.600) = 75.0 V/m
  Empirical offset is calibrated at ground (drone resting, ~0.11m clearance).
  RNGFND1_GNDCLEAR = 11 cm (TF03 lens to ground when resting on landing gear).
"""
import sys
import time
import glob
from pymavlink import mavutil

# Find first available FCU port
ports = glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*')
if not ports:
    print('[-] No flight controller detected.')
    sys.exit(1)
port = ports[0]
print(f"[+] Connecting to CUAV on {port} ...")

m = mavutil.mavlink_connection(port, baud=115200)
try:
    m.wait_heartbeat(timeout=10)
    print('[+] Heartbeat received')
except Exception as e:
    print(f'[-] Heartbeat failed: {e}')
    sys.exit(1)

def set_param(name, value):
    print(f"    Setting {name} = {value}")
    m.mav.param_set_send(m.target_system, m.target_component,
                         name.encode('ascii'), float(value),
                         mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    time.sleep(0.2)

# F-09: Unified calibration — matches configure_cuav_downward.py
set_param('RNGFND1_SCALING', 75.0)    # F-09: 180m / (3.0V - 0.6V) = 75.0 — correct for TF03 4-20mA / 150Ω
set_param('RNGFND1_OFFSET',  0.600)   # F-09: 0.600V at 4mA (0 m) — verify empirically at your site
set_param('RNGFND1_GNDCLEAR', 11)     # F-09: 11 cm — TF03 lens height from ground when on landing gear
print('[+] Parameters set. Rebooting autopilot to apply changes.')
m.reboot_autopilot()
print('[+] Reboot command sent.')
