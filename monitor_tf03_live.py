#!/usr/bin/env python3
"""
Benewake TF03 4-20mA LiDAR - Real-time Live Altitude & Voltage Monitor
Antigravity Autonomous Flight Systems
"""

import sys
import time
import glob
from pymavlink import mavutil

print("=" * 65)
print("     TF03 LiDAR Real-Time Live Monitor (CUAV X7 Pro ADC)")
print("=" * 65)

ports = glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*')
active_conn = None

for p in sorted(ports):
    try:
        m = mavutil.mavlink_connection(p, baud=115200)
        m.post_message = lambda msg: None
        hb = m.wait_heartbeat(timeout=1.0)
        if hb:
            active_conn = m
            print(f"[+] Connected to CUAV X7 Pro on: {p}")
            break
    except Exception:
        continue

if not active_conn:
    print("[-] Error: CUAV X7 Pro not detected on any USB port!")
    sys.exit(1)

m = active_conn
m.mav.request_data_stream_send(
    m.target_system, m.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_ALL, 5, 1
)

print("[+] Listening to live TF03 distance & voltage... (Press Ctrl+C to stop)")
print("-" * 65)
print(f"{'Time':<8} | {'Distance (m)':<14} | {'Voltage (V)':<12} | {'Current (mA)':<12} | {'Visual Bar'}")
print("-" * 65)

t_start = time.time()
last_v = 0.612
try:
    while True:
        msg = m.recv_msg()
        if not msg:
            time.sleep(0.01)
            continue
        
        t = msg.get_type()
        if t == 'RANGEFINDER':
            last_v = msg.voltage
            dist_m = msg.distance
            curr_ma = (last_v / 150.0) * 1000.0
            elapsed = time.time() - t_start
            
            bar_len = int(min(max(dist_m, 0.0) * 8, 25))
            bar = "#" * bar_len + "-" * (25 - bar_len)
            print(f"\r{elapsed:5.1f}s    | {dist_m:6.2f} m        | {last_v:6.4f} V     | {curr_ma:5.2f} mA      | [{bar}]", end="", flush=True)
            
        elif t == 'DISTANCE_SENSOR':
            dist_cm = msg.current_distance
            dist_m = dist_cm / 100.0
            curr_ma = (last_v / 150.0) * 1000.0
            elapsed = time.time() - t_start
            
            bar_len = int(min(max(dist_m, 0.0) * 8, 25))
            bar = "#" * bar_len + "-" * (25 - bar_len)
            print(f"\r{elapsed:5.1f}s    | {dist_m:6.2f} m        | {last_v:6.4f} V     | {curr_ma:5.2f} mA      | [{bar}]", end="", flush=True)

except KeyboardInterrupt:
    print("\n\n[+] Monitor stopped by user.")
