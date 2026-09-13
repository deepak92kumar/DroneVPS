import glob
import sys
import time
from pymavlink import mavutil

print("=" * 65)
print(" CUAV X7 Pro / ArduPilot MAVLink Connection Test")
print("=" * 65)

# Detect potential ports
candidate_ports = glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*") + ["/dev/ttyTHS1"]
print(f"Scanning serial ports: {candidate_ports}")

port_to_use = None
for p in candidate_ports:
    if "ACM" in p or "USB" in p:
        port_to_use = p
        break

if not port_to_use:
    port_to_use = "/dev/ttyACM0"

print(f"\n[+] Attempting connection on: {port_to_use} (Baud: 115200 / Auto)...")
print("Waiting for MAVLink HEARTBEAT from CUAV X7 Pro...")
print("(Make sure the USB Type-C cable is connected between CUAV and Jetson)\n")

try:
    master = mavutil.mavlink_connection(port_to_use, baud=115200, autoreconnect=True)
    msg = master.wait_heartbeat(timeout=10)
    
    if msg:
        print("[+] SUCCESS: MAVLink Heartbeat Received from CUAV X7 Pro!")
        print(f"  - System ID: {master.target_system}")
        print(f"  - Component ID: {master.target_component}")
        print(f"  - Autopilot Type: {mavutil.mavlink.enums['MAV_AUTOPILOT'][msg.autopilot].name}")
        print(f"  - Vehicle Type: {mavutil.mavlink.enums['MAV_TYPE'][msg.type].name}")
        print(f"  - Base Mode: {msg.base_mode}")
        print(f"  - System Status: {mavutil.mavlink.enums['MAV_STATE'][msg.system_status].name}")
        print("\n" + "=" * 65)
        print(" CUAV X7 PRO COMMUNICATION IS 100% VERIFIED!")
        print("=" * 65)
    else:
        print("[-] Timeout: No heartbeat received within 10 seconds.")
        print("Tip: If CUAV was just plugged in, check if port is detected using 'ls /dev/ttyACM*'.")

except Exception as e:
    print(f"[-] Connection failed: {e}")
    print("\nNote: Ensure the CUAV X7 Pro is plugged into Jetson via USB Type-C cable.")
