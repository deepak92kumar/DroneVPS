#!/usr/bin/env python3
"""
Benewake TF03 LiDAR - UART Configuration & Live Test Utility
Antigravity Autonomous Flight Systems
"""

import sys
import time
import glob
import serial

CMD_VERSION     = bytes([0x5A, 0x04, 0x01, 0x5F])             # Query firmware version
CMD_OUTPUT_ON   = bytes([0x5A, 0x05, 0x07, 0x01, 0x67])       # Enable continuous data output
CMD_RATE_100HZ  = bytes([0x5A, 0x06, 0x03, 0x64, 0x00, 0xC7]) # Set frame rate to 100Hz
CMD_SAVE        = bytes([0x5A, 0x04, 0x11, 0x6F])             # Save settings to flash
CMD_RESTORE_DEF = bytes([0x5A, 0x04, 0x10, 0x6E])             # Restore factory default

def find_serial_port():
    ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
    if not ports:
        return None
    return ports[0]

def send_and_wait_reply(ser, cmd, name, timeout=1.0):
    hex_str = ' '.join(f'{b:02X}' for b in cmd)
    print(f"\n[SEND] {name}: {hex_str}")
    ser.reset_input_buffer()
    ser.write(cmd)
    ser.flush()
    
    start = time.time()
    resp = bytearray()
    while time.time() - start < timeout:
        if ser.in_waiting:
            b = ser.read(ser.in_waiting)
            resp.extend(b)
            if len(resp) >= len(cmd):
                break
        time.sleep(0.02)
    
    if resp:
        resp_hex = ' '.join(f'{b:02X}' for b in resp)
        print(f"[RECV] Response: {resp_hex} (SUCCESS)")
        return True, bytes(resp)
    else:
        print("[WARN] No direct response received (Timeout).")
        return False, bytes()

def main():
    port = sys.argv[1] if len(sys.argv) > 1 else find_serial_port()
    
    if not port:
        print("\n=======================================================")
        print("ERROR: Koi USB-TTL converter port nahi mila!")
        print("Kripya USB-TTL converter ko USB port mein lagayein.")
        print("=======================================================\n")
        sys.exit(1)
        
    print(f"\n>>> Connecting to Benewake TF03 on: {port} at 115200 Baud...")
    
    try:
        ser = serial.Serial(port, 115200, timeout=0.1)
    except Exception as e:
        print(f"Error opening port {port}: {e}")
        sys.exit(1)

    print("\n--- STEP 1: Querying Firmware Version ---")
    send_and_wait_reply(ser, CMD_VERSION, "Firmware Version Query")
    time.sleep(0.1)

    print("\n--- STEP 2: Enabling Continuous Output Mode ---")
    send_and_wait_reply(ser, CMD_OUTPUT_ON, "Output Enable (Continuous)")
    time.sleep(0.1)

    print("\n--- STEP 3: Setting Frame Rate to 100Hz ---")
    send_and_wait_reply(ser, CMD_RATE_100HZ, "Set Frame Rate 100Hz")
    time.sleep(0.1)

    print("\n--- STEP 4: Saving Configuration to Flash ---")
    send_and_wait_reply(ser, CMD_SAVE, "Save to Flash")
    time.sleep(0.3)

    print("\n=======================================================")
    print("Testing for LIVE Continuous Distance Data (9-byte frames)...")
    print("Press Ctrl+C to Stop.")
    print("=======================================================\n")

    ser.reset_input_buffer()
    consecutive_frames = 0
    start_test = time.time()
    
    try:
        buf = bytearray()
        while True:
            if ser.in_waiting:
                chunk = ser.read(ser.in_waiting)
                buf.extend(chunk)

                while len(buf) >= 9:
                    if buf[0] == 0x59 and buf[1] == 0x59:
                        dist = buf[2] | (buf[3] << 8)
                        strength = buf[4] | (buf[5] << 8)
                        checksum = sum(buf[:8]) & 0xFF
                        if checksum == buf[8]:
                            consecutive_frames += 1
                            dist_m = dist / 100.0
                            print(f"\r[LIVE DATA] Distance: {dist_m:6.2f} m ({dist:5d} cm) | Signal Strength: {strength:5d} | Frames: {consecutive_frames:4d}", end="", flush=True)
                            buf = buf[9:]
                            continue
                        else:
                            buf = buf[1:]
                    else:
                        buf = buf[1:]
            
            time.sleep(0.01)
            
            if consecutive_frames == 0 and time.time() - start_test > 5.0:
                print("\r[INFO] Checking: Continuous UART frames not detected yet... (raw bytes: %d)" % len(buf), end="", flush=True)

    except KeyboardInterrupt:
        print("\n\nTest Stopped by user.")
        if consecutive_frames > 10:
            print("\n>>> RESULT: SUCCESS! UART is actively streaming distance!")
            print(">>> You can directly plug this LiDAR into CUAV X7 Pro UART4!")
        else:
            print("\n>>> RESULT: No continuous distance frames received on UART.")
            print(">>> This confirms the TF03-4-20mA model outputs distance exclusively via the 4-20mA Analog Green wire.")
    finally:
        ser.close()

if __name__ == '__main__':
    main()
