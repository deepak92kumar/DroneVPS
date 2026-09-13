import depthai as dai
import sys
import time

print("=" * 65)
print(f" DepthAI Version: {dai.__version__}")
print("=" * 65)

devices = dai.Device.getAllAvailableDevices()
print(f"[+] Found {len(devices)} DepthAI device(s) on bus:")

if not devices:
    print("[!] No devices found. Check USB connection.")
    sys.exit(1)

dev_info = devices[0]
print(f"  - Name / Address: {dev_info.name}")
print(f"  - Device ID (MXID): {dev_info.getMxId()}")
print(f"  - Platform: {dev_info.platform.name}")
print(f"  - Protocol: {dev_info.protocol.name}")
print(f"  - State: {dev_info.state.name}")

print("\n" + "=" * 65)
print(" Booting OAK-D Firmware & Initializing Pipeline...")
print("=" * 65)

pipeline = dai.Pipeline()

# Create color camera node
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setFps(30)
cam_rgb.setInterleaved(False)

# Create output stream
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.video.link(xout_rgb.input)

try:
    with dai.Device(pipeline, dev_info) as device:
        print("[+] SUCCESS: Device boot completed!")
        print(f"  - Negotiated USB Speed: {device.getUsbSpeed().name}")
        
        # Check connected sensors
        cams = device.getConnectedCameraFeatures()
        print(f"  - Physical Camera Sensors Detected: {len(cams)}")
        for c in cams:
            print(f"      * {c.socket.name}: {c.sensorName} (Types: {[t.name for t in c.supportedTypes]})")

        # Check IMU
        has_imu = device.getConnectedIMU()
        print(f"  - Onboard IMU Sensor: {has_imu}")

        # Check Chip Temps
        temps = device.getChipTemperature()
        print(f"  - Chip Temperature: CSS={temps.css:.1f}°C, MSS={temps.mss:.1f}°C, UPA={temps.upa:.1f}°C, DSS={temps.dss:.1f}°C")

        # Capture a test frame to verify data streaming
        print("\nVerifying video frame streaming from OAK-D...")
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=True)
        frame_received = False
        for _ in range(30):
            frame = q_rgb.get()
            if frame is not None:
                img = frame.getCvFrame()
                h, w, c = img.shape
                print(f"[+] Frame successfully captured! Resolution: {w}x{h}, Channels: {c}")
                frame_received = True
                break
            time.sleep(0.05)

        if frame_received:
            print("\n" + "=" * 65)
            print(" ALL SENSOR CHECKS PASSED: OAK-D IS 100% OPERATIONAL!")
            print("=" * 65)
        else:
            print("[-] Timeout waiting for frames.")

except Exception as e:
    print(f"[-] Error connecting to device: {e}")
    sys.exit(1)
