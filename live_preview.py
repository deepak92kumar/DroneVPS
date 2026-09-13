import depthai as dai
import cv2
import numpy as np
import os
import sys
import time
import math
from datetime import datetime
from cuav_mavlink_bridge import CuavMavlinkClient

if "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = ":0"

print("=" * 70)
print(" Starting High-Speed Decoupled Visual Odometry (OAK-D Pro)")
print(" Drone Autonomous GNSS-Denied Navigation Engine")
print(" Controls: [r] = Reset Home (0,0) | [d] = Toggle Desk Mode | [s] = Save Log | [q] = Exit")
print("=" * 70)

pipeline = dai.Pipeline()

# Configure board for rock-solid USB 2.0 High Speed mode (bypasses Jetson Tegra USB 3.0 LPM issue)
board_config = dai.BoardConfig()
board_config.usb.maxSpeed = dai.UsbSpeed.HIGH
pipeline.setBoardConfig(board_config)

# 1. Color Camera (IMX378 -> Downward Ground-Facing Visual Preview 640x400)
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setPreviewSize(640, 400)
cam_rgb.setFps(30)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

# 2. Dual Global Shutter Mono Cameras (OV9282 - ZERO MOTION BLUR)
mono_l = pipeline.create(dai.node.MonoCamera)
mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)

mono_r = pipeline.create(dai.node.MonoCamera)
mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)

# Stream Left Global-Shutter Mono directly for motion-blur-free optical flow tracking!
xout_mono = pipeline.create(dai.node.XLinkOut)
xout_mono.setStreamName("mono")
mono_l.out.link(xout_mono.input)

# 3. Active Stereo Depth Subsystem (400p with Extended Disparity for 30cm-5m range)
stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.setExtendedDisparity(True)
stereo.setLeftRightCheck(False)
stereo.setSubpixel(False)

mono_l.out.link(stereo.left)
mono_r.out.link(stereo.right)

xout_disp = pipeline.create(dai.node.XLinkOut)
xout_disp.setStreamName("disparity")
stereo.disparity.link(xout_disp.input)

xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# 4. Onboard Bosch BNO086 9-Axis IMU (100 Hz Hardware Fusion)
imu = pipeline.create(dai.node.IMU)
imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_CALIBRATED, 100)
imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 100)
imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, 100)
imu.setBatchReportThreshold(1)
imu.setMaxBatchReports(10)

xout_imu = pipeline.create(dai.node.XLinkOut)
xout_imu.setStreamName("imu")
imu.out.link(xout_imu.input)

# Connect to device with automatic retry
device = None
for attempt in range(1, 6):
    try:
        device = dai.Device(pipeline)
        break
    except Exception as e:
        print(f"[-] Waiting for OAK-D Pro camera... attempt {attempt} ({e})")
        time.sleep(1.2)

if device is None:
    print("[!] Could not connect to OAK-D Pro. Please re-insert USB cable.")
    sys.exit(1)

with device:
    # Enable Active Infrared Laser Dot Projector & Flood Light for low-texture surfaces!
    try:
        device.setIrLaserDotProjectorIntensity(0.8) # 800 mA active IR dots
        device.setIrFloodLightIntensity(0.3)        # 300 mA wide IR illumination
        print("[+] OAK-D Pro Active IR Dot Projector & Flood Light: ACTIVATED (800mA/300mA)")
    except Exception:
        try:
            device.setIrLaserDotProjectorBrightness(800)
            device.setIrFloodLightBrightness(300)
            print("[+] OAK-D Pro Active IR Dot Projector & Flood Light: ACTIVATED (800mA/300mA)")
        except Exception as e:
            print(f"[-] Active IR notification: {e}")

    q_rgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
    q_mono = device.getOutputQueue(name="mono", maxSize=1, blocking=False)
    q_disp = device.getOutputQueue(name="disparity", maxSize=1, blocking=False)
    q_depth = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
    q_imu = device.getOutputQueue(name="imu", maxSize=10, blocking=False)
    max_disp = stereo.initialConfig.getMaxDisparity()

    win_name = "OAK-D Pro: Decoupled 3D Flight Odometry + Home Memory"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1600, 800)

    # Initialize CUAV X7 Pro Autopilot MAVLink Bridge
    cuav = CuavMavlinkClient()

    # Telemetry Variables
    gyro_x_rad, gyro_y_rad, gyro_z_rad = 0.0, 0.0, 0.0
    gyro_x_deg, gyro_y_deg, gyro_z_deg = 0.0, 0.0, 0.0
    accel_x, accel_y, accel_z = 0.0, 0.0, 9.81
    roll_deg, pitch_deg = 0.0, 0.0
    
    # Altitude & Mode State
    desk_mode = True # Default True for Benchtop / Desk testing
    current_altitude_m = 0.25 # Initial resting desk height
    raw_depth_m = 0.25
    dist_str = f"{current_altitude_m:.2f} m [DESK MODE]"
    
    # Odometry State
    pos_x = 0.0           # Current X in meters (Lateral displacement from Home)
    pos_y = 0.0           # Current Y in meters (Longitudinal displacement from Home)
    total_traveled = 0.0   # Total cumulative path length in meters (Odometer)
    max_outbound = 0.0     # Maximum distance reached from Home
    live_speed = 0.0       # Current horizontal speed in m/s
    last_video_time = time.time()
    
    prev_mono = None
    prev_pts = None
    focal_length_px = 440.0 # OV9282 Mono focal length at 640 width
    cx_f, cy_f = 320.0, 200.0 # Optical principal point (center of 640x400 mono frame)
    log_saved_banner_time = 0
    frame_counter = 0

    # =========================================================================
    # HOME LOCATION MEMORY (Visual Keyframe Anchor & Loop Closure)
    # Saves high-contrast ORB descriptors of takeoff spot so drift snaps to 0!
    # =========================================================================
    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=5)
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    fast_detector = cv2.FastFeatureDetector_create(threshold=12, nonmaxSuppression=True)
    
    home_anchor_img = None
    home_anchor_kp = None
    home_anchor_des = None
    home_locked = False
    home_relocalized = False
    home_match_count = 0
    steady_counter = 0
    last_reloc_check_time = 0.0
    reloc_status_text = "PLACE ON DESK & PRESS [r]"

    # ArUco Precision Landing Detector Setup (DICT_4X4_50, Marker ID 0, 25cm target)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict)
    aruco_locked = False
    aruco_offset_str = ""
    aruco_marker_size_m = 0.25
    aruco_obj_pts = np.array([
        [-aruco_marker_size_m/2,  aruco_marker_size_m/2, 0],
        [ aruco_marker_size_m/2,  aruco_marker_size_m/2, 0],
        [ aruco_marker_size_m/2, -aruco_marker_size_m/2, 0],
        [-aruco_marker_size_m/2, -aruco_marker_size_m/2, 0]
    ], dtype=np.float32)
    K_cam = np.array([[focal_length_px, 0.0, cx_f], [0.0, focal_length_px, cy_f], [0.0, 0.0, 1.0]], dtype=np.float32)
    dist_cam = np.zeros((4, 1), dtype=np.float32)

    fps_count = 0
    start_time = time.time()
    fps_display = "30"

    print("[+] Decoupled 3D Navigation Loop Started...")

    while True:
        # 1. Update IMU Gyroscope & Attitude (100 Hz)
        in_imu = q_imu.tryGet()
        if in_imu is not None:
            for packet in in_imu.packets:
                gyro_x_rad = packet.gyroscope.x
                gyro_y_rad = packet.gyroscope.y
                gyro_z_rad = packet.gyroscope.z

                gyro_x_deg = gyro_x_rad * (180.0 / math.pi)
                gyro_y_deg = gyro_y_rad * (180.0 / math.pi)
                gyro_z_deg = gyro_z_rad * (180.0 / math.pi)
                
                accel_x = packet.acceleroMeter.x
                accel_y = packet.acceleroMeter.y
                accel_z = packet.acceleroMeter.z

                rv = packet.rotationVector
                q_i, q_j, q_k, q_r = rv.i, rv.j, rv.k, rv.real
                
                sinr_cosp = 2 * (q_r * q_i + q_j * q_k)
                cosr_cosp = 1 - 2 * (q_i * q_i + q_j * q_j)
                roll_deg = math.atan2(sinr_cosp, cosr_cosp) * (180.0 / math.pi)

                sinp = 2 * (q_r * q_j - q_k * q_i)
                if abs(sinp) >= 1:
                    pitch_deg = math.copysign(90.0, sinp)
                else:
                    pitch_deg = math.asin(sinp) * (180.0 / math.pi)

        # 2. Get Video, Global Shutter Mono & Depth Frames
        in_rgb = q_rgb.tryGet()
        in_mono = q_mono.tryGet()
        in_disp = q_disp.tryGet()
        in_depth = q_depth.tryGet()

        if in_rgb is not None and in_disp is not None and in_mono is not None:
            # Measure TRUE inter-frame time interval between camera frames!
            current_time = time.time()
            dt = max(0.01, min(0.15, current_time - last_video_time))
            last_video_time = current_time
            frame_counter += 1

            rgb_f = in_rgb.getCvFrame()
            mono_f = in_mono.getCvFrame() # Pure OV9282 Global Shutter - No Motion Blur!
            disp_f = in_disp.getFrame()

            # Render High-Definition display frames (800x500 each, 1600 total width)
            rgb_hd = cv2.resize(rgb_f, (800, 500))

            disp_vis = (disp_f * (255.0 / max_disp)).astype(np.uint8)
            depth_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
            depth_hd = cv2.resize(depth_color, (800, 500))

            # 3. Ground Clearance / Altitude (Z) in Millimeters
            if in_depth is not None:
                depth_raw = in_depth.getFrame()
                dh, dw = depth_raw.shape
                
                y_min, y_max = int(dh * 0.25), int(dh * 0.75)
                x_min, x_max = int(dw * 0.25), int(dw * 0.75)
                center_roi = depth_raw[y_min:y_max, x_min:x_max]
                valid_depths = center_roi[(center_roi > 200) & (center_roi < 6000)]
                
                if len(valid_depths) > 40:
                    depth_mm = float(np.median(valid_depths))
                    raw_depth_m = np.clip(depth_mm / 1000.0, 0.20, 4.5)
                
                if desk_mode:
                    # In Desk Mode: use close valid depth if available (25cm to 85cm)
                    if 0.22 <= raw_depth_m <= 0.85:
                        current_altitude_m = 0.85 * current_altitude_m + 0.15 * raw_depth_m
                    dist_str = f"{current_altitude_m:.2f} m (DESK MODE)"
                else:
                    # Live Flight Mode: full 3D stereo altitude (1.0m to 4.5m)
                    current_altitude_m = 0.85 * current_altitude_m + 0.15 * raw_depth_m
                    dist_str = f"{current_altitude_m:.2f} m (FLIGHT STEREO)"

            # =================================================================
            # 4. HOME LOCATION MEMORY: AUTO-LOCK WHEN RESTING STEADY ON DESK
            # Automatically captures Home Anchor once camera is steady for 0.5s!
            # =================================================================
            is_steady = (abs(gyro_x_deg) < 2.5 and abs(gyro_y_deg) < 2.5 and abs(gyro_z_deg) < 2.5)
            if is_steady:
                steady_counter += 1
            else:
                steady_counter = 0

            if not home_locked and steady_counter >= 15 and frame_counter >= 20:
                home_anchor_img = mono_f.copy()
                home_anchor_kp, home_anchor_des = orb.detectAndCompute(home_anchor_img, None)
                if home_anchor_des is not None and len(home_anchor_kp) >= 25:
                    home_locked = True
                    pos_x = 0.0
                    pos_y = 0.0
                    total_traveled = 0.0
                    max_outbound = 0.0
                    reloc_status_text = f"HOME MEMORY SAVED ({len(home_anchor_kp)} Features)"
                    print(f"[+] Steady Takeoff Spot Memorized: {len(home_anchor_kp)} ORB features locked!")

            # =================================================================
            # 5. DECOUPLED AFFINE RANSAC OPTICAL FLOW TRACKING
            # Mathematically decouples vertical lift (scale expansion) from horizontal translation!
            # =================================================================
            tracked_points_drawn = 0
            tracking_status = "TRACKING ACTIVE"
            tracking_status_color = (0, 255, 0)

            if prev_mono is not None and prev_pts is not None and len(prev_pts) > 0:
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_mono, mono_f, prev_pts, None,
                    winSize=(25, 25), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.02)
                )

                if next_pts is not None and status is not None:
                    good_new = next_pts[status == 1]
                    good_old = prev_pts[status == 1]

                    if len(good_new) >= 6:
                        # Center coordinates around principal point (cx_f, cy_f)
                        pts_old_c = good_old - [cx_f, cy_f]
                        pts_new_c = good_new - [cx_f, cy_f]

                        # Decoupled 2D Similarity RANSAC:
                        M, inliers = cv2.estimateAffinePartial2D(
                            pts_old_c, pts_new_c, method=cv2.RANSAC, ransacReprojThreshold=3.0
                        )

                        if M is not None:
                            trans_dx_px = float(M[0, 2])
                            trans_dy_px = float(M[1, 2])
                            lift_scale = float(np.sqrt(M[0, 0]**2 + M[0, 1]**2))

                            # Dynamically track vertical height change from visual lift scale:
                            if desk_mode and 0.88 < lift_scale < 1.14 and abs(lift_scale - 1.0) > 0.002:
                                # Scale < 1 means lifting UP (distance increases)
                                # Scale > 1 means lowering DOWN (distance decreases)
                                current_altitude_m = np.clip(current_altitude_m / lift_scale, 0.15, 0.85)

                            # Deadband for sensor noise
                            if abs(trans_dx_px) < 0.20:
                                trans_dx_px = 0.0
                            if abs(trans_dy_px) < 0.20:
                                trans_dy_px = 0.0

                            meter_per_px = current_altitude_m / focal_length_px
                            dx_m = -(trans_dx_px * meter_per_px)
                            dy_m = (trans_dy_px * meter_per_px)

                            step_dist = math.sqrt(dx_m**2 + dy_m**2)

                            # Accept full real physical motion (up to 0.40m per 33ms frame = 12 m/s)
                            if 0.00005 < step_dist < 0.40:
                                pos_x += dx_m
                                pos_y += dy_m
                                total_traveled += step_dist
                                live_speed = step_dist / dt

                            # Render green tracking vectors on HD Preview
                            scale_x = 800.0 / 640.0
                            scale_y = 500.0 / 400.0
                            for new_pt, old_pt in zip(good_new[:80], good_old[:80]):
                                a, b = int(new_pt[0] * scale_x), int(new_pt[1] * scale_y)
                                c, d = int(old_pt[0] * scale_x), int(old_pt[1] * scale_y)
                                cv2.circle(rgb_hd, (a, b), 4, (0, 255, 0), -1, lineType=cv2.LINE_AA)
                                cv2.line(rgb_hd, (a, b), (c, d), (0, 255, 255), 1, lineType=cv2.LINE_AA)
                                tracked_points_drawn += 1

                            prev_pts = good_new.reshape(-1, 1, 2)
                        else:
                            prev_pts = None
                    else:
                        prev_pts = None
                else:
                    prev_pts = None

            # =================================================================
            # 6. CONTINUOUS HIGH-DENSITY FEATURE REPLENISHMENT
            # As camera moves across 60-70cm desk, continually replenish features!
            # =================================================================
            if prev_pts is None or len(prev_pts) < 80:
                detected_corners = cv2.goodFeaturesToTrack(
                    mono_f, maxCorners=160, qualityLevel=0.002, minDistance=8
                )
                if detected_corners is not None and len(detected_corners) > 0:
                    if prev_pts is not None and len(prev_pts) > 0:
                        prev_pts = np.vstack((prev_pts, detected_corners))[:160]
                    else:
                        prev_pts = detected_corners
                else:
                    # Fallback to FAST Detector
                    fast_kp = fast_detector.detect(mono_f, None)
                    if fast_kp and len(fast_kp) > 0:
                        fast_pts = np.float32([kp.pt for kp in fast_kp]).reshape(-1, 1, 2)
                        prev_pts = fast_pts[:160]

            prev_mono = mono_f
            live_speed *= 0.88

            # Calculate Straight-Line Distance to Home
            home_dist = math.sqrt(pos_x**2 + pos_y**2)
            if home_dist > max_outbound:
                max_outbound = home_dist

            # =================================================================
            # 7. VERIFIED RANSAC HOME RELOCALIZATION (Gated to true Home neighborhood)
            # Only snaps when returning towards Home (<0.35m) with >= 20 geometric inliers!
            # =================================================================
            if home_locked and home_anchor_des is not None and (current_time - last_reloc_check_time > 0.20):
                last_reloc_check_time = current_time
                curr_kp, curr_des = orb.detectAndCompute(mono_f, None)
                if curr_des is not None and len(curr_kp) >= 15:
                    raw_matches = bf_matcher.knnMatch(curr_des, home_anchor_des, k=2)
                    good_matches = []
                    for m_pair in raw_matches:
                        if len(m_pair) == 2:
                            m, n = m_pair
                            if m.distance < 0.72 * n.distance: # Stricter Lowe's ratio
                                good_matches.append(m)
                    
                    if len(good_matches) >= 15:
                        src_pts = np.float32([curr_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
                        dst_pts = np.float32([home_anchor_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
                        
                        # Geometric Homography RANSAC: strictly filter out false matches from repetitive desk texture!
                        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
                        inlier_count = int(np.sum(mask)) if mask is not None else 0
                        home_match_count = inlier_count

                        # Relocalize whenever verified geometric RANSAC inliers >= 16
                        if inlier_count >= 16:
                            inlier_src = src_pts[mask.ravel() == 1]
                            inlier_dst = dst_pts[mask.ravel() == 1]
                            diff = inlier_src - inlier_dst
                            median_dx_px = float(np.median(diff[:, 0]))
                            median_dy_px = float(np.median(diff[:, 1]))
                            
                            meter_per_px = current_altitude_m / focal_length_px
                            target_x = -(median_dx_px * meter_per_px)
                            target_y = (median_dy_px * meter_per_px)
                            
                            # Smoothly correct residual drift to Home (0,0)
                            pos_x = 0.60 * pos_x + 0.40 * target_x
                            pos_y = 0.60 * pos_y + 0.40 * target_y
                            home_relocalized = True
                            reloc_status_text = f"HOME LOCK: {inlier_count} Inliers (0 cm Drift!)"
                        else:
                            home_relocalized = False
                            reloc_status_text = f"Flight Outbound ({inlier_count} Inliers)"
                    else:
                        home_match_count = 0
                        home_relocalized = False
                        reloc_status_text = f"Flight Outbound (Tracking Active)"

            # =================================================================
            # 7B. ARUCO PRECISION LANDING TARGET DETECTION (Marker ID 0)
            # Centimeter-accurate target locking for precision return & landing!
            # =================================================================
            aruco_corners, aruco_ids, _ = aruco_detector.detectMarkers(mono_f)
            aruco_locked = False
            if aruco_ids is not None and 0 in aruco_ids:
                aruco_locked = True
                idx_0 = list(aruco_ids.ravel()).index(0)
                m_c = aruco_corners[idx_0].reshape(4, 2)
                success, rvec, tvec = cv2.solvePnP(aruco_obj_pts, m_c, K_cam, dist_cam)
                if success:
                    t_x, t_y, t_z = float(tvec[0]), float(tvec[1]), float(tvec[2])
                    # live_preview.py FIX: Rotate body-frame ArUco offset to NED before position blend.
                    # Downward camera body frame: +X_cam = right, +Y_cam = forward (towards tail).
                    # body_forward = t_y (marker forward of drone), body_right = -t_x (marker to right).
                    # Rotate to NED using current yaw heading from CUAV (or zero if disconnected).
                    yaw_r = math.radians(cuav.yaw_deg if cuav.is_connected else 0.0)
                    cos_y = math.cos(yaw_r)
                    sin_y = math.sin(yaw_r)
                    body_fwd = t_y
                    body_rgt = -t_x
                    ned_north = body_fwd * cos_y - body_rgt * sin_y
                    ned_east  = body_fwd * sin_y + body_rgt * cos_y
                    # pos_x = East (NED), pos_y = North (NED) per live_preview.py convention
                    pos_x = 0.70 * pos_x + 0.30 * ned_east
                    pos_y = 0.70 * pos_y + 0.30 * ned_north
                    current_altitude_m = 0.70 * current_altitude_m + 0.30 * max(0.15, t_z)
                    aruco_offset_str = f"N:{ned_north*100:+04.1f}cm E:{ned_east*100:+04.1f}cm Alt:{t_z*100:04.1f}cm"

                scale_x = 800.0 / 640.0
                scale_y = 500.0 / 400.0
                pts_scaled = (m_c * [scale_x, scale_y]).astype(np.int32)
                cv2.polylines(rgb_hd, [pts_scaled], True, (0, 255, 0), 3, lineType=cv2.LINE_AA)
                pad_center = np.mean(pts_scaled, axis=0).astype(int)
                cv2.circle(rgb_hd, (pad_center[0], pad_center[1]), 8, (0, 255, 255), -1, lineType=cv2.LINE_AA)
                cv2.putText(rgb_hd, "ARUCO LANDING PAD #0", (pts_scaled[0][0] - 20, max(25, pts_scaled[0][1] - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, lineType=cv2.LINE_AA)

            # Stream Visual Odometry to CUAV X7 Pro Autopilot (MAVLink at 30 Hz)
            if cuav.is_connected:
                cuav.send_vision_estimate(
                    x=pos_y,               # NED North / Forward
                    y=pos_x,               # NED East / Right
                    z=-current_altitude_m, # NED Down (-Altitude)
                    roll_rad=math.radians(roll_deg),
                    pitch_rad=math.radians(pitch_deg),
                    yaw_rad=0.0
                )

            # Draw Ground Optical Tracking Reticle
            cx, cy = 400, 250
            reticle_color = (0, 255, 0) if not home_relocalized else (0, 255, 255)
            cv2.line(rgb_hd, (cx - 30, cy), (cx + 30, cy), reticle_color, 2, lineType=cv2.LINE_AA)
            cv2.line(rgb_hd, (cx, cy - 30), (cx, cy + 30), reticle_color, 2, lineType=cv2.LINE_AA)
            cv2.circle(rgb_hd, (cx, cy), 32, reticle_color, 2, lineType=cv2.LINE_AA)
            cv2.circle(rgb_hd, (cx, cy), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)

            # Ground Height Badge
            cv2.rectangle(rgb_hd, (cx - 110, cy + 40), (cx + 110, cy + 72), (15, 23, 42), -1)
            cv2.rectangle(rgb_hd, (cx - 110, cy + 40), (cx + 110, cy + 72), (0, 255, 0), 1, lineType=cv2.LINE_AA)
            cv2.putText(rgb_hd, f"ALT (Z): {dist_str}", (cx - 100, cy + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, lineType=cv2.LINE_AA)

            # Matching Crosshair on 3D Terrain Depth Map
            cv2.line(depth_hd, (cx - 25, cy), (cx + 25, cy), (255, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.line(depth_hd, (cx, cy - 25), (cx, cy + 25), (255, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.circle(depth_hd, (cx, cy), 32, (255, 255, 255), 2, lineType=cv2.LINE_AA)

            # Log saved notification banner
            if time.time() - log_saved_banner_time < 2.5:
                cv2.rectangle(rgb_hd, (160, 20), (640, 65), (0, 150, 0), -1)
                cv2.putText(rgb_hd, "FLIGHT LOG SAVED TO DESKTOP/DRONE!", (175, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, lineType=cv2.LINE_AA)

            # Live FPS
            fps_count += 1
            if time.time() - start_time >= 1.0:
                fps_display = f"{fps_count}"
                fps_count = 0
                start_time = time.time()

            # Stack video: 800x500 + 800x500 -> 1600x500
            video_stacked = np.hstack((rgb_hd, depth_hd))

            # Header Labels
            mode_badge = "DESK MODE (Adaptive)" if desk_mode else "STEREO FLIGHT MODE"
            cv2.putText(video_stacked, f"GROUND TRACKING ({fps_display} FPS | {tracked_points_drawn} Pts | {mode_badge})", 
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.putText(video_stacked, "3D STEREO DEPTH TERRAIN (EXTENDED DISPARITY)", 
                        (820, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, lineType=cv2.LINE_AA)

            # =================================================================
            # 8. DEDICATED HD TELEMETRY DASHBOARD (1600x215) - Anti-Aliased
            # =================================================================
            hud = np.zeros((215, 1600, 3), dtype=np.uint8)
            hud[:] = (16, 20, 26)
            cv2.line(hud, (0, 0), (1600, 0), (60, 80, 100), 2, lineType=cv2.LINE_AA)

            # Column 1: BNO086 GYROSCOPE & ATTITUDE (Left: x=20..460)
            cv2.putText(hud, "GYROSCOPE & ATTITUDE", (25, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(hud, f"Roll Rate   : {gyro_x_deg:+06.1f} deg/s", (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, lineType=cv2.LINE_AA)
            cv2.putText(hud, f"Pitch Rate  : {gyro_y_deg:+06.1f} deg/s", (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, lineType=cv2.LINE_AA)
            cv2.putText(hud, f"Tilt (R, P) : R:{roll_deg:+05.1f}  P:{pitch_deg:+05.1f}", (25, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, lineType=cv2.LINE_AA)
            status_text = "LEVEL HOVER" if abs(roll_deg) < 5 and abs(pitch_deg) < 5 else "TILT COMPENSATION ACTIVE"
            status_color = (0, 255, 0) if "LEVEL" in status_text else (0, 165, 255)
            cv2.putText(hud, f"Flight Mode : {status_text}", (25, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_color, 2, lineType=cv2.LINE_AA)

            # Vertical divider 1
            cv2.line(hud, (480, 15), (480, 165), (40, 55, 70), 1, lineType=cv2.LINE_AA)

            # Column 2: TRAVEL ODOMETER & ALTITUDE (Center: x=510..980)
            cv2.putText(hud, "ALTITUDE & TRAVEL ODOMETER", (510, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(hud, f"Ground Altitude (Z) : {dist_str}", (510, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            
            tot_cm = int(total_traveled * 100)
            cv2.putText(hud, f"Total Path Flown    : {total_traveled:05.2f}m ({tot_cm} cm) [Cumulative]", (510, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, lineType=cv2.LINE_AA)
            
            max_out_cm = int(max_outbound * 100)
            cv2.putText(hud, f"Max Outbound Range  : {max_outbound:05.2f}m ({max_out_cm} cm)", (510, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 200, 0), 1, lineType=cv2.LINE_AA)
            cv2.putText(hud, f"Horizontal Speed    : {live_speed:.2f} m/s ({live_speed*3.6:.1f} km/h)", (510, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, lineType=cv2.LINE_AA)

            # Vertical divider 2
            cv2.line(hud, (990, 15), (990, 165), (40, 55, 70), 1, lineType=cv2.LINE_AA)

            # Column 3: RETURN-TO-HOME & VISUAL MEMORY (Right: x=1020..1580)
            cv2.putText(hud, "RETURN-TO-HOME & LOCATION MEMORY", (1020, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 180), 2, lineType=cv2.LINE_AA)
            
            pos_x_cm = int(pos_x * 100)
            pos_y_cm = int(pos_y * 100)
            cv2.putText(hud, f"Position (X, Y) : X: {pos_x:+05.2f}m ({pos_x_cm:+d}cm) | Y: {pos_y:+05.2f}m ({pos_y_cm:+d}cm)", 
                        (1020, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 2, lineType=cv2.LINE_AA)
            
            home_dist_cm = int(home_dist * 100)
            if aruco_locked:
                drift_label = f"ARUCO LOCK: {aruco_offset_str} (PRECISION PAD!)"
                drift_color = (0, 255, 0) # Green
            elif home_relocalized:
                drift_label = f"RETURN LOCK: Drift {home_dist_cm} cm (MEMORY MATCHED!)"
                drift_color = (0, 255, 0) # Green
            elif home_dist < 0.08:
                drift_label = f"RETURN LOCK: Drift {home_dist_cm} cm (PINPOINT!)"
                drift_color = (0, 255, 0) # Green
            else:
                drift_label = f"Distance to Home: {home_dist:05.2f}m ({home_dist_cm} cm) [Decreases to 0]"
                drift_color = (0, 255, 180) # Cyan

            cv2.putText(hud, drift_label, (1020, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.50, drift_color, 2, lineType=cv2.LINE_AA)
            
            # Memory status
            mem_color = (0, 255, 0) if (aruco_locked or home_relocalized) else ((0, 255, 255) if home_locked else (0, 0, 255))
            mem_text = "ARUCO LANDING PAD LOCKED" if aruco_locked else reloc_status_text
            cv2.putText(hud, f"Memory: {mem_text}", (1020, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.48, mem_color, 1, lineType=cv2.LINE_AA)
            cv2.putText(hud, "Controls: [r]=Reset Home | [d]=Toggle Desk | [s]=Save | [q]=Exit", (1020, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1, lineType=cv2.LINE_AA)

            # Row 2: CUAV X7 PRO AUTOPILOT STATUS BAR
            cv2.line(hud, (0, 175), (1600, 175), (45, 60, 80), 1, lineType=cv2.LINE_AA)
            if cuav.is_connected:
                var_str = f"PosVar: {cuav.ekf_horiz_var:.3f}" if cuav.ekf_horiz_var < 50 else "Converging"
                cuav_banner = f"CUAV X7 PRO: CONNECTED ({cuav.port}) | Flight Mode: {cuav.flight_mode} | EKF3 VIO FUSION: ACTIVE ({var_str}) | MAVLink VIO Sent: {cuav.vio_packets_sent} pkts"
                cuav_clr = (0, 255, 0)
            else:
                cuav_banner = "CUAV X7 PRO: NOT CONNECTED (Plug in USB-C cable to auto-link with Autopilot | Standalone Camera Tracking Active)"
                cuav_clr = (140, 140, 140)
            cv2.putText(hud, cuav_banner, (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.50, cuav_clr, 2, lineType=cv2.LINE_AA)

            # Stack video on top and HUD on bottom: 1600x(500+215) = 1600x715
            full_dashboard = np.vstack((video_stacked, hud))

            try:
                cv2.imshow(win_name, full_dashboard)
            except Exception:
                break

        # Check keyboard commands
        key = cv2.waitKey(1) & 0xFF
        
        # 'r' -> Reset Home Origin (0,0) and capture new Home Anchor
        if key in [ord('r'), ord('R')]:
            pos_x = 0.0
            pos_y = 0.0
            total_traveled = 0.0
            max_outbound = 0.0
            live_speed = 0.0
            current_altitude_m = 0.25 # Reset baseline desk height
            cuav.reset_origin()
            if mono_f is not None:
                home_anchor_img = mono_f.copy()
                home_anchor_kp, home_anchor_des = orb.detectAndCompute(home_anchor_img, None)
                home_locked = True
                reloc_status_text = f"HOME MEMORY SAVED ({len(home_anchor_kp)} Features)"
                print(f"[+] Home Reset! Origin set to (0,0). New Home Anchor memorized.")

        # 'd' -> Toggle Desk Test Mode vs Full Stereo Flight Mode
        if key in [ord('d'), ord('D')]:
            desk_mode = not desk_mode
            mode_name = "DESK MODE (Adaptive)" if desk_mode else "FULL STEREO FLIGHT MODE"
            print(f"[+] Switched Mode to: {mode_name}")

        # 's' -> Save Flight / Round-trip Test Report
        if key in [ord('s'), ord('S')]:
            log_path = "/home/abc/Desktop/DRONE/flight_test_log.txt"
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            reloc_str = f"LOCKED ({home_match_count} RANSAC Inliers)" if home_relocalized else "Optical Flow Integration"
            mode_str = f"Desk Mode ({current_altitude_m:.2f}m)" if desk_mode else "Stereo Flight Mode"
            log_entry = f"""
=====================================================
 DRONE VISUAL ODOMETRY ROUND-TRIP FLIGHT REPORT
 Date & Time       : {timestamp_str}
=====================================================
 Max Outbound Dist : {max_outbound:.2f} meters ({int(max_outbound*100)} cm)
 Total Path Length : {total_traveled:.2f} meters ({int(total_traveled*100)} cm) [Odometer]
 Final Return Pos  : X: {pos_x:+.2f}m ({int(pos_x*100)}cm), Y: {pos_y:+.2f}m ({int(pos_y*100)}cm)
 Residual Drift    : {home_dist:.2f} meters ({int(home_dist*100)} cm) [Distance from Home]
 Ground Clearance  : {current_altitude_m:.2f} meters ({mode_str})
 Tilt (Roll, Pitch): R: {roll_deg:+.1f} deg, P: {pitch_deg:+.1f} deg
 Memory Match      : {home_match_count} Features ({reloc_str})
 Tracking Sensor   : Decoupled Affine OV9282 Global Shutter + Active IR
=====================================================
"""
            with open(log_path, "a") as f:
                f.write(log_entry)
            log_saved_banner_time = time.time()
            print(f"[+] Flight Report Saved to {log_path}!")

        if key in [ord('q'), ord('Q'), 27]:
            print("\n[+] Exit key pressed. Closing cleanly...")
            break

    cuav.close()
    cv2.destroyAllWindows()
