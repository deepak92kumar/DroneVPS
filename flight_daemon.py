#!/usr/bin/env python3
"""
Autonomous Headless Drone Navigation Daemon
Designed for onboard companion computer (NVIDIA Jetson Orin Nano).
Runs headlessly without GUI/Monitor.
Streams OAK-D Pro Visual Odometry + ArUco Precision Landing to CUAV X7 Pro EKF3.
"""

import os
# Must precede pymavlink import: MAVLink1 drops VISION_* covariance and reset_counter.
os.environ.setdefault("MAVLINK20", "1")
import sys
import time
import math
import glob
import signal
import logging
import argparse
import threading
import queue
from collections import deque
from datetime import datetime
import numpy as np
import cv2
import cv2.aruco as aruco
import depthai as dai

# Import CUAV MAVLink Client from the same folder
sys.path.insert(0, "/home/abc/SSD/DRONE")
from pymavlink import mavutil
from cuav_mavlink_bridge import CuavMavlinkClient

# NVMe SSD paths (primary) with eMMC fallback
_SSD_BASE   = "/home/abc/SSD/DRONE"
_EMMC_BASE  = "/home/abc/Desktop/DRONE"
_BASE       = _SSD_BASE if os.path.ismount("/home/abc/SSD") or os.path.exists(_SSD_BASE) else _EMMC_BASE
LOG_DIR     = os.path.join(_BASE, "flight_logs")
IMAGES_DIR  = os.path.join(_BASE, "flight_images")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Global running flag
RUNNING = True
_cuav_ref = None  # F-14: global ref so signal handler can send LAND on exit

def handle_exit(signum, frame):
    global RUNNING
    logging.info(f"Received shutdown signal ({signum}). Exiting cleanly...")
    RUNNING = False
    # F-14: Send LAND command to FC before daemon exits
    if _cuav_ref is not None and _cuav_ref.is_connected and _cuav_ref.is_armed:
        try:
            _cuav_ref.master.mav.command_long_send(
                _cuav_ref.master.target_system,
                _cuav_ref.master.target_component,
                mavutil.mavlink.MAV_CMD_NAV_LAND,
                0, 0, 0, 0, 0, 0, 0, 0
            )
            logging.warning("[!] Signal received while armed — LAND command sent to FC")
        except Exception:
            pass

signal.signal(signal.SIGTERM, handle_exit)
signal.signal(signal.SIGINT, handle_exit)

class AsyncImageLogger(threading.Thread):
    """
    Decoupled background image recording worker.
    Ensures 2 FPS logging and ArUco captures never block the 30 Hz VIO loop.
    """
    def __init__(self, max_queue=25):
        super().__init__(daemon=True)
        self.queue = queue.Queue(maxsize=max_queue)
        self.running = True
        self.start()

    def run(self):
        while self.running:
            try:
                task = self.queue.get(timeout=0.2)
                if task is None:
                    continue
                path, img_data, is_bytes = task
                if is_bytes:
                    tmp_path = path + ".tmp"
                    with open(tmp_path, "wb") as f:
                        f.write(img_data)
                    os.replace(tmp_path, path)
                else:
                    tmp_path = path + ".tmp.jpg"
                    cv2.imwrite(tmp_path, img_data, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    os.replace(tmp_path, path)
                self.queue.task_done()
            except queue.Empty:
                pass
            except Exception:
                pass

    def log_image(self, path, img_data, is_bytes=False):
        try:
            self.queue.put_nowait((path, img_data, is_bytes))
            return True
        except queue.Full:
            return False

    def stop(self):
        self.running = False

def setup_logging(test_mode=False):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"flight_{timestamp}.log")
    
    handlers = [logging.FileHandler(log_file)]
    if test_mode or sys.stdout.isatty():
        handlers.append(logging.StreamHandler(sys.stdout))
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers
    )
    logging.info("=" * 65)
    logging.info(" Autonomous Drone Navigation Daemon Starting (Headless Mode)")
    logging.info(f" Log file: {log_file}")
    logging.info("=" * 65)
    return log_file

def build_oak_pipeline(max_exposure_us=3000, max_iso=1200):
    pipeline = dai.Pipeline()

    board_config = dai.BoardConfig()
    board_config.usb.maxSpeed = dai.UsbSpeed.SUPER
    pipeline.setBoardConfig(board_config)

    # 1. Global Shutter OV9282 Downward Mono Camera @ 640x400, 100 FPS
    mono_l = pipeline.create(dai.node.MonoCamera)
    mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_l.setFps(100)
    mono_l.initialControl.setAutoExposureEnable()
    mono_l.initialControl.setAutoExposureLimit(max_exposure_us)

    # Output Left Mono stream for ArUco & flight recording
    xout_mono = pipeline.create(dai.node.XLinkOut)
    xout_mono.setStreamName("mono")
    mono_l.out.link(xout_mono.input)

    # 2. OAK-D Myriad-X FeatureTracker (Optical Flow on-chip @ 100 FPS, Full 2 SHAVEs)
    feature_tracker = pipeline.create(dai.node.FeatureTracker)
    feature_tracker.setHardwareResources(2, 2)  # Full 2 SHAVE cores & memory slices allocated
    feature_tracker.initialConfig.setCornerDetector(
        dai.RawFeatureTrackerConfig.CornerDetector.Type.HARRIS
    )
    feature_tracker.initialConfig.setNumTargetFeatures(250)
    feature_tracker.initialConfig.setMotionEstimator(True)
    mono_l.out.link(feature_tracker.inputImage)

    xout_ft = pipeline.create(dai.node.XLinkOut)
    xout_ft.setStreamName("features")
    feature_tracker.outputFeatures.link(xout_ft.input)



    pipeline.setXLinkChunkSize(0)

    # 5. Onboard Bosch BNO086 9-Axis IMU (400 Hz)
    imu = pipeline.create(dai.node.IMU)
    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_CALIBRATED, 400)
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 400)
    imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, 400)
    imu.setBatchReportThreshold(5)
    imu.setMaxBatchReports(20)

    xout_imu = pipeline.create(dai.node.XLinkOut)
    xout_imu.setStreamName("imu")
    imu.out.link(xout_imu.input)

    return pipeline


def main():
    global RUNNING
    parser = argparse.ArgumentParser(description="Autonomous Drone Navigation Daemon")
    parser.add_argument("--test", action="store_true", help="Run in test mode for 15 seconds")
    parser.add_argument("--duration", type=int, default=0, help="Max run duration in seconds (0 = infinite)")
    parser.add_argument("--home-lat",  type=float, default=30.895560,
                        help="Home/EKF origin latitude  (degrees). Default: 30.895560")
    parser.add_argument("--home-lon",  type=float, default=74.613858,
                        help="Home/EKF origin longitude (degrees). Default: 74.613858")
    parser.add_argument("--home-alt",  type=float, default=0.0,
                        help="Home/EKF origin altitude MSL in meters (0.0 = auto-read ground baro)")
    parser.add_argument("--max-exposure", type=int, default=1500, help="Max auto exposure microseconds (1.5ms reduces motion blur)")
    parser.add_argument("--max-iso", type=int, default=1200, help="Maximum ISO for auto exposure")
    parser.add_argument("--marker-size", type=float, default=0.242, help="Outer black-border edge of the printed ArUco marker in meters (measured print: 0.242 m)")
    parser.add_argument("--no-prec-loiter", dest="prec_loiter", action="store_false", default=True, help="Do not auto-enable ArduPilot Precision Loiter over ArUco #0 in LOITER")
    parser.add_argument("--save-images", action="store_true", default=True, help="Save camera and ArUco detection frames to disk (default: True)")
    parser.add_argument("--port", type=str, default="/dev/ttyTHS1", help="Primary serial/UART port for CUAV (default: /dev/ttyTHS1)")
    parser.add_argument("--no-save-images", dest="save_images", action="store_false", help="Disable saving images")
    parser.add_argument("--rc-drift-report", action="store_true", help="Enable RC input drift analysis reporting")
    parser.add_argument("--outdoor-gps", action="store_true", default=False, help="Enable automatic GPS switching outdoors if GPS fix is strong")
    args = parser.parse_args()

    setup_logging(test_mode=args.test)

    # Base directory for saved flight recordings
    images_base_dir = IMAGES_DIR  # NVMe SSD path
    flight_record_dir = None
    flight_img_seq = 0
    last_armed_state = False
    last_armed_true_time = 0.0
    if args.save_images:
        logging.info(f"[+] Smart Flight Recording READY: Images will record to {images_base_dir} only when ARMED.")

    # Initialize CUAV X7 Pro MAVLink client in background thread
    logging.info(f"[*] Initializing CUAV X7 Pro MAVLink Bridge on {args.port}...")
    cuav = CuavMavlinkClient(preferred_port=args.port)
    global _cuav_ref
    _cuav_ref = cuav  # F-14: expose to signal handler for emergency LAND

    if args.home_lat is not None and args.home_lon is not None:
        logging.info(f"[+] Custom Home Coordinates loaded: {args.home_lat:.7f}, {args.home_lon:.7f} (Alt: {args.home_alt:.1f}m)")
        cuav.gps_lat = float(args.home_lat)
        cuav.gps_lon = float(args.home_lon)
        if args.home_alt != 0.0:
            cuav.gps_alt = float(args.home_alt)

    # ArUco Marker Detector Setup (DICT_4X4_50, Marker ID 0, 25cm landing pad)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    aruco_params = aruco.DetectorParameters()
    aruco_params.adaptiveThreshWinSizeMin = 3
    aruco_params.adaptiveThreshWinSizeMax = 23
    aruco_params.adaptiveThreshWinSizeStep = 10
    aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_NONE
    aruco_detector = aruco.ArucoDetector(aruco_dict, aruco_params)
    marker_size_m = float(args.marker_size) # Default 0.25m (25cm) target pad
    aruco_obj_pts = np.array([
        [-marker_size_m/2,  marker_size_m/2, 0],
        [ marker_size_m/2,  marker_size_m/2, 0],
        [ marker_size_m/2, -marker_size_m/2, 0],
        [-marker_size_m/2, -marker_size_m/2, 0]
    ], dtype=np.float32)

    # OV9282 Camera Intrinsics (640x400 default)
    focal_length_px = 396.0
    cx_f, cy_f = 320.0, 200.0
    K = np.array([
        [focal_length_px, 0.0, cx_f],
        [0.0, focal_length_px, cy_f],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    # Connect to OAK-D Pro camera with retry loop (up to 30 seconds for cold boot)
    pipeline = build_oak_pipeline(args.max_exposure, args.max_iso)
    device = None
    max_retries = 25
    for attempt in range(1, max_retries + 1):
        if not RUNNING:
            break
        try:
            logging.info(f"[*] Connecting to OAK-D Pro camera (attempt {attempt}/{max_retries})...")
            device = dai.Device(pipeline)
            logging.info(f"[+] SUCCESS: Connected to OAK-D Pro camera! Negotiated USB Speed: {device.getUsbSpeed().name}")
            break
        except Exception as e:
            logging.warning(f"[-] Waiting for OAK-D Pro camera: {e}")
            time.sleep(1.2)

    if device is None:
        logging.error("[!] Could not connect to OAK-D Pro camera after retries. Exiting.")
        cuav.close()
        sys.exit(1)

    with device:
        # 1. Load exact factory calibration matrix and distortion coefficients from OAK-D Pro EEPROM (640x400)
        try:
            calib = device.readCalibration()
            eeprom_K = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, 640, 400), dtype=np.float32)
            eeprom_dist = np.array(calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B), dtype=np.float32)
            if eeprom_K[0, 0] > 50.0:
                K = eeprom_K
                dist_coeffs = eeprom_dist.reshape(-1, 1)
                focal_length_px = float(K[0, 0])
                cx_f = float(K[0, 2])
                cy_f = float(K[1, 2])
                logging.info(f"[+] OAK-D EEPROM Calibration Loaded (640x400): fx={focal_length_px:.1f}, fy={K[1,1]:.1f}, cx={cx_f:.1f}, cy={cy_f:.1f}")
        except Exception as e:
            logging.warning(f"[-] Could not load EEPROM calibration ({e}), using defaults")

        # Floodlight only. Laser dots ride with the airframe and look static to FeatureTracker
        # (false zero velocity). Diffuse IR texture is what optical flow needs.
        try:
            device.setIrLaserDotProjectorIntensity(0.0)
            device.setIrFloodLightIntensity(0.7)
            logging.info("[+] OAK-D Pro IR: laser OFF, floodlight 700 mA (sharper flow, no speckle)")
        except Exception:
            try:
                device.setIrLaserDotProjectorBrightness(0)
                device.setIrFloodLightBrightness(700)
                logging.info("[+] OAK-D Pro IR: laser OFF, floodlight 700 mA (sharper flow, no speckle)")
            except Exception as e:
                logging.warning(f"[-] Active IR warning: {e}")

        # Queues for 640x400 Mono @ 60 FPS, 640x400 Stereo Depth & 400 Hz IMU (RGB OFF)
        q_mono = device.getOutputQueue(name="mono", maxSize=2, blocking=False)
        q_depth = None  # StereoDepth removed — TF03 LiDAR provides hardware altitude
        q_rgb = None
        q_imu = device.getOutputQueue(name="imu", maxSize=20, blocking=False)
        q_features = device.getOutputQueue(name="features", maxSize=4, blocking=False)
        ft_prev_features = {}  # id -> (x,y) from previous frame
        logging.info("[+] OAK-D FeatureTracker @ 120 FPS (Myriad-X on-chip) ACTIVE")
        image_logger = AsyncImageLogger(max_queue=120)
        video_writer = None   # MP4 VideoWriter (opened when ARMED)
        VIDEO_FPS = 30.0      # MP4 output FPS (every 4th frame at 120 FPS)
        latest_rgb_data = None

        # State Variables in Earth Local NED Frame (North, East, Down)
        pos_north = 0.0 # NED North in meters (+X)
        pos_east = 0.0  # NED East in meters (+Y)
        current_altitude_m = 0.00 # Drone Flight Altitude AGL in meters (0.00m on ground, NED Down = -current_altitude_m)
        raw_depth_m = 0.13
        depth_pts = 0
        last_valid_depth_time = 0.0
        total_traveled = 0.0
        max_outbound = 0.0
        max_altitude = 0.00
        live_speed = 0.0
        roll_deg = 0.0
        pitch_deg = 0.0
        yaw_deg = 0.0
        oak_roll_deg = 0.0
        oak_pitch_deg = 0.0
        oak_yaw_deg = 0.0
        prev_roll_rad = None
        prev_pitch_rad = None
        # Frame-time derotation: (host-synced ts s, roll rad, pitch rad) per OAK rotation-vector sample.
        imu_att_hist = deque(maxlen=128)
        # Measured on this OAK @100 FPS: IMU lands 1-12 ms after its frame; 15 ms brackets 98.5% of frames.
        IMU_SYNC_WAIT_S = 0.015
        # Change F: tilt between frames from integrated gyro. The rotation vector is smoothed and
        # under-reads fast tilt (flight fits: 525-680 px/rad vs f=396). (host ts s, gx, gy) rad/s.
        gyro_hist = deque(maxlen=512)
        prev_der_ts = None
        # Gyro axis/sign is verified at runtime against rotation-vector deltas before it is used.
        tilt_val_pairs = deque(maxlen=60)       # (gyro_x_sum, gyro_y_sum, rv_roll_sum, rv_pitch_sum)
        tilt_win = [0.0, 0.0, 0.0, 0.0, 0.0]    # gx, gy, rv_roll, rv_pitch, seconds
        gyro_derot = {"ok": False, "sign_r": 1.0, "sign_p": 1.0, "verdict": None}

        def drain_imu():
            nonlocal oak_roll_deg, oak_pitch_deg, oak_yaw_deg
            in_imu = q_imu.tryGet()
            if in_imu is None:
                return
            for packet in in_imu.packets:
                rv = packet.rotationVector
                q_i, q_j, q_k, q_r = rv.i, rv.j, rv.k, rv.real
                sinr_cosp = 2 * (q_r * q_i + q_j * q_k)
                cosr_cosp = 1 - 2 * (q_i * q_i + q_j * q_j)
                oak_roll_deg = math.atan2(sinr_cosp, cosr_cosp) * 57.2958

                sinp = 2 * (q_r * q_j - q_k * q_i)
                if abs(sinp) >= 1:
                    oak_pitch_deg = math.copysign(90.0, sinp)
                else:
                    oak_pitch_deg = math.asin(sinp) * 57.2958

                # 100 Hz Onboard IMU Yaw: atan2(2*(qr*qk + qi*qj), 1 - 2*(qj^2 + qk^2))
                siny_cosp = 2 * (q_r * q_k + q_i * q_j)
                cosy_cosp = 1 - 2 * (q_j * q_j + q_k * q_k)
                oak_yaw_deg = math.atan2(siny_cosp, cosy_cosp) * 57.2958
                try:
                    imu_att_hist.append((rv.getTimestamp().total_seconds(),
                                         math.radians(oak_roll_deg), math.radians(oak_pitch_deg)))
                except Exception:
                    pass
                try:
                    g = packet.gyroscope
                    gyro_hist.append((g.getTimestamp().total_seconds(), float(g.x), float(g.y)))
                except Exception:
                    pass

        def attitude_at(t):
            """(roll, pitch) rad linearly interpolated at host-synced time t, and whether t was bracketed."""
            if not imu_att_hist:
                return None, False
            newest = imu_att_hist[-1]
            if t >= newest[0]:
                return (newest[1], newest[2]), t == newest[0]
            if t <= imu_att_hist[0][0]:
                return (imu_att_hist[0][1], imu_att_hist[0][2]), False
            for i in range(len(imu_att_hist) - 1, 0, -1):
                t0, r0, p0 = imu_att_hist[i - 1]
                t1, r1, p1 = imu_att_hist[i]
                if t0 <= t <= t1:
                    a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
                    dr = (r1 - r0 + math.pi) % (2.0 * math.pi) - math.pi
                    return (r0 + a * dr, p0 + a * (p1 - p0)), True
            return (newest[1], newest[2]), False

        def gyro_tilt_delta(t0, t1):
            """Integrate gyro x/y (rad) over [t0, t1] (host-synced s); None if samples do not cover it."""
            if t1 <= t0 or len(gyro_hist) < 2 or gyro_hist[0][0] > t0 or gyro_hist[-1][0] < t1:
                return None
            sx = sy = 0.0
            prev = None
            for s in gyro_hist:
                if prev is not None and prev[0] >= t1:
                    break
                if prev is not None and s[0] > t0:
                    ta, tb = prev[0], s[0]
                    if tb - ta > 0.02:
                        return None
                    a0, a1 = max(ta, t0), min(tb, t1)
                    if a1 > a0 and tb > ta:
                        k0, k1 = (a0 - ta) / (tb - ta), (a1 - ta) / (tb - ta)
                        sx += 0.5 * ((prev[1] + (s[1] - prev[1]) * k0) + (prev[1] + (s[1] - prev[1]) * k1)) * (a1 - a0)
                        sy += 0.5 * ((prev[2] + (s[2] - prev[2]) * k0) + (prev[2] + (s[2] - prev[2]) * k1)) * (a1 - a0)
                prev = s
            return sx, sy

        def update_gyro_validation(gdx, gdy, rv_dr, rv_dp, dt_s):
            """Compare ~0.4 s windows of gyro vs rotation-vector tilt. Gyro derotation turns ON only when
            both axes agree (>=15 moving windows, |corr|>=0.85, 0.7<=|slope|<=2.0) and OFF if agreement is lost."""
            tilt_win[0] += gdx
            tilt_win[1] += gdy
            tilt_win[2] += rv_dr
            tilt_win[3] += rv_dp
            tilt_win[4] += dt_s
            if tilt_win[4] < 0.4:
                return
            tilt_val_pairs.append(tuple(tilt_win[:4]))
            tilt_win[:] = [0.0, 0.0, 0.0, 0.0, 0.0]
            verdict = []
            for gi, ri in ((0, 2), (1, 3)):
                pts = [(p[gi], p[ri]) for p in tilt_val_pairs if abs(p[ri]) > 0.005]
                if len(pts) < 15:
                    return
                gv = np.array([p[0] for p in pts])
                rvv = np.array([p[1] for p in pts])
                verdict.append((float(np.dot(gv, rvv) / np.dot(rvv, rvv)), float(np.corrcoef(gv, rvv)[0, 1])))
            gyro_derot["verdict"] = verdict
            agree = all(abs(c) >= 0.85 and 0.7 <= abs(sl) <= 2.0 and np.sign(sl) == np.sign(c) for sl, c in verdict)
            if agree and not gyro_derot["ok"]:
                gyro_derot.update(ok=True, sign_r=float(np.sign(verdict[0][0])), sign_p=float(np.sign(verdict[1][0])))
                logging.info(f"[+] Gyro tilt derotation ON: roll slope {verdict[0][0]:+.2f} corr {verdict[0][1]:+.2f} | "
                             f"pitch slope {verdict[1][0]:+.2f} corr {verdict[1][1]:+.2f}")
                cuav.send_statustext(">>> TILT: GYRO DEROT ON", severity=6)
            elif gyro_derot["ok"]:
                signs = (gyro_derot["sign_r"], gyro_derot["sign_p"])
                if not all(abs(c) >= 0.6 and np.sign(sl) == s for (sl, c), s in zip(verdict, signs)):
                    gyro_derot["ok"] = False
                    logging.warning(f"[!] Gyro tilt derotation OFF (agreement lost {verdict}) — using rotation vector")
                    cuav.send_statustext(">>> TILT: GYRO DEROT OFF", severity=4)

        last_aruco_lock_time = 0.0
        last_aruco_offset = (0.0, 0.0)
        last_aruco_z = 0.0
        last_aruco_ang_x = 0.0
        last_aruco_ang_y = 0.0
        last_aruco_size_rad = 0.0
        aruco_hits = deque(maxlen=128)   # times of valid ArUco #0 PnP solutions
        ARUCO_CONFIRM_HITS = 3           # within 0.5 s before any LANDING_TARGET is sent
        PREC_LOITER_MIN_HITS = 8         # within 1.0 s before Precision Loiter is enabled
        PLND_LAG_S = 0.03                # measured frame->wire 18.7 ms median / 28.8 p95, + serial/FC ~5 ms
        aruco_target_sent = 0
        t_mono_host = None
        ARUCO_LOST_S = 1.5               # no valid PnP detection for this long -> LOST
        ARUCO_NOTIFY_GAP_S = 1.0         # min gap between lock/lost RC messages
        aruco_notify = {"state": "lost", "sent": "lost", "since": 0.0, "held": 0.0, "last_tx": 0.0}

        def aruco_notice(now_t, hits, st, can_send):
            """Track ArUco #0 LOCK/LOST and return the state to announce ("locked"/"lost") or None.
            LOCK = >= ARUCO_CONFIRM_HITS valid detections within 0.5 s (same gate as LANDING_TARGET).
            LOST = no valid detection for ARUCO_LOST_S. Announcements are >= ARUCO_NOTIFY_GAP_S apart;
            a change during that gap (or while the FC link is down) is announced afterwards."""
            last_hit = hits[-1] if hits else None
            if st["state"] == "lost" and sum(1 for t in hits if now_t - t <= 0.5) >= ARUCO_CONFIRM_HITS:
                st["state"], st["since"] = "locked", now_t
                logging.info("[+] ArUco #0 LOCK")
            elif st["state"] == "locked" and (last_hit is None or now_t - last_hit > ARUCO_LOST_S):
                st["state"], st["held"], st["since"] = "lost", now_t - st["since"], now_t
                logging.info(f"[-] ArUco #0 LOST (held {st['held']:.1f} s)")
            if (not can_send) or st["state"] == st["sent"] or now_t - st["last_tx"] < ARUCO_NOTIFY_GAP_S:
                return None
            st["sent"], st["last_tx"] = st["state"], now_t
            return st["state"]
        last_video_time = time.time()
        last_summary_write_time = time.time()
        last_badge_time = 0.0
        prev_mono = None
        prev_pts = None
        ft_prev_seq = None
        total_dropped_frames = 0
        prev_hw_ts = None

        raw_vel_north = 0.0
        raw_vel_east = 0.0
        filtered_vel_north = 0.0
        filtered_vel_east = 0.0
        live_fps = 30.0
        transport_ms = 0.0
        proc_ms = 0.0
        send_ms = 0.0
        total_latency_ms = 0.0
        vel_log_path = os.path.join(LOG_DIR, "velocity_audit_30hz.csv")
        # F-12: Use 64KB memory buffer instead of line-buffered (buffering=1) to prevent
        #        30 Hz disk flushes from stalling the VIO loop on slow eMMC storage.
        vel_log_file = open(vel_log_path, "w", buffering=65536)
        vel_log_file.write("timestamp,delta_north,delta_east,dt,raw_vel_north,raw_vel_east,filtered_vel_north,filtered_vel_east,live_speed,altitude,roll,pitch,yaw,camera_fps\n")

        frame_count = 0
        vio_sent_count = 0
        aruco_detect_count = 0
        aruco_locked = False
        is_airborne = False
        last_ground_yaw_deg = 0.0
        on_ground = True
        height_source = "GROUND_GATE"
        fps = 30.0
        last_status_log_time = time.time()
        last_nav_eval_time = 0.0
        current_nav_mode = 'VIO' # Defaults to pure VIO until clean GPS confirmed
        is_airborne = False
        height_source = "INIT"
        tf03_h = 0.00
        tf03_age = 999.0
        tf03_valid = False
        start_time = time.time()
        last_f04_log = 0.0
        _f04_paused = False
        _vio_blind = True
        _vio_fresh = 0
        VIO_FRESH_FRAMES = 5
        VIO_TRUST_HOLD_S = 2.5  # skip takeoff 1–2s blur before fusing XY
        _vio_clear_since = 0.0
        _dbg_last_yaw = None
        _dbg_last_att_src = ""
        _dbg_att_log_t = 0.0
        _dbg_f04_log_t = 0.0
        _dbg_conn_t = 0.0

        logging.info("[+] Feature ROI mask OFF (camera under battery, full 640x400 floor FOV).")

        # Configure High-Speed 90Hz VIO & Instant Loiter Braking on CUAV X7 Pro
        cuav_params_configured = False
        def apply_cuav_parameters():
            nonlocal cuav_params_configured
            # Never (re)write FC config in flight, e.g. after a MAVLink reconnect while armed.
            if not cuav.is_connected or not cuav.armed_state_known or cuav.is_armed:
                return False
            try:
                cuav.set_parameter('VISO_DELAY_MS', 90.0)
                cuav.set_parameter('LOIT_BRK_DELAY', 0.10)
                cuav.set_parameter('LOIT_BRK_ACCEL', 25.0)
                # Static EKF sources, written only while disarmed; never switched in flight.
                src_ok = all([
                    cuav.set_parameter('EK3_SRC1_POSXY', 6.0),
                    cuav.set_parameter('EK3_SRC1_VELXY', 6.0),
                    cuav.set_parameter('EK3_SRC1_YAW', 1.0),
                ])
                cuav._posxy_static_ok = bool(src_ok)
                if not src_ok:
                    logging.error("[!] EK3_SRC1 POSXY=6 / VELXY=6 / YAW=1 not confirmed by FC")
                cuav.set_parameter('EK3_SRC1_VELZ', 0.0)
                cuav.set_parameter('EK3_SRC1_POSZ', 2.0)
                cuav.set_parameter('VISO_TYPE', 1.0)
                cuav.set_parameter('VISO_QUAL_MIN', 0.0)  # accept unknown quality from MAVLink1 VPE
                viso_now = cuav.get_parameter('VISO_TYPE')
                logging.info(f"[+] VISO_TYPE readback={viso_now} (1=MAVLink). Change requires CUAV reboot once.")
                if viso_now is None or abs(float(viso_now) - 1.0) > 0.1:
                    logging.error("[!] VISO_TYPE is not 1 — EKF will ignore VISION_POSITION_ESTIMATE")
                    cuav.send_statustext("VISO_TYPE FAIL | REBOOT CUAV", severity=2)
                else:
                    cuav.send_statustext("VISO_TYPE=1 | REBOOT CUAV ONCE", severity=6)
                cuav.set_parameter('AHRS_GPS_USE', 0.0)
                cuav.set_parameter('TERRAIN_ENABLE', 0)
                cuav.set_parameter('FS_EKF_ACTION', 2.0)  # 2 = AltHold if mode needs position (0 = report only)
                cuav.set_parameter('FS_EKF_THRESH', 1.20)
                # EK3_*_I_GATE is percent-of-sigma (range 100–1000). Default 500 = 5σ.
                gate_ok = (
                    cuav.set_parameter('EK3_VEL_I_GATE', 300.0)
                    and cuav.set_parameter('EK3_POS_I_GATE', 300.0)
                )
                if not gate_ok:
                    logging.warning("[!] EK3 VEL/POS I_GATE=300 not confirmed on FC")
                plnd_ok = (
                    cuav.set_parameter('PLND_ENABLED', 1.0)
                    and cuav.set_parameter('PLND_TYPE', 1.0)
                    and cuav.set_parameter('PLND_EST_TYPE', 1.0)
                )
                # PLND_CAM_ORIENT does not exist on Copter 4.7 (it only burned a 2.5 s timeout).
                plnd_ok = (
                    plnd_ok
                    and cuav.set_parameter('PLND_ORIENT', 25.0)
                    and cuav.set_parameter('PLND_LAG', PLND_LAG_S)
                )
                # Fence demands a position estimate. Indoor GNSS-denied: fence off.
                # Do NOT zero ARMING_CHECK — that hides compass/IMU/battery faults.
                cuav.set_parameter('FENCE_ENABLE', 0.0)
                home_lat = float(getattr(args, 'home_lat', 30.895560) or 30.895560)
                home_lon = float(getattr(args, 'home_lon', 74.613858) or 74.613858)
                origin_ok = cuav.set_ekf_origin(lat=home_lat, lon=home_lon)
                if origin_ok:
                    logging.info(
                        f"[+] EKF origin set {home_lat:.6f},{home_lon:.6f} "
                        "(anchor only — not GPS)"
                    )
                    cuav.send_statustext("EKF ORIGIN OK | ARM ALT_HOLD", severity=6)
                else:
                    logging.warning("[!] EKF origin NOT set — LOITER will keep Need Position Estimate")
                cuav_params_configured = True
                if plnd_ok:
                    logging.info("[+] CUAV EKF3/VIO: static POSXY=6 VELXY=6 YAW=1 GATE=300 VISO=1 (no in-flight source writes)")
                else:
                    logging.warning("[!] CUAV EKF OK but PLND params not confirmed — hover-on-marker may not engage")
                return True
            except Exception as e:
                logging.warning(f"[-] CUAV parameter configuration warning: {e}")
                return False

        if cuav.is_connected:
            apply_cuav_parameters()

        logging.info("[+] Autonomous Navigation Loop ACTIVE. Streaming to CUAV X7 Pro (Earth NED Frame)...")

        while RUNNING:
            now = time.time()
            if cuav.is_connected and not cuav_params_configured:
                apply_cuav_parameters()
            elif not cuav.is_connected:
                cuav_params_configured = False
                # #region agent log
                try:
                    if (now - _dbg_conn_t) > 5.0:
                        _dbg_conn_t = now
                        import json as _json
                        with open("/home/abc/SSD/DRONE/.cursor/debug-153c11.log", "a") as _df:
                            _df.write(_json.dumps({"sessionId":"153c11","runId":"post-fix","hypothesisId":"F","location":"flight_daemon.py:cuav_wait","message":"cuav_disconnected","data":{"oak_yaw":round(oak_yaw_deg,2)},"timestamp":int(time.time()*1000)})+"\n")
                except Exception:
                    pass
                # #endregion
            if args.test and (now - start_time >= 15.0):
                logging.info("[*] Test mode completed (15s elapsed).")
                break
            if args.duration > 0 and (now - start_time >= args.duration):
                logging.info(f"[*] Duration limit ({args.duration}s) reached.")
                break

            # 1. Update IMU Attitude (100 Hz) and CUAV Compass Heading
            drain_imu()

            # Dual-Source Attitude & Heading Fusion:
            # Priority 1: CUAV Compass/EKF Yaw if fresh (within 0.5s)
            # Priority 2: OAK-D 100 Hz Hardware IMU Yaw (Zero latency, never stuck at 0.00°)
            cuav_att_age = now - getattr(cuav, 'last_attitude_time', 0.0)
            if cuav.is_connected and (cuav_att_age <= 0.5):
                roll_deg = cuav.roll_deg
                pitch_deg = cuav.pitch_deg
                yaw_deg = cuav.yaw_deg
                _att_src = "CUAV"
            else:
                roll_deg = oak_roll_deg
                pitch_deg = oak_pitch_deg
                yaw_deg = cuav.yaw_deg  # no OAK yaw fallback: OAK yaw is ~170° off airframe heading
                _att_src = "OAK"
            # #region agent log
            try:
                _yaw_jump = None if _dbg_last_yaw is None else (yaw_deg - _dbg_last_yaw)
                _do_att = (_yaw_jump is not None and abs(_yaw_jump) > 8.0) or _att_src != _dbg_last_att_src or (is_airborne and (now - _dbg_att_log_t) > 0.5)
                if _do_att:
                    _dbg_att_log_t = now
                    import json as _json
                    with open("/home/abc/SSD/DRONE/.cursor/debug-153c11.log", "a") as _df:
                        _df.write(_json.dumps({"sessionId":"153c11","hypothesisId":"C","location":"flight_daemon.py:attitude","message":"yaw_source","data":{"src":_att_src,"cuav_age_s":round(cuav_att_age,3),"yaw":round(yaw_deg,2),"oak_yaw":round(oak_yaw_deg,2),"yaw_ddeg":None if _yaw_jump is None else round(_yaw_jump,2),"roll":round(roll_deg,2),"pitch":round(pitch_deg,2),"mode":getattr(cuav,"flight_mode",""),"armed":bool(cuav.is_armed),"airborne":bool(is_airborne)},"timestamp":int(time.time()*1000)})+"\n")
                _dbg_last_yaw = yaw_deg
                _dbg_last_att_src = _att_src
            except Exception:
                pass
            # #endregion

            # 2. Check Ground Altitude with Hardware TF03 LiDAR (Primary) & Autopilot Barometer (Fallback)
            tf03_h, tf03_age, tf03_valid = cuav.get_tf03_height(max_age_s=0.25)
            # Subtract physical mounting clearance (0.11m = 11cm baseline)
            drone_agl_from_lidar = max(0.0, tf03_h - cuav.lidar_ground_clearance_m)
            was_airborne = is_airborne

            if tf03_valid:
                # Ground threshold: 6cm above landing gear, or disarmed on ground up to 14cm
                on_ground = (drone_agl_from_lidar <= 0.06) or (not cuav.is_armed and drone_agl_from_lidar <= 0.14)
            else:
                on_ground = (not cuav.is_armed) and (getattr(cuav, 'rel_alt_m', 0.0) <= 0.20)

            if on_ground and not is_airborne:
                current_altitude_m = 0.00
                height_source = "GROUND_GATE"
                is_airborne = False
            elif tf03_valid and drone_agl_from_lidar > 0.06:
                current_altitude_m = float(np.clip(drone_agl_from_lidar, 0.0, 50.0))
                height_source = f"TF03({getattr(cuav, 'tf03_msg_type', 'LIDAR')})"
                # Hysteresis latch: once airborne, stay airborne while armed and >0.10m
                is_airborne = (cuav.is_armed and current_altitude_m >= 0.18) or (current_altitude_m >= 0.30) or (is_airborne and cuav.is_armed and current_altitude_m >= 0.10)
            elif cuav.is_connected and cuav.is_armed and getattr(cuav, 'rel_alt_m', 0.0) > 0.18:
                baro_agl = max(0.0, cuav.rel_alt_m)
                current_altitude_m = float(np.clip(0.85 * current_altitude_m + 0.15 * baro_agl, 0.0, 50.0))
                height_source = "BARO_REL"
                is_airborne = True
            else:
                height_source = "HOLD"
                is_airborne = is_airborne and cuav.is_armed

            if (not was_airborne) and is_airborne:
                dyaw = ((yaw_deg - last_ground_yaw_deg + 180.0) % 360.0) - 180.0
                if abs(dyaw) >= 20.0:
                    logging.warning(
                        f"[!] Compass/yaw swing on lift-off: ground {last_ground_yaw_deg:+.1f}° → air {yaw_deg:+.1f}° "
                        f"(Δ{dyaw:+.1f}°). Takeoff from wood/non-magnetic pad; VIO NED uses this heading."
                    )
            if not is_airborne:
                last_ground_yaw_deg = yaw_deg

            # F-07: Zero odom only while armed on the ground (motor vibration).
            # Disarmed slides are used to verify flow sign/scale.
            allow_flow = bool(is_airborne or (not cuav.is_armed))
            if on_ground and cuav.is_armed and not is_airborne:
                delta_north = 0.0
                delta_east = 0.0
                raw_vel_north = 0.0
                raw_vel_east = 0.0
                filtered_vel_north = 0.0
                filtered_vel_east = 0.0
                pos_north = 0.0
                pos_east = 0.0
                live_speed = 0.0

            # Blind-zone gate before flow integrate: never bank 3cm climb into the filter.
            if tf03_valid and tf03_h >= 0.50 and not on_ground:
                if _vio_blind:
                    cuav.reset_counter = (cuav.reset_counter + 1) % 256
                    _vio_fresh = 0
                    _vio_clear_since = now
                    delta_north = 0.0
                    delta_east = 0.0
                    raw_vel_north = 0.0
                    raw_vel_east = 0.0
                    filtered_vel_north = 0.0
                    filtered_vel_east = 0.0
                    pos_north = 0.0
                    pos_east = 0.0
                    live_speed = 0.0
                    logging.info("[+] VIO warmup (TF03>=0.50m) — wipe XY, need 1.0s + 5 clean frames")
                _vio_blind = False
            elif (not tf03_valid) or tf03_h < 0.40 or on_ground:
                if not _vio_blind:
                    cuav.reset_counter = (cuav.reset_counter + 1) % 256
                    logging.info("[+] VIO trust OFF (blind zone / TF03<0.40m)")
                _vio_blind = True
                _vio_fresh = 0
                _vio_clear_since = 0.0
                delta_north = 0.0
                delta_east = 0.0
                raw_vel_north = 0.0
                raw_vel_east = 0.0
                filtered_vel_north = 0.0
                filtered_vel_east = 0.0
                pos_north = 0.0
                pos_east = 0.0
                live_speed = 0.0

            # 3. Get OV9282 Global Shutter Features & Frame

            in_ft = q_features.tryGet()
            if in_ft is not None:
                frame_count += 1
                t_host_rx = time.monotonic()
                t_frame_host = None
                try:
                    t_frame_host = in_ft.getTimestamp().total_seconds()
                    transport_ms = max(0.0, (t_host_rx - t_frame_host) * 1000.0)
                except Exception:
                    pass

                # Hardware Sequence & Frame Drop Detection
                try:
                    seq = in_ft.getSequenceNum()
                    if ft_prev_seq is not None:
                        seq_diff = seq - ft_prev_seq
                        if seq_diff > 2:
                            dropped = seq_diff - 1
                            total_dropped_frames += dropped
                            # Only wipe feature history on significant blackout (>20 frames / >0.2s)
                            # Minor drops preserve hardware feature IDs so VIO tracking does not black out (prevents F-04)
                            if seq_diff > 20:
                                ft_prev_features = {}
                                prev_pts = None
                                prev_roll_rad = None
                                prev_pitch_rad = None
                                prev_der_ts = None
                            logging.warning(f"[!] Frame drop detected: {dropped} frames lost (seq: {ft_prev_seq}->{seq})")
                    ft_prev_seq = seq
                except Exception:
                    pass
                t_proc_start = time.perf_counter()
                mono_f = None
                t_mono_host = None
                in_mono = q_mono.tryGet()
                if in_mono is not None:
                    mono_f = in_mono.getCvFrame()
                    try:
                        t_mono_host = in_mono.getTimestamp().total_seconds()
                    except Exception:
                        t_mono_host = None
                # Smart Armed-Only Flight Recording:
                # Automatically starts recording to a new folder when ARMED, stops when DISARMED (protects disk!)
                if args.save_images:
                    is_active = cuav.is_armed or args.test
                    if is_active:
                        last_armed_true_time = now
                        if not last_armed_state:
                            last_armed_state = True
                            flight_img_seq = 0
                            folder_prefix = "test" if args.test else "flight"
                            flight_record_dir = os.path.join(images_base_dir, datetime.now().strftime(f"{folder_prefix}_%Y%m%d_%H%M%S"))
                            os.makedirs(flight_record_dir, exist_ok=True)
                            logging.info(f"[+] DRONE ARMED: Starting flight recording -> {flight_record_dir}")
                            logging.info(f"[+] Armed Flight Recording: Saving 2 FPS images to {flight_record_dir}")
                    else:
                        # Debounce: wait 2.5 seconds of consistent disarm before closing folder
                        if last_armed_state and (now - last_armed_true_time > 2.5):
                            last_armed_state = False
                            logging.info(f"[*] DRONE DISARMED: Flight recording completed ({flight_img_seq} images) -> {flight_record_dir}")
                            pass
                            try:
                                os.sync()
                            except Exception:
                                pass
                            flight_record_dir = None

                # Save periodic mono & RGB frames (2 FPS / every 15 frames) ONLY while ARMED
                # Using decoupled AsyncImageLogger so disk I/O NEVER blocks the 30 Hz VIO loop
                pass

                if flight_record_dir and (frame_count % 15 == 0) and flight_img_seq < 1000:
                    flight_img_seq += 1
                    img_path = os.path.join(flight_record_dir, f"frame_{flight_img_seq:05d}.jpg")
                    image_logger.log_image(img_path, mono_f, is_bytes=False)

                # Hardware timestamp delta for dt (eliminates host OS thread jitter)
                hw_dt = None
                if t_frame_host is not None and prev_hw_ts is not None:
                    diff_ts = t_frame_host - prev_hw_ts
                    if 0.005 < diff_ts < 0.15:
                        hw_dt = diff_ts
                prev_hw_ts = t_frame_host
                dt = hw_dt if hw_dt is not None else max(0.01, min(0.15, now - last_video_time))
                last_video_time = now
                delta_north = 0.0
                delta_east = 0.0
                v_nudge_north = 0.0
                v_nudge_east = 0.0

                # 4. ArUco Precision Landing Detection (Marker ID 0)
                # Throttled to 10-15 Hz to conserve Jetson CPU and prevent camera queue backlog
                corners, ids = None, None
                is_landing_phase = (cuav.is_connected and getattr(cuav, 'flight_mode', '') == "LAND")
                aruco_stride = 3 if (is_landing_phase or (now - last_aruco_lock_time <= 1.0)) else 8
                if (frame_count % aruco_stride == 0) and mono_f is not None:
                    corners, ids, _ = aruco_detector.detectMarkers(mono_f)
                    if ids is not None and 0 in ids:
                        aruco_detect_count += 1
                        last_aruco_lock_time = now

                # Robust Debounce Hold: Keep aruco_locked TRUE for 0.6s after detection
                # This completely eliminates the locked/unlocked flickering on RC!
                aruco_locked = (now - last_aruco_lock_time <= 0.6)
                if ids is not None and 0 in ids:
                    idx = list(ids.ravel()).index(0)
                    m_corners = corners[idx].reshape(4, 2)

                    # Save annotated detection image when ArUco is locked (visual confirmation during flight)
                    if flight_record_dir and (aruco_detect_count % 5 == 1) and flight_img_seq < 1000:
                        flight_img_seq += 1
                        annotated = cv2.cvtColor(mono_f, cv2.COLOR_GRAY2BGR)
                        cv2.polylines(annotated, [m_corners.astype(np.int32)], True, (0, 255, 0), 2)
                        cv2.putText(annotated, f"ArUco #0 Locked ({aruco_detect_count})",
                                    (int(m_corners[0][0]), max(20, int(m_corners[0][1]) - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                        aruco_img_path = os.path.join(flight_record_dir, f"aruco_{flight_img_seq:05d}_ID0.jpg")
                        image_logger.log_image(aruco_img_path, annotated, is_bytes=False)

                    sides = [float(np.linalg.norm(m_corners[i] - m_corners[(i + 1) % 4])) for i in range(4)]
                    geom_ok = min(sides) >= 8.0 and max(sides) <= 2.5 * min(sides)
                    if geom_ok:
                        success, rvec, tvec = cv2.solvePnP(aruco_obj_pts, m_corners, K, dist_coeffs,
                                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    else:
                        success, rvec, tvec = False, None, None
                    if success:
                        t_x, t_y, t_z = float(tvec[0]), float(tvec[1]), float(tvec[2])
                        # Rotate body camera offset to Earth NED frame using yaw heading
                        yaw_rad = math.radians(yaw_deg)
                        cos_yaw = math.cos(yaw_rad)
                        sin_yaw = math.sin(yaw_rad)
                        # Correct Coordinate Frame:
                        # In downward camera:
                        # +Y_cam is towards tail -> if marker is at +Y_cam, drone is forward (+X_body)
                        # -X_cam is towards port -> if marker is at -X_cam, drone is right (+Y_body)
                        body_forward_m = t_y
                        body_right_m = -t_x
                        pad_offset_north = body_forward_m * cos_yaw - body_right_m * sin_yaw
                        pad_offset_east = body_forward_m * sin_yaw + body_right_m * cos_yaw
                        last_aruco_lock_time = now
                        last_aruco_offset = (pad_offset_north, pad_offset_east)
                        last_aruco_z = t_z
                        # Line of sight from the undistorted PnP translation. Camera frame: +x right,
                        # +y toward tail, +z down (image axes validated against EKF velocity).
                        # ArduPilot MAVLink backend: forward = -tan(angle_y), right = tan(angle_x).
                        last_aruco_ang_x = math.atan2(t_x, t_z)
                        last_aruco_ang_y = math.atan2(t_y, t_z)
                        side_px = float(np.mean(sides))
                        last_aruco_size_rad = math.atan2(side_px, focal_length_px)
                        aruco_los_dist = float(np.linalg.norm(tvec))
                        aruco_hits.append(now)
                        # Send only this fresh detection (never re-send stale angles), and only after the
                        # marker was seen repeatedly (rejects single-frame false positives).
                        if (cuav.is_connected and t_z > 0.08
                                and sum(1 for t in aruco_hits if now - t <= 0.5) >= ARUCO_CONFIRM_HITS):
                            if cuav.send_landing_target(
                                last_aruco_ang_x, last_aruco_ang_y, aruco_los_dist,
                                last_aruco_size_rad, last_aruco_size_rad,
                                time_usec=None if t_mono_host is None else int(t_mono_host * 1e6),
                            ):
                                aruco_target_sent += 1

                        # Use ArUco altitude ONLY as a fallback if TF03 LiDAR and Stereo Depth are unavailable
                        aruco_agl = max(0.0, t_z - 0.03)
                        if not tf03_valid and height_source in ["HOLD", "BARO_REL"]:
                            current_altitude_m = float(np.clip(0.80 * current_altitude_m + 0.20 * aruco_agl, 0.00, 15.00))
                            height_source = "ARUCO_Z"

                # 5. OAK-D Myriad-X FeatureTracker Optical Flow (Zero Jetson CPU!)
                if in_ft is not None:
                    cur_features = {f.id: (f.position.x, f.position.y) for f in in_ft.trackedFeatures}
                    common_ids = [k for k in cur_features if k in ft_prev_features]
                    if len(common_ids) >= 8:
                        old_pts = np.float32([ft_prev_features[k] for k in common_ids])
                        new_pts = np.float32([cur_features[k] for k in common_ids])
                        good_old = old_pts
                        good_new = new_pts
                    else:
                        good_old = np.empty((0, 2), dtype=np.float32)
                        good_new = np.empty((0, 2), dtype=np.float32)
                    ft_prev_features = cur_features
                else:
                    good_old = np.empty((0, 2), dtype=np.float32)
                    good_new = np.empty((0, 2), dtype=np.float32)

                if len(good_new) >= 8:
                    pts_old_c = good_old - [cx_f, cy_f]
                    pts_new_c = good_new - [cx_f, cy_f]

                    M, inliers = cv2.estimateAffinePartial2D(
                        pts_old_c, pts_new_c, method=cv2.RANSAC, ransacReprojThreshold=4.0
                    )
                    inlier_count = int(np.sum(inliers)) if inliers is not None else 0
                    inlier_ratio = (inlier_count / len(good_new)) if len(good_new) > 0 else 0.0

                    if M is not None and inlier_count >= 8 and inlier_ratio >= 0.20:
                        scale = float(np.linalg.norm(M[:2, 0]))
                        if 0.70 < scale < 1.30:
                            trans_dx = float(M[0, 2])
                            trans_dy = float(M[1, 2])

                            pitch_rad = math.radians(pitch_deg)
                            roll_rad = math.radians(roll_deg)
                            oak_pitch_rad = math.radians(oak_pitch_deg)
                            oak_roll_rad = math.radians(oak_roll_deg)

                            # Derotate with OAK 100 Hz IMU (camera timestamp domain), not 20 Hz CUAV ATTITUDE.
                            # Use attitude interpolated at this frame's timestamp, waiting briefly for the
                            # IMU sample that brackets it (latest-sample attitude lagged on ~45% of frames).
                            imu_bracketed = False
                            der_roll_rad, der_pitch_rad = oak_roll_rad, oak_pitch_rad
                            if t_frame_host is not None:
                                imu_wait_end = time.monotonic() + IMU_SYNC_WAIT_S
                                while ((not imu_att_hist or imu_att_hist[-1][0] < t_frame_host)
                                       and time.monotonic() < imu_wait_end):
                                    drain_imu()
                                    time.sleep(0.0005)
                                att_frame, imu_bracketed = attitude_at(t_frame_host)
                                if att_frame is not None:
                                    der_roll_rad, der_pitch_rad = att_frame
                            _tdx0, _tdy0 = trans_dx, trans_dy
                            _droll, _dpitch, _tsx, _tsy = 0.0, 0.0, 0.0, 0.0
                            _tilt_src, _gd = "none", None
                            if prev_roll_rad is not None and prev_pitch_rad is not None:
                                delta_roll = der_roll_rad - prev_roll_rad
                                delta_pitch = der_pitch_rad - prev_pitch_rad
                                _tilt_src = "rotvec"
                                if t_frame_host is not None and prev_der_ts is not None and imu_bracketed:
                                    _gd = gyro_tilt_delta(prev_der_ts, t_frame_host)
                                if _gd is not None:
                                    update_gyro_validation(_gd[0], _gd[1], delta_roll, delta_pitch,
                                                           t_frame_host - prev_der_ts)
                                    if gyro_derot["ok"]:
                                        delta_roll = gyro_derot["sign_r"] * _gd[0]
                                        delta_pitch = gyro_derot["sign_p"] * _gd[1]
                                        _tilt_src = "gyro"
                                else:
                                    tilt_win[:] = [0.0, 0.0, 0.0, 0.0, 0.0]
                                _droll, _dpitch = delta_roll, delta_pitch
                                if abs(delta_roll) < 0.20 and abs(delta_pitch) < 0.20:
                                    tilt_shift_x = -focal_length_px * math.tan(delta_roll)
                                    tilt_shift_y = focal_length_px * math.tan(delta_pitch)  # flight logs: ty ≈ +f·dPitch
                                    _tsx, _tsy = tilt_shift_x, tilt_shift_y
                                    trans_dx -= tilt_shift_x
                                    trans_dy -= tilt_shift_y
                            prev_roll_rad = der_roll_rad
                            prev_pitch_rad = der_pitch_rad
                            prev_der_ts = t_frame_host if imu_bracketed else None

                            pure_trans_dx = trans_dx if abs(trans_dx) >= 0.08 else 0.0
                            pure_trans_dy = trans_dy if abs(trans_dy) >= 0.08 else 0.0

                            cos_tilt = max(0.70, math.cos(oak_pitch_rad) * math.cos(oak_roll_rad))
                            cam_lever_m = max(0.0, cuav.camera_ground_clearance_m - cuav.lidar_ground_clearance_m)
                            optical_height_m = max(0.03, current_altitude_m * cos_tilt + cam_lever_m)
                            meter_per_px = optical_height_m / focal_length_px
                            dx_body = -(pure_trans_dx * meter_per_px)
                            dy_body = (pure_trans_dy * meter_per_px)
                            step_dist = math.sqrt(dx_body**2 + dy_body**2)
                            # #region agent log
                            try:
                                if frame_count % 10 == 0:
                                    import json as _json
                                    with open("/home/abc/SSD/DRONE/.cursor/debug-153c11.log", "a") as _df:
                                        _df.write(_json.dumps({"sessionId":"153c11","runId":"post-fix","hypothesisId":"G","location":"flight_daemon.py:affine","message":"raw_flow","data":{"allow":bool(allow_flow),"armed":bool(cuav.is_armed),"air":bool(is_airborne),"nfeat":int(len(good_new)),"inliers":int(inlier_count),"scale":round(scale,3),"tx":round(_tdx0,3),"ty":round(_tdy0,3),"step":round(step_dist,5),"h":round(optical_height_m,3)},"timestamp":int(time.time()*1000)})+"\n")
                            except Exception:
                                pass
                            # #endregion

                            if (not allow_flow) or _vio_blind:
                                dx_body = 0.0
                                dy_body = 0.0
                                step_dist = 0.0
                                delta_north = 0.0
                                delta_east = 0.0
                                pos_north = 0.0
                                pos_east = 0.0
                                live_speed = 0.0

                            # Marker centering is done by ArduPilot PrecLand / PrecLoiter from LANDING_TARGET.
                            # A fake vision-velocity nudge would corrupt EKF3 velocity fusion, so none is added.
                            v_nudge_north = 0.0
                            v_nudge_east  = 0.0

                            if allow_flow and (not _vio_blind) and 0.00005 < step_dist < 0.35:
                                # Rotate with the gyro-integrated flow heading, not the EKF yaw fed back.
                                flow_hdg = cuav.get_flow_heading_deg()
                                _yaw_flow = yaw_deg if flow_hdg is None else flow_hdg
                                yaw_rad = math.radians(_yaw_flow)
                                cos_psi = math.cos(yaw_rad)
                                sin_psi = math.sin(yaw_rad)

                                delta_north = dy_body * cos_psi - dx_body * sin_psi
                                delta_east = dy_body * sin_psi + dx_body * cos_psi

                                eff_delta_north = delta_north + (v_nudge_north * dt)
                                eff_delta_east  = delta_east  + (v_nudge_east  * dt)

                                pos_north += eff_delta_north
                                pos_east  += eff_delta_east
                                total_traveled += step_dist
                                live_speed = step_dist / dt
                                # #region agent log
                                try:
                                    if frame_count % 10 == 0:
                                        import json as _json
                                        with open("/home/abc/SSD/DRONE/.cursor/debug-153c11.log", "a") as _df:
                                            _df.write(_json.dumps({"sessionId":"153c11","runId":"post-fix","hypothesisId":"A,B,E","location":"flight_daemon.py:flow_ned","message":"flow_to_ned","data":{"dt":round(dt,4),"tx0":round(_tdx0,3),"ty0":round(_tdy0,3),"tsx":round(_tsx,3),"tsy":round(_tsy,3),"droll":round(_droll,5),"dpitch":round(_dpitch,5),"dx_b":round(dx_body,4),"dy_b":round(dy_body,4),"dN":round(eff_delta_north,4),"dE":round(eff_delta_east,4),"yaw":round(yaw_deg,2),"yaw_flow":round(_yaw_flow,2),"h_opt":round(optical_height_m,3),"mpp":round(meter_per_px,6),"imu_sync":bool(imu_bracketed),"tilt_src":_tilt_src,"gdr":None if _gd is None else round(_gd[0],5),"gdp":None if _gd is None else round(_gd[1],5),"nfeat":int(len(good_new)),"inliers":int(inlier_count),"mode":getattr(cuav,"flight_mode",""),"airborne":bool(is_airborne),"armed":bool(cuav.is_armed)},"timestamp":int(time.time()*1000)})+"\n")
                                except Exception:
                                    pass
                                # #endregion
                                if args.rc_drift_report:
                                    cuav.send_named_value("DRIFT_NORTH", delta_north)
                                    cuav.send_named_value("DRIFT_EAST", delta_east)

                            prev_pts = good_new.reshape(-1, 1, 2)
                        else:
                            prev_pts = None
                    else:
                        prev_pts = None
                else:
                    prev_pts = None

                # Feature replenishment handled in hardware at 120 FPS by OAK-D Myriad-X
                prev_mono = mono_f

                # 6. Velocity Estimation & Low-Pass Filter (Active 30Hz Feedback to EKF3)
                if (not allow_flow) or _vio_blind:
                    raw_vel_north = 0.0
                    raw_vel_east = 0.0
                    filtered_vel_north = 0.0
                    filtered_vel_east = 0.0
                else:
                    # Mathematically unified: velocity is exactly d(pos)/dt
                    # Full dynamic envelope: +/- 4.0 m/s raw & +/- 3.5 m/s filtered covers rapid stick maneuvers & braking flares
                    # Heavy smoothing (0.70/0.30) eliminates noise spikes and prevents toilet-bowling phase-shift
                    raw_vel_north = float(np.clip((delta_north / dt) + v_nudge_north, -4.0, 4.0))
                    raw_vel_east  = float(np.clip((delta_east / dt)  + v_nudge_east,  -4.0, 4.0))
                    filtered_vel_north = float(np.clip(0.70 * filtered_vel_north + 0.30 * raw_vel_north, -3.5, 3.5))
                    filtered_vel_east  = float(np.clip(0.70 * filtered_vel_east  + 0.30 * raw_vel_east,  -3.5, 3.5))
                live_fps = 1.0 / dt if dt > 0 else 30.0

                try:
                    vel_log_file.write(
                        f"{now:.4f},{delta_north:.6f},{delta_east:.6f},{dt:.4f},"
                        f"{raw_vel_north:.6f},{raw_vel_east:.6f},"
                        f"{filtered_vel_north:.6f},{filtered_vel_east:.6f},"
                        f"{live_speed:.6f},{current_altitude_m:.3f},"
                        f"{roll_deg:.2f},{pitch_deg:.2f},{yaw_deg:.2f},"
                        f"{live_fps:.1f}\n"
                    )
                except Exception:
                    pass

                t_proc_done = time.perf_counter()
                proc_ms = max(0.0, (t_proc_done - t_proc_start) * 1000.0)

                # Stream VISION_POSITION + SPEED. LOITER needs ExtNav position (POSXY=6).
                # Blind zone: freeze at 0,0 with high covariance so VisOdom stays healthy
                # without banking 3cm climb into the loiter target.
                tracking_ok = (prev_pts is not None and len(prev_pts) >= 8)
                if (not _vio_blind) and tracking_ok:
                    _vio_fresh += 1
                    if _vio_fresh == VIO_FRESH_FRAMES:
                        logging.info("[+] VIO trust warmup complete (5 clean frames)")
                else:
                    _vio_fresh = 0
                if cuav.is_connected and (frame_count % 2 == 0):
                    try:
                        if getattr(args, 'outdoor_gps', False) and not cuav.origin_set and vio_sent_count > 15:
                            if args.home_lat is not None and args.home_lon is not None:
                                cuav.set_home_position(args.home_lat, args.home_lon, args.home_alt if args.home_alt != 0.0 else None)
                            else:
                                cuav.set_ekf_origin()

                        held = (_vio_clear_since > 0.0) and ((now - _vio_clear_since) >= VIO_TRUST_HOLD_S)
                        flow_ok = tracking_ok and (not _vio_blind) and (_vio_fresh >= VIO_FRESH_FRAMES) and held
                        spd = math.hypot(filtered_vel_north, filtered_vel_east)
                        latched = bool(getattr(cuav, '_posxy_loiter', False) or getattr(cuav, '_ekf_xy_ok', False))
                        # After latch, stick fwd/back is allowed. Before latch, skip takeoff blur
                        # (2.5s above 0.5m) and do not fuse if already running away (>1.2 m/s).
                        ready_to_fuse = bool(
                            flow_ok
                            and current_altitude_m >= 1.00
                            and (latched or spd < 1.20)
                        )
                        trust_vision = flow_ok
                        if trust_vision:
                            _f04_paused = False
                            if latched or ready_to_fuse:
                                vel_cov = 0.25
                                pos_cov = 0.50
                            else:
                                vel_cov = 1.00
                                pos_cov = 5.00
                            vn, ve = filtered_vel_north, filtered_vel_east
                            px, py = pos_north, pos_east
                        else:
                            if getattr(cuav, '_vio_trust_announced', False):
                                cuav._vio_trust_announced = False
                                cuav._ekf_xy_ok = False
                                cuav._posxy_loiter = False
                                cuav.send_statustext(">>> VIO MUTED | WAIT TAKEOFF", severity=4)
                            if not _f04_paused:
                                cuav.reset_counter = (cuav.reset_counter + 1) % 256
                                _f04_paused = True
                            vel_cov = 5.0
                            pos_cov = 5.0
                            vn, ve = 0.0, 0.0
                            px, py = 0.0, 0.0
                            if is_airborne and (now - last_f04_log > 2.0):
                                last_f04_log = now
                                if _vio_blind:
                                    why = "blind-zone TF03"
                                elif not held:
                                    why = "need 2.5s above 0.50m (skip takeoff blur)"
                                elif _vio_fresh < VIO_FRESH_FRAMES:
                                    why = "warmup frames"
                                else:
                                    why = "weak track"
                                logging.warning(f"[!] VIO muted ({why}) — airborne: VIO packets suppressed")

                        # Airborne + untrusted: send nothing. A fake 0,0 / 0 m/s would be fused as truth.
                        suppress_vio = bool(is_airborne and not trust_vision)
                        sent_p = False
                        sent = False
                        if not suppress_vio:
                            sent_p = cuav.send_vision_estimate(
                                x=px, y=py, z=-float(current_altitude_m),
                                roll_rad=math.radians(roll_deg),
                                pitch_rad=math.radians(pitch_deg),
                                yaw_rad=math.radians(yaw_deg),
                                pos_cov=pos_cov,
                            )
                            sent = cuav.send_vision_speed(vx=vn, vy=ve, vz=0.0, vel_cov=vel_cov)
                        if sent or sent_p:
                            vio_sent_count += 1
                        if ready_to_fuse and (not getattr(cuav, '_vio_trust_announced', False)):
                            if not sent_p:
                                logging.error("[!] VISION_POSITION_ESTIMATE not actually sent — FC has no XY")
                                cuav.send_statustext(">>> VPE SEND FAIL | NO LOITER", severity=2)
                            else:
                                posxy_ok = bool(getattr(cuav, '_posxy_static_ok', False))  # static, confirmed while disarmed
                                cuav._posxy_loiter = bool(posxy_ok)
                                flags = int(getattr(cuav, 'ekf_status_flags', 0))
                                ekf_xy = bool(flags & (8 | 16))
                                logging.info(
                                    f"[+] VPE sent={sent_p} vel sent={sent} POSXY=6 confirmed={posxy_ok} "
                                    f"EKF flags=0x{flags:x} horiz_pos={ekf_xy}"
                                )
                                cuav._vio_trust_announced = True
                                if posxy_ok and ekf_xy:
                                    cuav._ekf_xy_ok = True
                                    cuav.send_statustext(">>> VIO TRUST ON | LOITER OK", severity=6)
                                elif posxy_ok:
                                    cuav._ekf_xy_ok = False
                                    cuav.send_statustext(">>> VIO SENT | WAIT EKF POS", severity=4)
                                    logging.warning("[!] Packets on wire but EKF horiz pos flag still off — PreArm stays")
                                else:
                                    cuav._ekf_xy_ok = False
                                    logging.error("[!] EK3_SRC1_POSXY=6 NOT confirmed")
                                    cuav.send_statustext(">>> POSXY=6 FAIL | NO LOITER", severity=2)
                        elif ready_to_fuse and sent_p and (not getattr(cuav, '_ekf_xy_ok', False)):
                            flags = int(getattr(cuav, 'ekf_status_flags', 0))
                            if flags & (8 | 16):
                                cuav._ekf_xy_ok = True
                                logging.info(f"[+] EKF horiz pos flag ON (0x{flags:x}) — LOITER allowed")
                                cuav.send_statustext(">>> VIO TRUST ON | LOITER OK", severity=6)
                        # LANDING_TARGET is sent once per fresh, confirmed ArUco detection (ArUco block above).
                        # #region agent log
                        try:
                            if (vio_sent_count % 8 == 0) or (not tracking_ok and (now - _dbg_f04_log_t) > 0.2):
                                if not tracking_ok:
                                    _dbg_f04_log_t = now
                                import json as _json
                                with open("/home/abc/SSD/DRONE/.cursor/debug-153c11.log", "a") as _df:
                                    _df.write(_json.dumps({"sessionId":"153c11","runId":"post-fix","hypothesisId":"D","location":"flight_daemon.py:vio_send","message":"vision_speed","data":{"sent":bool(sent),"ok":bool(tracking_ok),"n":round(pos_north,3),"e":round(pos_east,3),"vn":round(vn,3),"ve":round(ve,3),"vel_cov":vel_cov,"mode":getattr(cuav,"flight_mode",""),"posvar":round(getattr(cuav,"ekf_horiz_var",-1),3),"velvar":round(getattr(cuav,"ekf_vel_var",-1),3),"reset":int(getattr(cuav,"reset_counter",0)),"airborne":bool(is_airborne)},"timestamp":int(time.time()*1000)})+"\n")
                        except Exception:
                            pass
                        # #endregion
                    except Exception as vio_err:
                        logging.error(f"[!] VIO stream error: {vio_err}")

                t_send_done = time.perf_counter()
                send_ms = max(0.0, (t_send_done - t_proc_done) * 1000.0)
                total_latency_ms = transport_ms + proc_ms + send_ms

                # Stream RC display telemetry every 12 frames (~10 Hz) so UniRC shows Lat, Lon, GL, AGL, and Dist
                if cuav.is_connected and (frame_count % 12 == 0):
                    cur_lat = cuav.gps_lat + (pos_north / 111139.0)
                    cos_lat = math.cos(math.radians(cuav.gps_lat))
                    cur_lon = cuav.gps_lon + (pos_east / (111139.0 * cos_lat if abs(cos_lat) > 1e-4 else 111139.0))
                    cur_gl = cuav.baro_alt
                    cur_agl = current_altitude_m
                    cur_dist = math.sqrt(pos_north**2 + pos_east**2)
                    cuav.send_rc_display_telemetry(
                        lat=cur_lat,
                        lon=cur_lon,
                        gl_alt_m=cur_gl,
                        agl_alt_m=cur_agl,
                        dist_to_home_m=cur_dist,
                        speed_m_s=live_speed,
                        heading_deg=yaw_deg
                    )

                # Track distance metrics
                dist_h = math.sqrt(pos_north**2 + pos_east**2)
                if dist_h > max_outbound:
                    max_outbound = dist_h
                if current_altitude_m > max_altitude:
                    max_altitude = current_altitude_m

            # Periodic Status Log every 3 seconds
            if now - last_status_log_time >= 3.0:
                last_status_log_time = now
                fps = frame_count / 3.0
                frame_count = 0
                # Real-Time Anti-Jamming & Anti-Spoofing Diagnostic
                is_jammed, gps_diag = cuav.check_gps_jamming(camera_speed_m_s=live_speed)
                shield_badge = "[SHIELD: CAM VIO ACTIVE]" if is_jammed else "[GPS SECURE]"
                ekf_desc = (
                    f"PosVar: {cuav.ekf_horiz_var:.3f} | VelVar: {cuav.ekf_vel_var:.3f} | "
                    f"EKF:0x{int(getattr(cuav,'ekf_status_flags',0)):x}"
                ) if cuav.ekf_horiz_var < 50 else "Converging"
                cuav_str = f"CUAV: {cuav.port} ({cuav.flight_mode}) | {ekf_desc}" if cuav.is_connected else "CUAV: WAITING"
                tf03_volt = getattr(cuav, 'tf03_voltage', 0.0)
                stereo_str = f"Depth:{raw_depth_m:.2f}m({depth_pts}p)" if (now - last_valid_depth_time <= 0.5) else "Depth:N/A"
                tf03_log = f"TF03_H={tf03_h:.2f}m (V:{tf03_volt:.3f}V) {stereo_str} [{height_source}]"
                aruco_badge = f"[ARUCO#0 LOCKED dN:{last_aruco_offset[0]:+.2f}m dE:{last_aruco_offset[1]:+.2f}m]" if (now - last_aruco_lock_time <= 1.0) else "[ARUCO: SEARCHING]"
                vel_status = f"Vel: N:{filtered_vel_north:+.2f} E:{filtered_vel_east:+.2f}m/s"
                logging.info(
                    f"[VIO {fps:.1f} FPS | Drops:{total_dropped_frames}] Pos: N:{pos_north:+05.2f}m E:{pos_east:+05.2f}m Z:{current_altitude_m:04.2f}m (Yaw:{yaw_deg:+.1f}°) | "
                    f"{vel_status} | {aruco_badge} | {tf03_log} | "
                    f"Lat: {total_latency_ms:.1f}ms (xfer:{transport_ms:.1f}ms proc:{proc_ms:.1f}ms send:{send_ms:.1f}ms) | "
                    f"VIO Sent: {vio_sent_count} | {cuav_str} | {shield_badge} {gps_diag}"
                )

            # Automated Autonomous Navigation Source Switcher (Every 2.5s)
            if now - last_nav_eval_time >= 2.5:
                last_nav_eval_time = now
                if cuav.is_connected:
                    is_clean_gps = getattr(args, 'outdoor_gps', False) and (cuav.gps_fix_type >= 3 and cuav.gps_satellites >= 8 and cuav.gps_hacc <= 4.0 and not is_jammed)
                    if is_clean_gps and current_nav_mode != 'GPS':
                        ok1 = cuav.set_parameter('EK3_SRC1_POSXY', 3.0)
                        ok2 = cuav.set_parameter('EK3_SRC1_VELXY', 3.0)
                        ok3 = cuav.set_parameter('EK3_SRC1_VELZ', 3.0)
                        ok4 = cuav.set_parameter('AHRS_GPS_USE', 1.0)
                        # F-05: Only switch mode if FC confirmed all parameters
                        if ok1 and ok2 and ok3 and ok4:
                            current_nav_mode = 'GPS'
                            cuav.send_statustext(f"AI NAV: GPS LOCKED ({cuav.gps_satellites} SATS) | MAP READY", severity=6)
                            logging.info(f"[+] AI AUTO-SWITCH: Switched to Primary GPS ({cuav.gps_satellites} Sats, 3D Fix)")
                        else:
                            logging.warning("[!] AI AUTO-SWITCH to GPS failed — FC did not confirm parameter change")
                    elif not is_clean_gps and current_nav_mode != 'VIO':
                        ok1 = cuav.set_parameter('EK3_SRC1_POSXY', 6.0)
                        ok2 = cuav.set_parameter('EK3_SRC1_VELXY', 6.0)  # Active 30Hz VIO Velocity
                        ok3 = cuav.set_parameter('EK3_SRC1_VELZ', 0.0)
                        ok4 = cuav.set_parameter('AHRS_GPS_USE', 0.0)
                        # F-05: Only switch mode if FC confirmed all parameters
                        if ok1 and ok2 and ok3 and ok4:
                            current_nav_mode = 'VIO'
                            cuav.send_statustext("AI NAV: OAK-D VIO ACTIVE | GNSS-DENIED HOLD", severity=4)
                            logging.info("[+] AI AUTO-SWITCH: GPS unavailable. Switched to OAK-D Pro VIO.")
                        else:
                            logging.warning("[!] AI AUTO-SWITCH to VIO failed — FC did not confirm parameter change")

            # Instant RC Alerts for Critical Flight Events (Rate-Limited to Prevent GCS Freezing)
            aruco_event = aruco_notice(now, aruco_hits, aruco_notify, cuav.is_connected)
            if cuav.is_connected:
                if aruco_event == "locked":
                    off_m = math.hypot(last_aruco_offset[0], last_aruco_offset[1])
                    cuav.send_statustext(f">>> ARUCO LOCK #0 | z{last_aruco_z:.1f}m off{off_m:.2f}m", severity=6)
                elif aruco_event == "lost":
                    cuav.send_statustext(f">>> ARUCO LOST #0 | held {aruco_notify['held']:.0f}s", severity=4)

                # Precision Loiter over ArUco #0: only in LOITER, airborne, with a confirmed marker.
                # ArduPilot itself falls back to normal LOITER on stick input (>0.5 m/s^2) or 2 s target loss.
                marker_confirmed = sum(1 for t in aruco_hits if now - t <= 1.0) >= PREC_LOITER_MIN_HITS
                in_loiter = getattr(cuav, 'flight_mode', '') == "LOITER"
                want_precloit = bool(args.prec_loiter and in_loiter and is_airborne and marker_confirmed)
                prec_on = getattr(cuav, '_prec_loiter_on', False)
                last_pl_cmd = getattr(cuav, '_prec_loiter_cmd_t', 0.0)
                marker_lost = (not aruco_hits) or (now - aruco_hits[-1] > 3.0)
                if want_precloit and ((not prec_on) or now - last_pl_cmd >= 2.0):
                    if cuav.set_precision_loiter(True):
                        cuav._prec_loiter_cmd_t = now
                        if not prec_on:
                            cuav._prec_loiter_on = True
                            logging.info("[+] Precision Loiter ON (ArUco #0 confirmed in LOITER)")
                            cuav.send_statustext(">>> PREC LOITER ON | PAD #0", severity=6)
                elif prec_on and ((not in_loiter) or (not is_airborne) or marker_lost):
                    if cuav.set_precision_loiter(False):
                        cuav._prec_loiter_on = False
                        cuav._prec_loiter_cmd_t = now
                        logging.info("[*] Precision Loiter OFF (left LOITER / landed / marker lost)")
                        cuav.send_statustext(">>> PREC LOITER OFF", severity=4)

                prev_air = getattr(cuav, '_prev_is_airborne', False)
                last_air_trans_time = getattr(cuav, '_last_air_trans_time', 0.0)
                if (is_airborne != prev_air) and (now - last_air_trans_time >= 4.0):
                    if is_airborne:
                        cuav.send_statustext(f">>> AIRBORNE! VIO HOLD ({current_altitude_m:.2f}m)", severity=6)
                    else:
                        cuav.send_statustext(">>> TOUCHDOWN DETECTED", severity=6)
                    cuav._prev_is_airborne = is_airborne
                    cuav._last_air_trans_time = now

            # Multi-Page Detailed Live Telemetry Notifications on RC Screen (every 4.0s to avoid spamming)
            if now - last_badge_time >= 4.00:
                last_badge_time = now
                if cuav.is_connected:
                    rc_page = getattr(cuav, '_rc_notify_page', 0)
                    fps_val = int(fps) if 'fps' in locals() and fps > 0 else 15
                    has_aruco = aruco_notify["state"] == "locked"

                    if rc_page == 0:
                        # Page 1: 3D Position, Target Tracking & Altitude
                        if has_aruco:
                            rc_msg = f"PAD#0 N:{last_aruco_offset[0]:+.2f} E:{last_aruco_offset[1]:+.2f} | {fps_val}FPS"
                        elif on_ground:
                            rc_msg = f"VIO GND Z:0.00m | {fps_val}FPS"
                        else:
                            rc_msg = f"POS N:{pos_north:+.2f} E:{pos_east:+.2f} Z:{current_altitude_m:.2f}m | {fps_val}FPS"
                        cuav._rc_notify_page = 1
                    else:
                        # Page 2: Sensor Source, Flight Speed & Distance
                        src = "GND" if on_ground else ("TF03" if "TF03" in height_source else "CAM")
                        pad_tag = "PAD:OK" if has_aruco else "PAD:NO"
                        rc_msg = f"ALT:{current_altitude_m:.2f}m({src}) Spd:{live_speed:.1f}m/s | {fps_val}FPS"
                        cuav._rc_notify_page = 0

                    cuav.send_statustext(rc_msg[:50], severity=6)

            # Periodically write latest_flight_summary.txt and notify RC GCS every 10 seconds
            if now - last_summary_write_time >= 10.0:
                last_summary_write_time = now
                # Announce Jetson health to RC transmitter via MAVLink STATUSTEXT & Named Values
                if cuav.is_connected:
                    cuav.send_named_value("JETSON_FPS", fps)
                    cuav.send_named_value("JETSON_OK", 1.0)
                dur_s = now - start_time
                final_h_dist = math.sqrt(pos_north**2 + pos_east**2)
                summary_text = (
                    f"=================================================================\n"
                    f"             AUTONOMOUS DRONE FLIGHT MISSION SUMMARY\n"
                    f" Date & Time           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"=================================================================\n"
                    f" Flight Duration       : {dur_s:.1f} sec ({dur_s/60:.1f} mins)\n"
                    f" Max Outbound Distance : {max_outbound:.2f} meters ({max_outbound*100:.0f} cm)\n"
                    f" Total Path Flown      : {total_traveled:.2f} meters ({total_traveled*100:.0f} cm) [Odometer]\n"
                    f" Current Position      : North:{pos_north:+.2f}m, East:{pos_east:+.2f}m (Yaw:{yaw_deg:.1f}°)\n"
                    f" Distance to Home      : {final_h_dist:.2f} meters ({final_h_dist*100:.0f} cm)\n"
                    f" Max Altitude Reached  : {max_altitude:.2f} meters\n"
                    f" Total VIO Packets Sent: {vio_sent_count} packets (30 Hz)\n"
                    f" ArUco Pad Detections  : {aruco_detect_count} times\n"
                    f" GPS Health            : Sats={cuav.gps_satellites}, Fix={cuav.gps_fix_type}\n"
                    f" Autopilot EKF Status  : PosVar={cuav.ekf_horiz_var:.3f}, Mode={cuav.flight_mode}\n"
                    f" Navigation Sensor     : OAK-D Pro Stereo VIO + Active IR + ArUco Target\n"
                    f"=================================================================\n"
                )
                try:
                    with open("/home/abc/SSD/DRONE/latest_flight_summary.txt", "w") as sf:
                        sf.write(summary_text)
                except Exception:
                    pass

            time.sleep(0.002)

    logging.info("[*] Shutting down Navigation Daemon...")
    # Write Final Flight Mission Report
    dur_s = time.time() - start_time
    final_h_dist = math.sqrt(pos_north**2 + pos_east**2)
    final_report = (
        f"\n=================================================================\n"
        f"             FINAL FLIGHT MISSION REPORT SUMMARY\n"
        f"=================================================================\n"
        f" Flight Duration       : {dur_s:.1f} sec ({dur_s/60:.1f} mins)\n"
        f" Max Outbound Distance : {max_outbound:.2f} meters ({max_outbound*100:.0f} cm)\n"
        f" Total Path Flown      : {total_traveled:.2f} meters ({total_traveled*100:.0f} cm) [Odometer]\n"
        f" Final Touchdown Pos   : North:{pos_north:+.2f}m, East:{pos_east:+.2f}m (Yaw:{yaw_deg:.1f}°)\n"
        f" Residual Return Drift : {final_h_dist:.2f} meters ({final_h_dist*100:.0f} cm)\n"
        f" Max Altitude Reached  : {max_altitude:.2f} meters\n"
        f" Total VIO Packets Sent: {vio_sent_count} packets (30 Hz)\n"
        f" ArUco Pad Detections  : {aruco_detect_count} times\n"
        f" GPS Satellites        : {cuav.gps_satellites} (Fix: {cuav.gps_fix_type})\n"
        f" Autopilot Final Status: PosVar={cuav.ekf_horiz_var:.3f}, Mode={cuav.flight_mode}\n"
        f"=================================================================\n"
    )
    logging.info(final_report)
    try:
        with open("/home/abc/SSD/DRONE/latest_flight_summary.txt", "w") as sf:
            sf.write(final_report)
    except Exception:
        pass

    image_logger.stop()
    cuav.close()
    # CQ-5: Close vel_log_file in finally-equivalent block to flush buffered CSV data
    try:
        vel_log_file.flush()
        vel_log_file.close()
    except Exception:
        pass
    logging.info("[+] Navigation Daemon exited cleanly.")

if __name__ == "__main__":
    main()
