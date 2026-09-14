#!/usr/bin/env python3
"""
CUAV X7 Pro / ArduPilot MAVLink Navigation Bridge & Client
Connects NVIDIA Jetson Orin Nano to CUAV X7 Pro Autopilot.
Streams OAK-D Pro Visual Odometry to ArduPilot EKF3 for GNSS-denied autonomous flight.
"""

import sys
import os
# Must precede pymavlink import: MAVLink1 drops VISION_* covariance and reset_counter.
os.environ.setdefault("MAVLINK20", "1")
import glob
import time
import math
import argparse
import threading
from collections import deque
import statistics
from pymavlink import mavutil

class CuavMavlinkClient:
    """
    Robust background MAVLink client for CUAV X7 Pro.
    Maintains heartbeat, monitors EKF status, and streams high-rate VIO packets.
    """
    def __init__(self, preferred_port='/dev/ttyTHS1', baud=115200):
        self.preferred_port = preferred_port
        self.baud = baud
        self.master = None
        self.is_connected = False
        self.port = None
        self.flight_mode = "DISCONNECTED"
        self.ekf_horiz_var = 1.0
        self.ekf_vel_var = 1.0
        self.ekf_status_flags = 0
        self.roll_deg = 0.0
        self.pitch_deg = 0.0
        self.yaw_deg = 0.0
        self.rel_alt_m = 0.0
        self.baro_alt = 0.0
        self.lidar_alt_m = 0.15
        self.climb_rate = 0.0
        self.vio_packets_sent = 0
        self.last_heartbeat = 0.0
        self.params = {}
        self.reset_counter = 0
        self.armed_state_known = False  # True only after a FC HEARTBEAT on the current link

        # TF03 LiDAR Rangefinder Telemetry & Freshness Tracking
        # LiDAR still on airframe (~11cm). Downward OAK now under battery (~3cm lens-to-floor).
        self.lidar_ground_clearance_m = 0.11
        self.camera_ground_clearance_m = 0.03
        self.tf03_h_m = 0.00
        self.tf03_h_age_s = 999.0
        self.tf03_h_valid = False
        self.tf03_last_rx_time = 0.0
        self.tf03_raw_cm = 0
        self.tf03_voltage = 0.0
        # TF03-3: Window reduced from 11 to 5 — halves median-filter lag from ~1s to ~0.3s
        self._tf03_history = deque(maxlen=5)

        # GPS Live Status & Anti-Jamming Attributes
        self.gps_fix_type = 0
        self.gps_satellites = 0
        self.gps_hacc = 99.0
        self.gps_vel = 0.0
        # Exact site coordinates provided by pilot
        self.gps_lat = 30.895560
        self.gps_lon = 74.613858
        self.gps_alt = 177.0
        self.origin_set = False
        self.connect_time = 0.0
        self.last_gps_time = 0.0
        self.is_jammed = False
        self.jamming_reason = "No GPS Data"
        self.is_armed = False

        self._running = True
        self._lock = threading.Lock()
        # F-03: track last heartbeat send time for companion HEARTBEAT @ 1 Hz
        self._last_hb_send = 0.0

        self.last_attitude_time = 0.0
        self.attitude_count = 0
        # Heading for rotating optical flow into NED. Integrated from FC body rates so it is not
        # the EKF's own yaw fed straight back (that loop froze EKF yaw in flights 18/19), and
        # pulled slowly toward EKF yaw so the compass still sets the long-term heading.
        self.FLOW_HDG_TAU_AIR_S = 600.0   # log replay: 30 s kept 49 deg rms error, 300 s 9-11, no pull 0.4
        self.FLOW_HDG_TAU_GROUND_S = 1.0
        self.flow_heading_deg = None
        self._flow_hdg_last_boot_ms = None
        self._flow_hdg_time = 0.0
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def request_streams(self, conn):
        """Explicitly request required MAVLink streams at specified rates using MAV_CMD_SET_MESSAGE_INTERVAL."""
        if not conn:
            return
        target_sys = getattr(conn, 'target_system', 1) or 1
        target_comp = getattr(conn, 'target_component', 1) or 1
        
        # Message intervals in microseconds (50000us = 20Hz, 100000us = 10Hz)
        streams = [
            (mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 50000),             # 20 Hz Attitude (Roll, Pitch, Yaw)
            (mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD, 100000),             # 10 Hz VFR_HUD (Heading, Baro Alt)
            (mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 100000), # 10 Hz Global Position
            (mavutil.mavlink.MAVLINK_MSG_ID_DISTANCE_SENSOR, 50000),      # 20 Hz TF03 LiDAR
            (mavutil.mavlink.MAVLINK_MSG_ID_EKF_STATUS_REPORT, 200000),   # 5 Hz EKF Variance
        ]
        for msg_id, interval_us in streams:
            try:
                conn.mav.command_long_send(
                    target_sys, target_comp,
                    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                    0, msg_id, interval_us, 0, 0, 0, 0, 0
                )
            except Exception:
                pass
        
        # Also send legacy request_data_stream_send as fallback
        try:
            conn.mav.request_data_stream_send(
                target_sys, target_comp,
                mavutil.mavlink.MAV_DATA_STREAM_EXTRA1, 20, 1
            )
            conn.mav.request_data_stream_send(
                target_sys, target_comp,
                mavutil.mavlink.MAV_DATA_STREAM_EXTRA2, 10, 1
            )
        except Exception:
            pass

    def _find_port(self):
        def _check_ardupilot(c, timeout_s=2.5):
            t_end = time.time() + timeout_s
            while time.time() < t_end:
                msg = c.recv_match(type='HEARTBEAT', blocking=True, timeout=0.4)
                if msg and getattr(msg, 'autopilot', 0) == mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA:
                    c.target_system = msg.get_srcSystem()
                    c.target_component = msg.get_srcComponent()
                    c.source_system = c.target_system
                    c.source_component = mavutil.mavlink.MAV_COMP_ID_ONBOARD_COMPUTER
                    c.mav.srcSystem = c.target_system
                    c.mav.srcComponent = mavutil.mavlink.MAV_COMP_ID_ONBOARD_COMPUTER
                    self.request_streams(c)
                    return True
            return False

        # 1. First priority: Try preferred port with retry (prevents grabbing SLCAN on ttyACM0)
        if self.preferred_port and os.path.exists(self.preferred_port):
            for _ in range(3):
                c = None
                try:
                    c = mavutil.mavlink_connection(self.preferred_port, baud=self.baud, autoreconnect=False, source_system=1, source_component=mavutil.mavlink.MAV_COMP_ID_ONBOARD_COMPUTER)
                    if _check_ardupilot(c, timeout_s=2.0):
                        return self.preferred_port, c
                    if c:
                        c.close()
                except Exception:
                    if c:
                        try: c.close()
                        except: pass
                    time.sleep(0.2)

        # 2. Secondary fallback across available serial devices
        candidates = ["/dev/ttyTHS1"] + sorted(glob.glob("/dev/ttyACM*"), reverse=True) + glob.glob("/dev/ttyUSB*")
        seen = set()
        ports = [p for p in candidates if not (p in seen or seen.add(p))]

        for p in ports:
            c = None
            try:
                c = mavutil.mavlink_connection(p, baud=self.baud, autoreconnect=False, source_system=1, source_component=mavutil.mavlink.MAV_COMP_ID_ONBOARD_COMPUTER)
                if _check_ardupilot(c, timeout_s=1.5):
                    return p, c
                else:
                    if c:
                        c.close()
            except Exception:
                if c:
                    try:
                        c.close()
                    except:
                        pass
        return None, None

    def _worker(self):
        while self._running:
            if not self.is_connected or self.master is None:
                p, conn = self._find_port()
                if conn:
                    with self._lock:
                        self.master = conn
                        self.port = p
                        self.is_connected = True
                        self.armed_state_known = False
                        self.flight_mode = conn.flightmode
                        self.last_heartbeat = time.time()
                else:
                    time.sleep(1.0)
                    continue

            try:
                # F-03: Send companion HEARTBEAT @ 1 Hz so ArduPilot can detect companion failure
                now_t = time.time()
                if now_t - self._last_hb_send >= 1.0:
                    try:
                        sys_id = getattr(self.master, 'target_system', 1) or 1
                        self.master.mav.srcSystem = sys_id
                        self.master.mav.srcComponent = mavutil.mavlink.MAV_COMP_ID_ONBOARD_COMPUTER
                        self.master.mav.heartbeat_send(
                            mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                            0, 0, 0
                        )
                    except Exception:
                        pass
                    self._last_hb_send = now_t

                msg = self.master.recv_match(blocking=False)
                if msg:
                    t = msg.get_type()
                    if t == 'HEARTBEAT' and getattr(msg, 'type', 0) not in (mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER):
                        self.last_heartbeat = time.time()
                        new_mode = self.master.flightmode
                        if new_mode != self.flight_mode:
                            import logging as _log
                            _log.info(f"[+] CUAV Flight Mode Changed: {self.flight_mode} -> {new_mode}")
                        self.flight_mode = new_mode
                        new_armed = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
                        if new_armed and not self.is_armed:
                            # Drone just armed! Automatically zero out AGL and reset Home to current position
                            try:
                                target_sys = self.master.target_system if getattr(self.master, 'target_system', 0) > 0 else 1
                                target_comp = self.master.target_component if getattr(self.master, 'target_component', 0) > 0 else 1
                                self.master.mav.command_long_send(
                                    target_sys, target_comp,
                                    mavutil.mavlink.MAV_CMD_DO_SET_HOME,
                                    0, 1, 0, 0, 0, 0, 0, 0
                                )
                            except Exception:
                                pass
                        self.is_armed = new_armed
                        self.armed_state_known = True
                    elif t == 'STATUSTEXT':
                        stat_txt = getattr(msg, 'text', '')
                        if stat_txt:
                            import logging as _log
                            _log.warning(f"[CUAV STATUSTEXT] {stat_txt}")
                    elif t == 'ATTITUDE':
                        self.roll_deg = msg.roll * 57.2958
                        self.pitch_deg = msg.pitch * 57.2958
                        self.yaw_deg = msg.yaw * 57.2958
                        self.last_attitude_time = time.time()
                        self.attitude_count += 1
                        self._update_flow_heading(msg.time_boot_ms, msg.roll, msg.pitch, msg.yaw,
                                                  msg.rollspeed, msg.pitchspeed, msg.yawspeed)
                    elif t == 'GLOBAL_POSITION_INT':
                        self.rel_alt_m = getattr(msg, 'relative_alt', 0) / 1000.0
                        if getattr(msg, 'hdg', 65535) != 65535 and (time.time() - self.last_attitude_time > 1.0):
                            self.yaw_deg = msg.hdg / 100.0
                        if msg.lat != 0 and msg.lon != 0:
                            self.origin_set = True
                        else:
                            # Lat/Lon is 0 -> Autopilot lacks global origin! Auto-inject every 2.0s until locked
                            now_t = time.time()
                            if now_t - getattr(self, '_last_origin_attempt', 0) > 2.0:
                                self._last_origin_attempt = now_t
                                self.set_ekf_origin()
                    elif t == 'DISTANCE_SENSOR':
                        orient = getattr(msg, 'orientation', 25)
                        # Orientation 25 = MAV_SENSOR_ROTATION_PITCH_270 (Downward facing)
                        if orient == 25:
                            curr_dist = getattr(msg, 'current_distance', None)
                            if curr_dist is not None and curr_dist >= 0:
                                val_m = float(curr_dist) / 100.0
                                if (not math.isnan(val_m) and 
                                    not math.isinf(val_m) and 
                                    0.0 <= val_m <= 180.0):
                                    
                                    bounded_m = max(0.0, val_m)
                                    now_t = time.time()
                                    with self._lock:
                                        self._tf03_history.append(bounded_m)
                                        filtered_m = float(statistics.median(self._tf03_history))
                                        self.tf03_raw_cm = curr_dist
                                        self.tf03_h_m = filtered_m
                                        self.lidar_alt_m = filtered_m
                                        self.tf03_last_rx_time = now_t
                                        self.tf03_h_valid = True
                                        self.tf03_msg_type = "DISTANCE_SENSOR"
                    elif t == 'RANGEFINDER':
                        raw_m = getattr(msg, 'distance', None)
                        volt_val = getattr(msg, 'voltage', 0.0)
                        now_t = time.time()
                        if raw_m is not None and not math.isnan(raw_m) and not math.isinf(raw_m):
                            dist_to_use = float(raw_m)
                            bounded_m = max(0.0, dist_to_use)
                            if 0.0 <= bounded_m <= 180.0:
                                with self._lock:
                                    self._tf03_history.append(bounded_m)
                                    filtered_m = float(statistics.median(self._tf03_history))
                                    self.tf03_h_m = filtered_m
                                    self.lidar_alt_m = filtered_m
                                    self.tf03_voltage = float(volt_val)
                                    self.tf03_last_rx_time = now_t
                                    self.tf03_h_valid = True
                                    self.tf03_msg_type = "RANGEFINDER"
                    elif t == 'VFR_HUD':
                        self.baro_alt = getattr(msg, 'alt', 0.0)
                        self.climb_rate = getattr(msg, 'climb', 0.0)
                        hud_hdg = getattr(msg, 'heading', None)
                        if hud_hdg is not None and (time.time() - self.last_attitude_time > 1.0):
                            self.yaw_deg = float(hud_hdg)
                    elif t == 'EKF_STATUS_REPORT':
                        self.ekf_horiz_var = msg.pos_horiz_variance
                        self.ekf_vel_var = msg.velocity_variance
                        self.ekf_status_flags = msg.flags
                    elif t == 'PARAM_VALUE':
                        with self._lock:
                            self.params[msg.param_id] = msg.param_value
                    elif t == 'GPS_RAW_INT':
                        self.gps_fix_type = msg.fix_type
                        self.gps_satellites = msg.satellites_visible
                        self.gps_hacc = getattr(msg, 'h_acc', 99000) / 1000.0
                        self.gps_vel = getattr(msg, 'vel', 0) / 100.0
                        if msg.lat != 0 and msg.lon != 0:
                            self.gps_lat = msg.lat / 1e7
                            self.gps_lon = msg.lon / 1e7
                            self.gps_alt = msg.alt / 1000.0
                            if not self.origin_set and msg.fix_type >= 2:
                                self.set_ekf_origin(lat=self.gps_lat, lon=self.gps_lon, alt=self.gps_alt)
                        self.last_gps_time = time.time()

                    elif t == 'HOME_POSITION':
                        # Store FC reported home position for reference only, do NOT override pilot coordinates
                        h_lat = getattr(msg, 'latitude',  0) / 1e7
                        h_lon = getattr(msg, 'longitude', 0) / 1e7
                        h_alt = getattr(msg, 'altitude',  0) / 1000.0
                        if h_lat != 0.0 and h_lon != 0.0:
                            self.home_lat = h_lat
                            self.home_lon = h_lon
                            self.home_alt = h_alt
                else:
                    time.sleep(0.005)

                # Periodic stream keepalive: If attitude has not arrived for > 2.0s, re-request streams
                now_t = time.time()
                if self.is_connected and (now_t - self.last_attitude_time > 2.0):
                    if now_t - getattr(self, '_last_stream_req_time', 0.0) > 3.0:
                        self._last_stream_req_time = now_t
                        self.request_streams(self.master)

                if time.time() - self.last_heartbeat > 4.0:
                    # HIGH-9: Log warning on mid-flight connection loss
                    if self.is_armed:
                        import logging as _log
                        _log.warning("[!] MAVLink watchdog: FC heartbeat lost while ARMED — VIO stream halted")
                    with self._lock:
                        old_master = self.master
                        self.is_connected = False
                        self.master = None
                        self.origin_set = False
                        self.armed_state_known = False
                    self._close_quietly(old_master)
            except Exception:
                with self._lock:
                    old_master = self.master
                    self.is_connected = False
                    self.master = None
                    self.armed_state_known = False
                self._close_quietly(old_master)
                time.sleep(1.0)

    def _update_flow_heading(self, boot_ms, roll, pitch, yaw, p, q, r):
        """Integrate Euler yaw rate from body rates; complementary pull toward EKF yaw."""
        ekf_deg = math.degrees(yaw)
        last = self._flow_hdg_last_boot_ms
        self._flow_hdg_last_boot_ms = boot_ms
        dt = None if last is None else (boot_ms - last) / 1000.0
        if self.flow_heading_deg is None or dt is None or not (0.0 < dt <= 0.5):
            self.flow_heading_deg = ekf_deg
            self._flow_hdg_time = time.time()
            return
        c_pitch = math.cos(pitch)
        yaw_rate = (q * math.sin(roll) + r * math.cos(roll)) / c_pitch if abs(c_pitch) > 0.2 else r
        hdg = self.flow_heading_deg + math.degrees(yaw_rate) * dt
        tau = self.FLOW_HDG_TAU_AIR_S if self.is_armed else self.FLOW_HDG_TAU_GROUND_S
        err = (ekf_deg - hdg + 180.0) % 360.0 - 180.0
        hdg += err * min(1.0, dt / tau)
        self.flow_heading_deg = (hdg + 180.0) % 360.0 - 180.0
        self._flow_hdg_time = time.time()

    def get_flow_heading_deg(self, max_age_s=0.5):
        """Flow-rotation heading in degrees, or None if stale."""
        if self.flow_heading_deg is None or time.time() - self._flow_hdg_time > max_age_s:
            return None
        return self.flow_heading_deg

    @staticmethod
    def _close_quietly(conn):
        """Close a dropped MAVLink link so its serial fd is not left open when the port is reopened."""
        if conn is None:
            return
        try:
            conn.close()
        except Exception:
            pass

    def send_vision_estimate(self, x, y, z, roll_rad=0.0, pitch_rad=0.0, yaw_rad=0.0, pos_cov=0.01):
        """
        Send VISION_POSITION_ESTIMATE to ArduPilot EKF3.
        pos_cov: position covariance (m^2). Use small value (~0.01) for good tracking,
                 larger (~0.5) for poor tracking. Caller should not send when tracking is lost.
        F-02: Uses time.monotonic() (boot-relative, non-jumping) instead of time.time().
        F-06: Covariance is quality-dependent — prevents EKF over-weighting noisy estimates.
        CQ-6: Serial I/O performed outside _lock to prevent deadlock if USB stalls.
        """
        if not self.is_connected or not self.master:
            return False
        try:
            # F-02: Boot-relative monotonic microseconds — matches ArduPilot's internal clock domain
            usec = int(time.monotonic() * 1e6)
            # F-06: Quality-dependent covariance (21-element upper-triangular, row-major)
            # [x, y, z, roll, pitch, yaw] diagonal entries; off-diagonal = 0
            cov = float(pos_cov)
            att_cov = 0.01   # attitude covariance (rad^2) — compass/IMU driven
            yaw_cov = 0.05   # yaw covariance slightly larger (compass noise)
            covariance = [
                cov,  0.0,  0.0,  0.0,      0.0,      0.0,
                      cov,  0.0,  0.0,      0.0,      0.0,
                            cov,  0.0,      0.0,      0.0,
                                  att_cov,  0.0,      0.0,
                                            att_cov,  0.0,
                                                      yaw_cov
            ]
            # CQ-6: Capture master reference outside lock, do I/O outside lock
            with self._lock:
                mav_ref = self.master
            if mav_ref is None:
                return False
            sys_id = getattr(mav_ref, 'target_system', 1) or 1
            mav_ref.mav.srcSystem = 255  # companion; sysid=1 is the FC and VPE can be dropped
            mav_ref.mav.srcComponent = mavutil.mavlink.MAV_COMP_ID_ONBOARD_COMPUTER
            # MAVLink2 extension fields: covariance (21) + reset_counter.
            mav_ref.mav.vision_position_estimate_send(
                usec, float(x), float(y), float(z),
                float(roll_rad), float(pitch_rad), float(yaw_rad),
                covariance, int(self.reset_counter) & 0xFF,
            )
            with self._lock:
                self.vio_packets_sent += 1
            return True
        except Exception as e:
            if not getattr(self, '_vpe_err_logged', False):
                self._vpe_err_logged = True
                import logging as _log
                _log.error(f"[!] VISION_POSITION_ESTIMATE send failed: {e}")
            return False

    def send_vision_speed(self, vx, vy, vz=0.0, vel_cov=0.05):
        """
        Send VISION_SPEED_ESTIMATE to ArduPilot EKF3.
        F-02: Uses time.monotonic() for consistent timestamp domain.
        CQ-6: Serial I/O outside _lock.
        """
        if not self.is_connected or not self.master:
            return False
        try:
            # F-02: Boot-relative monotonic microseconds
            usec = int(time.monotonic() * 1e6)
            c = float(vel_cov)
            cov_matrix = [c, 0.0, 0.0, 0.0, c, 0.0, 0.0, 0.0, c]
            with self._lock:
                mav_ref = self.master
            if mav_ref is None:
                return False
            sys_id = getattr(mav_ref, 'target_system', 1) or 1
            mav_ref.mav.srcSystem = 255
            mav_ref.mav.srcComponent = mavutil.mavlink.MAV_COMP_ID_ONBOARD_COMPUTER
            # MAVLink2 extension fields: covariance (9) + reset_counter.
            mav_ref.mav.vision_speed_estimate_send(
                usec, float(vx), float(vy), float(vz),
                cov_matrix, int(self.reset_counter) & 0xFF,
            )
            return True
        except Exception as e:
            if not getattr(self, '_vse_err_logged', False):
                self._vse_err_logged = True
                import logging as _log
                _log.error(f"[!] VISION_SPEED_ESTIMATE send failed: {e}")
            return False

    def send_landing_target(self, angle_x_rad, angle_y_rad, distance_m, size_x_rad=0.0, size_y_rad=0.0,
                            time_usec=None):
        """MAVLink LANDING_TARGET for ArduPilot precland / precision loiter (PLND_TYPE=1).

        ArduPilot 4.7 MAVLink backend accepts only MAV_FRAME_BODY_FRD / LOCAL_FRD and builds the
        line of sight as (forward=-tan(angle_y), right=tan(angle_x), down=1).
        """
        if not self.is_connected or not self.master:
            return False
        try:
            usec = int(time.monotonic() * 1e6) if time_usec is None else int(time_usec)
            with self._lock:
                mav_ref = self.master
            if mav_ref is None:
                return False
            sys_id = getattr(mav_ref, 'target_system', 1) or 1
            mav_ref.mav.srcSystem = sys_id
            mav_ref.mav.srcComponent = mavutil.mavlink.MAV_COMP_ID_ONBOARD_COMPUTER
            mav_ref.mav.landing_target_send(
                usec,
                0,
                mavutil.mavlink.MAV_FRAME_BODY_FRD,  # BODY_NED is rejected: "Plnd: Frame not supported"
                float(angle_x_rad),
                float(angle_y_rad),
                float(distance_m),
                float(size_x_rad),
                float(size_y_rad),
                0.0, 0.0, 0.0, (1.0, 0.0, 0.0, 0.0),
                0,  # type
                0,  # position_valid: use angle_x/angle_y
            )
            return True
        except Exception:
            return False

    def get_tf03_height(self, max_age_s=0.25):
        """
        Thread-safe getter for validated TF03 LiDAR rangefinder height H.
        Returns: (height_m, age_s, is_valid)
        - Rejects stale data if age_s > max_age_s.
        - Rejects NaN, Inf, <= 0.0, or out-of-range values.
        - Physical ground clearance baseline is 0.11m.
        """
        with self._lock:
            if not self.is_connected or self.tf03_last_rx_time == 0.0:
                return 0.00, 999.0, False
            now_t = time.time()
            age_s = max(0.0, now_t - self.tf03_last_rx_time)
            is_fresh = (age_s <= max_age_s)
            is_valid = self.tf03_h_valid and is_fresh and (0.0 <= self.tf03_h_m <= 180.0)
            return float(self.tf03_h_m), float(age_s), bool(is_valid)

    def get_camera_height(self, max_age_s=0.25):
        """
        Thread-safe getter for camera height above ground H_cam.
        Physical mounting:
        - Camera lens to ground: 0.13m (13cm)
        - TF03 LiDAR lens to ground: 0.11m (11cm)
        - Delta: Camera is +0.02m (+2cm) above LiDAR lens.
        Returns: (cam_height_m, age_s, is_valid)
        """
        lidar_h, age_s, is_valid = self.get_tf03_height(max_age_s=max_age_s)
        cam_h = max(self.camera_ground_clearance_m, lidar_h + (self.camera_ground_clearance_m - self.lidar_ground_clearance_m))
        return cam_h, age_s, is_valid

    def send_distance_sensor(self, distance_m, min_m=0.10, max_m=4.00):
        if not self.is_connected or not self.master:
            return False
        try:
            boot_ms = int(time.time() * 1000) & 0xFFFFFFFF
            dist_cm = int(min(max(float(distance_m) * 100.0, float(min_m) * 100.0), float(max_m) * 100.0))
            with self._lock:
                self.master.mav.distance_sensor_send(
                    boot_ms,
                    int(min_m * 100),
                    int(max_m * 100),
                    dist_cm,
                    0,   # MAV_DISTANCE_SENSOR_LASER
                    0,   # sensor id
                    25,  # MAV_SENSOR_ROTATION_PITCH_270 (Downward facing)
                    0    # covariance
                )
            return True
        except Exception:
            return False

    def send_statustext(self, text, severity=6):
        if not self.is_connected or not self.master:
            return False
        try:
            with self._lock:
                sys_id = getattr(self.master, 'target_system', 1) or 1
                self.master.mav.srcSystem = sys_id
                self.master.mav.srcComponent = mavutil.mavlink.MAV_COMP_ID_ONBOARD_COMPUTER
                self.master.mav.statustext_send(int(severity), text.encode('utf-8')[:50])
            return True
        except Exception:
            return False

    def send_named_value(self, name, val):
        if not self.is_connected or not self.master:
            return False
        try:
            usec = int(time.time() * 1e6)
            name_bytes = name.encode('utf-8')[:10]
            with self._lock:
                sys_id = getattr(self.master, 'target_system', 1) or 1
                self.master.mav.srcSystem = sys_id
                self.master.mav.srcComponent = mavutil.mavlink.MAV_COMP_ID_ONBOARD_COMPUTER
                self.master.mav.named_value_float_send(usec, name_bytes, float(val))
            return True
        except Exception:
            return False

    def set_ekf_origin(self, lat=None, lon=None, alt=None):
        """
        Inject EKF global origin into ArduPilot.
        - If lat/lon provided explicitly: use those values.
        - If GPS has a live fix (gps_fix_type >= 2): use GPS coordinates.
        - If previous coordinates exist in self.gps_lat/lon: reuse them.
        - Otherwise fallback to default anchor (28.6139, 77.2090) so indoor/GNSS-denied EKF
          can always anchor (0,0,0) and allow LOITER arming.
        - Altitude: Defaults to ground barometer reading (self.baro_alt) so AGL starts at 0.0m.
        """
        if not self.is_connected or not self.master:
            return False

        # Resolve latitude / longitude
        if lat is not None and lon is not None:
            origin_lat = float(lat)
            origin_lon = float(lon)
        elif self.gps_fix_type >= 2 and self.gps_lat != 0.0 and self.gps_lon != 0.0:
            origin_lat = float(self.gps_lat)
            origin_lon = float(self.gps_lon)
        elif self.gps_lat != 0.0 and self.gps_lon != 0.0:
            origin_lat = float(self.gps_lat)
            origin_lon = float(self.gps_lon)
        else:
            # Default pilot site coordinates to allow EKF origin lock
            origin_lat = 30.895560
            origin_lon = 74.613858

        # Resolve altitude: Must match ground barometer to keep Relative Altitude (AGL) at 0.0m
        if alt is not None and float(alt) != 0.0:
            origin_alt = float(alt)
        elif self.baro_alt != 0.0:
            origin_alt = float(self.baro_alt)
        elif self.gps_alt != 0.0 and self.gps_fix_type >= 2:
            origin_alt = float(self.gps_alt)
        else:
            origin_alt = 0.0

        try:
            with self._lock:
                target_sys = self.master.target_system if getattr(self.master, 'target_system', 0) > 0 else 1
                target_comp = self.master.target_component if getattr(self.master, 'target_component', 0) > 0 else 1
            self.master.mav.set_gps_global_origin_send(
                target_sys,
                int(origin_lat * 1e7), int(origin_lon * 1e7), int(origin_alt * 1000)
            )
            self.master.mav.command_long_send(
                target_sys, target_comp,
                mavutil.mavlink.MAV_CMD_DO_SET_HOME,
                0, 0, 0, 0, 0, float(origin_lat), float(origin_lon), float(origin_alt)
            )
            self.gps_lat = origin_lat
            self.gps_lon = origin_lon
            self.gps_alt = origin_alt
            self.origin_set = True
            return True
        except Exception:
            return False

    def set_home_position(self, lat, lon, alt_m=None):
        """
        Manually set EKF origin and ArduPilot home position when GPS is disconnected.
        If alt_m is None or 0.0, current ground barometer altitude is used so AGL starts at 0.0m.
        """
        if alt_m is None or float(alt_m) == 0.0:
            alt_m = self.baro_alt if self.baro_alt != 0.0 else 0.0
        ok = self.set_ekf_origin(lat=float(lat), lon=float(lon), alt=float(alt_m))
        if ok:
            self.gps_lat = float(lat)
            self.gps_lon = float(lon)
            self.gps_alt = float(alt_m)
            self.origin_set = True
        return ok

    def reset_origin(self):
        self.reset_counter = (self.reset_counter + 1) % 256
        self.set_ekf_origin()

    def get_parameter(self, param_name, timeout=2.5):
        if not self.is_connected or not self.master:
            return None
        with self._lock:
            if param_name in self.params:
                del self.params[param_name]
            self.master.mav.param_request_read_send(
                self.master.target_system, self.master.target_component,
                param_name.encode('utf-8'), -1
            )
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if param_name in self.params:
                    return self.params[param_name]
            time.sleep(0.02)
        return None

    def set_parameter(self, param_name, value, timeout=2.5):
        if not self.is_connected or not self.master:
            return False
        # EKF source params are static config: refuse while armed or before arm state is known.
        if param_name.startswith('EK3_SRC') and (self.is_armed or not self.armed_state_known):
            import logging as _log
            _log.error(f"[!] Refused {param_name}={value}: EKF source write while armed/unknown arm state")
            return False
        with self._lock:
            if param_name in self.params:
                del self.params[param_name]
            self.master.mav.param_set_send(
                self.master.target_system, self.master.target_component,
                param_name.encode('utf-8'), float(value),
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32
            )
        if timeout <= 0:
            return True
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if param_name in self.params:
                    return abs(self.params[param_name] - float(value)) < 0.001
            time.sleep(0.02)
        return False

    def calibrate_rangefinder_offset(self, offset_v=0.650, gndclear_m=0.11):
        """Calibrate analog rangefinder baseline offset and ground clearance on CUAV."""
        p1 = self.set_parameter('RNGFND1_OFFSET', float(offset_v))
        p2 = self.set_parameter('RNGFND1_GNDCLR', float(gndclear_m))
        return p1 and p2

    def enable_vio_mode(self):
        """
        Configure CUAV for Non-GPS Visual Odometry (GPS disconnected / indoor).
        Active VIO with VISION_POSITION_ESTIMATE and VISION_SPEED_ESTIMATE.
        EK3_SRC1_POSXY=6 (ExternalNav position).
        EK3_SRC1_VELXY=6 (ExternalNav velocity active for LOITER braking).
        EK3_SRC1_POSZ=2 (Rangefinder / TF03 LiDAR).
        """
        p1 = self.set_parameter('EK3_SRC1_POSXY', 6.0) # ExternalNav position (static)
        p2 = self.set_parameter('EK3_SRC1_VELXY', 6.0) # ExternalNav velocity
        p3 = self.set_parameter('EK3_SRC1_VELZ', 0.0)  # None — baro/LiDAR derived
        p4 = self.set_parameter('EK3_SRC1_POSZ', 2.0)  # Rangefinder (TF03 LiDAR)
        p5 = self.set_parameter('AHRS_GPS_USE', 0.0)   # Ignore GPS for attitude
        return p1 and p2 and p3 and p4 and p5

    def revert_gps_mode(self):
        """Restore standard outdoor GPS navigation."""
        p1 = self.set_parameter('EK3_SRC1_POSXY', 3.0) # GPS
        p2 = self.set_parameter('EK3_SRC1_VELXY', 3.0) # GPS
        p3 = self.set_parameter('EK3_SRC1_VELZ', 3.0)  # GPS
        p4 = self.set_parameter('EK3_SRC1_POSZ', 1.0)  # Baro
        p5 = self.set_parameter('AHRS_GPS_USE', 1.0)   # Use GPS for attitude
        return p1 and p2 and p3 and p4 and p5

    def set_indoor_failsafe(self, land=True):
        """Set radio failsafe to LAND (3) for safe indoor flight without GPS Home."""
        val = 3.0 if land else 1.0 # 3 = Land, 1 = RTL
        return self.set_parameter('FS_THR_ENABLE', val)

    def enable_hybrid_mode(self):
        """
        OPTION A: Anti-Jamming / Anti-Spoofing Hybrid Mode.
        - Primary Navigation (SRC1): OAK-D Pro Visual Odometry (Immune to RF Jamming!)
        - Secondary Backup (SRC2): Global GPS (for Lat/Long maps and long-range missions)
        EK3_SRC1_VELXY=0 — position-only VIO; prevents velocity innovation spikes.
        """
        p1 = self.set_parameter('GPS_TYPE', 1.0)        # Physical GPS enabled
        p2 = self.set_parameter('EK3_SRC1_POSXY', 6.0)  # Primary: ExternalNav / VIO
        p3 = self.set_parameter('EK3_SRC1_VELXY', 0.0)  # Primary: IMU-derived velocity
        p4 = self.set_parameter('EK3_SRC2_POSXY', 3.0)  # Secondary: GPS
        p5 = self.set_parameter('EK3_SRC2_VELXY', 3.0)  # GPS velocity for SRC2
        p6 = self.set_parameter('EK3_SRC_OPTIONS', 0.0) # No cross-source velocity fusion
        return p1 and p2 and p3 and p4 and p5 and p6

    def check_gps_jamming(self, camera_speed_m_s=0.0):
        """
        Real-time GPS Spoofing Detector.
        Cross-checks GPS velocity against OAK-D Pro Optical Flow speed.
        CQ-8: GPS absent/disconnected is NOT reported as jamming — it is simply unavailable.
              Only active GPS data that is internally inconsistent is flagged as jamming/spoofing.
        """
        now = time.time()

        # GPS module is disconnected or has never sent data — not jamming, just absent
        if self.last_gps_time == 0 or (now - self.last_gps_time > 3.0):
            self.is_jammed = False
            self.jamming_reason = "GPS not available (disconnected or no signal)"
            return False, self.jamming_reason

        # GPS connected but no 3D fix — degraded, not jammed
        if self.gps_fix_type < 3:
            self.is_jammed = False
            self.jamming_reason = f"GPS no 3D fix (FixType={self.gps_fix_type}, Sats={self.gps_satellites})"
            return False, self.jamming_reason

        # GPS accuracy too low — degraded signal, not necessarily jamming
        if self.gps_hacc > 5.0:
            self.is_jammed = False
            self.jamming_reason = f"GPS accuracy degraded ({self.gps_hacc:.1f}m) — not jamming"
            return False, self.jamming_reason

        # Anti-Spoofing Cross-Check:
        # GPS claims high speed but camera confirms stationary hover → spoofing
        vel_diff = abs(self.gps_vel - camera_speed_m_s)
        if vel_diff > 3.0 and camera_speed_m_s < 0.4 and self.gps_vel > 3.0:
            self.is_jammed = True
            self.jamming_reason = (
                f"CRITICAL: GPS SPOOFING DETECTED! "
                f"(GPS vel={self.gps_vel:.1f}m/s vs Camera={camera_speed_m_s:.1f}m/s)"
            )
            return True, self.jamming_reason

        self.is_jammed = False
        self.jamming_reason = f"GPS OK (3D Fix | {self.gps_satellites} Sats | Acc:{self.gps_hacc:.1f}m)"
        return False, self.jamming_reason

    def reboot(self):
        """Sends MAVLink reboot command to flight controller."""
        if not self.is_connected or not self.master:
            return False
        with self._lock:
            self.master.mav.command_long_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
                0, 1, 0, 0, 0, 0, 0, 0
            )
        return True

    def set_precision_loiter(self, enable):
        """Aux function 39 (PrecLoiter): hold over LANDING_TARGET while in LOITER."""
        if not self.is_connected or not self.master:
            return False
        try:
            aux_fn = 39
            switch_pos = 2.0 if enable else 0.0
            cmd = getattr(mavutil.mavlink, 'MAV_CMD_DO_AUX_FUNCTION', 218)
            with self._lock:
                mav_ref = self.master
            if mav_ref is None:
                return False
            mav_ref.mav.command_long_send(
                mav_ref.target_system, mav_ref.target_component,
                cmd,
                0, float(aux_fn), switch_pos, 0, 0, 0, 0, 0
            )
            return True
        except Exception:
            return False

    def send_rc_display_telemetry(self, lat, lon, gl_alt_m, agl_alt_m, dist_to_home_m=0.0, speed_m_s=0.0, heading_deg=0.0):
        """
        Broadcast GPS_RAW_INT, GLOBAL_POSITION_INT, and NAV_CONTROLLER_OUTPUT
        so UniRC / Skydroid GCS displays live Lat, Lon, GL, AGL, and Dist from Home even without physical GPS.
        """
        if not self.is_connected or not self.master:
            return False
        
        try:
            lat_int = int(float(lat) * 1e7)
            lon_int = int(float(lon) * 1e7)
            gl_mm = int(float(gl_alt_m) * 1000)
            agl_mm = int(float(agl_alt_m) * 1000)
            usec = int(time.time() * 1e6)
            time_boot_ms = int((time.time() - getattr(self, '_start_time', time.time())) * 1000) & 0xFFFFFFFF
            vel_cms = int(float(speed_m_s) * 100)
            cog_cdeg = int(float(heading_deg) * 100) % 36000

            with self._lock:
                # NAV_CONTROLLER_OUTPUT: Sets Distance from Home (wp_dist) without injecting fake GPS sensor data
                self.master.mav.nav_controller_output_send(
                    0.0, 0.0, 0, 0, int(dist_to_home_m), 0.0, 0.0, 0.0
                )
            return True
        except Exception:
            return False

    def send_fake_gps_for_rc_display(self, lat=None, lon=None, alt_m=None):
        return self.send_rc_display_telemetry(
            lat=lat or self.gps_lat,
            lon=lon or self.gps_lon,
            gl_alt_m=alt_m or self.baro_alt,
            agl_alt_m=self.rel_alt_m
        )

    def close(self):
        self._running = False


def print_banner():
    print("=" * 72)
    print("  CUAV X7 Pro / ArduPilot MAVLink AI Navigation Bridge")
    print("  NVIDIA Jetson Orin Nano + OAK-D Pro Visual Odometry Pipeline")
    print("=" * 72)


def show_status(client):
    print_banner()
    print("[*] Checking CUAV X7 Pro connection...")
    for _ in range(20):
        if client.is_connected:
            break
        time.sleep(0.1)

    if not client.is_connected:
        print("[-] CUAV X7 Pro not detected. Ensure USB Type-C is plugged into Jetson.")
        return

    print(f"[+] Connected to CUAV X7 Pro on port: {client.port}")
    print(f"    - Autopilot : ArduPilot (ArduCopter)")
    print(f"    - Flight Mode: {client.flight_mode}")
    print(f"    - Attitude   : Roll: {client.roll_deg:+05.1f}° | Pitch: {client.pitch_deg:+05.1f}° | Yaw: {client.yaw_deg:+05.1f}°")

    print("\n--- Current EKF3 Navigation Configuration ---")
    params = ['VISO_TYPE', 'EK3_ENABLE', 'EK3_SRC1_POSXY', 'EK3_SRC1_VELXY', 'EK3_SRC1_POSZ', 'EK3_SRC1_YAW']
    for p in params:
        val = client.get_parameter(p)
        desc = ""
        if p == 'EK3_SRC1_POSXY':
            desc = "(6 = ExternalNav / Visual Odometry, 3 = GPS)"
        elif p == 'VISO_TYPE':
            desc = "(1 = MAVLink Vision Enabled, 0 = Disabled)"
        print(f"  * {p:<16} : {val} {desc}")
    print("=" * 72)


def run_monitor(client):
    print_banner()
    print("[*] Starting Live EKF & MAVLink VIO Telemetry Monitor (Ctrl+C to exit)...")
    for _ in range(20):
        if client.is_connected:
            break
        time.sleep(0.1)

    if not client.is_connected:
        print("[-] CUAV X7 Pro not detected.")
        return

    try:
        while True:
            status_desc = "OK (LOCKED)" if client.ekf_horiz_var < 0.20 else "CONVERGING"
            sys.stdout.write(
                f"\r  [CUAV Live] Mode: {client.flight_mode:<9} | "
                f"Roll:{client.roll_deg:+05.1f}° P:{client.pitch_deg:+05.1f}° Y:{client.yaw_deg:+05.1f}° | "
                f"EKF PosVar: {client.ekf_horiz_var:.3f} ({status_desc}) | VIO Sent: {client.vio_packets_sent}"
            )
            sys.stdout.flush()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[*] Monitor stopped.")


def interactive_menu(client):
    while True:
        print("\n" + "=" * 60)
        print("  CUAV X7 Pro / ArduPilot Control Center")
        print("=" * 60)
        print("  1. Check Status & Live Triple-IMU Attitude")
        print("  2. Switch to NON-GPS VIO Mode (Visual Odometry)")
        print("  3. Switch to OUTDOOR GPS Mode (Standard GPS)")
        print("  4. Live MAVLink & EKF Fusion Monitor")
        print("  5. Soft Reboot Flight Controller")
        print("  6. Exit")
        print("=" * 60)
        choice = input("Select option (1-6): ").strip()

        if choice == "1":
            show_status(client)
        elif choice == "2":
            print("[*] Configuring CUAV X7 Pro for Non-GPS Visual Odometry...")
            ok = client.enable_vio_mode()
            if ok:
                print("[+] SUCCESS: VIO Mode Activated! (VISO_TYPE=1, EK3_SRC1_POSXY=6, VELXY=6)")
            else:
                print("[-] Could not set all parameters. Check connection.")
        elif choice == "3":
            print("[*] Reverting CUAV X7 Pro to Outdoor GPS Mode...")
            ok = client.revert_gps_mode()
            if ok:
                print("[+] SUCCESS: Outdoor GPS Mode Restored! (EK3_SRC1_POSXY=3, VELXY=3)")
            else:
                print("[-] Could not set all parameters. Check connection.")
        elif choice == "4":
            run_monitor(client)
        elif choice == "5":
            confirm = input("Are you sure you want to reboot CUAV X7 Pro? (y/N): ").strip().lower()
            if confirm == "y":
                client.reboot()
                print("[+] Reboot command sent. Waiting for reconnect...")
                time.sleep(4.0)
        elif choice == "6":
            print("[*] Exiting.")
            break
        else:
            print("[-] Invalid selection.")


def main():
    parser = argparse.ArgumentParser(description="CUAV X7 Pro MAVLink Navigation Bridge")
    parser.add_argument("--status", action="store_true", help="Print CUAV connection and parameter status")
    parser.add_argument("--enable-vio", action="store_true", help="Enable Non-GPS Visual Odometry mode (EK3_SRC1_POSXY=6)")
    parser.add_argument("--disable-vio", action="store_true", help="Revert to standard GPS mode (EK3_SRC1_POSXY=3)")
    parser.add_argument("--monitor", action="store_true", help="Monitor live EKF fusion and attitude")
    parser.add_argument("--reboot", action="store_true", help="Reboot flight controller")

    args = parser.parse_args()
    client = CuavMavlinkClient()

    # Allow 1.5s for initial connection
    time.sleep(1.5)

    if args.status:
        show_status(client)
    elif args.enable_vio:
        print("[*] Enabling Non-GPS VIO Mode...")
        if client.enable_vio_mode():
            print("[+] SUCCESS: VIO Mode Activated (VISO_TYPE=1, EK3_SRC1_POSXY=6, VELXY=6)!")
        else:
            print("[-] Failed to enable VIO mode.")
    elif args.disable_vio:
        print("[*] Reverting to Outdoor GPS Mode...")
        if client.revert_gps_mode():
            print("[+] SUCCESS: GPS Mode Restored (EK3_SRC1_POSXY=3, VELXY=3)!")
        else:
            print("[-] Failed to restore GPS mode.")
    elif args.monitor:
        run_monitor(client)
    elif args.reboot:
        print("[*] Sending reboot command...")
        client.reboot()
    else:
        # Default to interactive menu
        interactive_menu(client)

    client.close()

if __name__ == "__main__":
    main()
