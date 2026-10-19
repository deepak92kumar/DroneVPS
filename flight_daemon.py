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
import fcntl
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
from flow_shadow import LongBaselineTracks, JetsonLKWorker, px_to_ned_velocity

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

def pick_origin(user_lat, user_lon, default_lat, default_lon):
    """EKF origin from SCR_USER1/2 (decimal degrees, set on the RC), else the default site.

    Returns (lat, lon, "USER" | "DEFAULT"). 0/missing/out-of-range values fall back to the default.
    """
    try:
        lat = float(user_lat)
        lon = float(user_lon)
    except (TypeError, ValueError):
        return default_lat, default_lon, "DEFAULT"
    if (math.isfinite(lat) and math.isfinite(lon)
            and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0
            and abs(lat) > 1e-6 and abs(lon) > 1e-6):
        return lat, lon, "USER"
    return default_lat, default_lon, "DEFAULT"


# Ground/airborne thresholds for the analog TF03. On the ground (lens 12.5 cm up) the FC reports
# ~0.35 m, i.e. 0.22 m AGL: at that range one ADC step is ~10 cm and the sensor is at its 0.1 m
# minimum, so near-ground readings carry ~0.25 m of uncertainty. The old 0.06/0.18 m thresholds
# would have called the drone airborne while it stood on the ground.
GROUND_AGL_M = 0.35
AIRBORNE_AGL_M = 0.60
AIRBORNE_HOLD_AGL_M = 0.45


def ground_state(agl_m, armed, was_airborne):
    """(on_ground, is_airborne) from TF03 AGL, with a latch so it does not toggle in the air."""
    on_ground = agl_m <= GROUND_AGL_M
    airborne = ((armed and agl_m >= AIRBORNE_AGL_M)
                or (agl_m >= AIRBORNE_AGL_M + 0.20)
                or (was_airborne and armed and agl_m >= AIRBORNE_HOLD_AGL_M))
    return on_ground, airborne


# Height used to scale the flow. The old 50 m ceiling silently under-scaled every velocity above
# it: at 61.9 m on 17 Sep the drone was told it was at 50 m, so the velocity sent to the EKF was
# 24 % too small. The TF03's 180 m rating is for 90 % reflectivity; over grass the datasheet gives
# about 70 m, so the lidar is faded out over 55-70 m and the barometer carries the height above it.
MAX_FLOW_ALT_M = 120.0
LIDAR_FULL_TRUST_M = 55.0
LIDAR_FADE_TOP_M = 70.0


def lidar_weight(lidar_agl_m, full_trust_m=LIDAR_FULL_TRUST_M, fade_top_m=LIDAR_FADE_TOP_M):
    """How much of the height comes from the lidar: 1.0 up to full_trust_m, 0.0 above fade_top_m,
    linear between, so the handover to the barometer has no step in it."""
    if lidar_agl_m <= full_trust_m:
        return 1.0
    if lidar_agl_m >= fade_top_m:
        return 0.0
    return (fade_top_m - float(lidar_agl_m)) / (fade_top_m - full_trust_m)


def baro_offset_trusted(lidar_agl_m, lidar_prev_m, dt_s, weight, airborne, armed, max_rate_ms=0.5):
    """Should this sample be used to learn the barometer offset?

    Flight 33 showed the offset swinging 0.26-1.60 m in fifteen seconds during a climb: the
    autopilot's height arrives a few times a second while the lidar reading is current, so during
    a climb their difference is mostly the climb rate, not the terrain. Only learn while the lidar
    alone carries the height and the drone is holding altitude.
    """
    if not (airborne and armed and weight >= 1.0):
        return False
    if lidar_prev_m is None or dt_s <= 0.0:
        return False
    return abs(float(lidar_agl_m) - float(lidar_prev_m)) / dt_s <= max_rate_ms


def height_source_changed(kind, last_kind, weight, last_weight, band=0.25):
    """Is this height-source change worth a log line? Flight 33 wrote twenty lines flip-flopping
    between TF03 and TF03+BARO(100%) as the height crossed 55.0 m. Report a kind change only once
    the blend has actually moved, so the boundary itself stays quiet."""
    if kind != last_kind and (last_weight is None or abs(weight - last_weight) >= band):
        return True
    return last_weight is not None and abs(weight - last_weight) >= band


def track_baro_offset(offset, lidar_agl_m, baro_rel_m, trusted, tau_samples=200.0, max_step_m=15.0):
    """Learn (lidar AGL - barometer height above home) while the lidar is trusted.

    The lidar measures the ground below; the barometer measures the takeoff point. Without this
    offset the handover would step by whatever the terrain has done since takeoff. Learned slowly
    so a single bad lidar sample cannot move it.
    """
    if not trusted:
        return offset
    meas = float(lidar_agl_m) - float(baro_rel_m)
    if offset is None:
        return meas
    if abs(meas - offset) > max_step_m:
        # Flight 54: a 174 m lidar spike is worth 18 m of offset in a fifth of a second, because
        # the filter moves by a share of the error and that error was enormous. Terrain does not
        # do this; only a bad reading does.
        return offset
    a = 1.0 / max(1.0, tau_samples)
    return (1.0 - a) * offset + a * meas


def lidar_agrees_with_baro(lidar_agl_m, baro_agl_m, tol_m=8.0, tol_frac=0.15):
    """Is the lidar reading consistent with the barometer's AGL? Log 29 carried an 81.7 m spike
    while the drone hovered at 4 m; above the ground gate that spike would set the flow scale."""
    if baro_agl_m is None:
        return True
    return abs(float(lidar_agl_m) - float(baro_agl_m)) <= tol_m + tol_frac * float(baro_agl_m)


def flow_confidence(height_m, err_per_m=0.05, err_min=0.1, err_max=0.9):
    """Confidence 0-100 for VISION_POSITION_DELTA, from the flow error measured in flight.

    ArduPilot maps confidence 100 to EK3_VIS_VERR_MIN (0.1 m/s) and 0 to EK3_VIS_VERR_MAX (0.9),
    which is the only way to make the EKF's trust follow the height. Flights 26-28 measured roughly
    0.15 m/s at 3 m, 0.4 at 10 m and 0.9-1.9 at 20 m, i.e. about 0.05 m/s per metre of height.
    """
    err = min(max(err_per_m * max(float(height_m), 0.0), err_min), err_max)
    return max(0.0, min(100.0, 100.0 * (err_max - err) / (err_max - err_min)))


# Every marker the drone may see, with the width of its black square in metres. The 20 x 20 m
# field carries fifteen: a nested centre target (45 / 16 / 0) and twelve 1.2 m markers on the grid.
# ID 15 is the standalone 1.13 m pad - it cannot stay ID 0, because ID 0 is now the 0.12 m marker
# in the middle of the field target and a single size for both would put the pose out by 9x.
# Velocity from the long-baseline tracker instead of one frame pair. Measured over 216 856
# airborne samples (17-18 Sep): below about 30 m the 87 ms baseline cuts the velocity noise by
# roughly a factor of two, because there the error is mostly fixed per-frame pixel noise and a
# longer baseline divides it down. Above 40 m it stops helping and then hurts - up there the
# error is dominated by rotation and timing, which do not shrink with the baseline, while the
# drone's real motion over 87 ms starts to be blurred away.
#
#   height    live      long baseline     gain
#    5-15 m   1.21 m/s     0.57 m/s       2.14x
#   20-30 m   2.43 m/s     1.25 m/s       1.95x
#   30-40 m   2.82 m/s     1.62 m/s       1.74x
#   40-50 m   3.09 m/s     2.82 m/s       1.09x
#   50-60 m   3.22 m/s     3.43 m/s       0.94x   - worse
LB_FULL_M = 30.0
LB_NONE_M = 40.0


def long_baseline_weight(height_m, full_m=LB_FULL_M, none_m=LB_NONE_M):
    """How much of the velocity comes from the long-baseline tracker, by height."""
    if height_m <= full_m:
        return 1.0
    if height_m >= none_m:
        return 0.0
    return (none_m - float(height_m)) / (none_m - full_m)


def blend_velocity(live_n, live_e, lb_n, lb_e, weight):
    """Mix the two velocity estimates. The long-baseline one is dropped whenever it is missing
    or not finite, so a gap in the tracker can never blank the velocity sent to the autopilot."""
    if weight <= 0.0 or lb_n is None or lb_e is None:
        return float(live_n), float(live_e), 0.0
    if not (math.isfinite(lb_n) and math.isfinite(lb_e)):
        return float(live_n), float(live_e), 0.0
    w = max(0.0, min(1.0, float(weight)))
    return (w * float(lb_n) + (1.0 - w) * float(live_n),
            w * float(lb_e) + (1.0 - w) * float(live_e), w)


# Camera watchdog. On 18 Sep (flight 36) the OAK dropped off the USB bus at 25 m. The main loop
# blocked inside the camera call, so the daemon simply went quiet: vision stopped, and the pilot
# learned of it only nine seconds later, from the autopilot's EKF failsafe. A separate thread
# watches for frames and tells the pilot within a second. It only warns - it never touches
# the flight mode.
CAMERA_STALL_S = 1.0
CAMERA_REPEAT_S = 5.0


def camera_watch_step(state, now, last_frame_t, armed,
                      stall_s=CAMERA_STALL_S, repeat_s=CAMERA_REPEAT_S):
    """What the camera watchdog should announce now: ("lost", age_s), ("back", age_s) or None.

    state holds whether a loss is being reported and when it was last said. Only while armed:
    on the ground a paused camera is harmless and a warning would just be noise.
    """
    if last_frame_t is None:
        return None
    age = now - last_frame_t
    if state.get("lost"):
        if age < stall_s:
            state["lost"] = False
            return ("back", age)
        if armed and now - state.get("said", 0.0) >= repeat_s:
            state["said"] = now
            return ("lost", age)
        return None
    if armed and age >= stall_s:
        state["lost"] = True
        state["said"] = now
        return ("lost", age)
    return None


class CameraWatchdog(threading.Thread):
    """Warns the pilot when frames stop arriving, from outside the (possibly stuck) main loop."""

    def __init__(self, cuav, frame_clock):
        super().__init__(daemon=True)
        self.cuav = cuav
        self.frame_clock = frame_clock      # one-element list, main loop writes the last frame time
        self.state = {}
        self.running = True
        self.start()

    def run(self):
        while self.running:
            try:
                msg = camera_watch_step(self.state, time.monotonic(), self.frame_clock[0],
                                        bool(getattr(self.cuav, "is_armed", False)))
                if msg is not None:
                    kind, age = msg
                    if kind == "lost":
                        text = f">>> CAMERA LOST {age:.0f}s - TAKE ALT_HOLD"
                        logging.error(f"[!] Camera stalled: no frame for {age:.1f} s")
                        self.cuav.send_statustext(text, severity=2)
                    else:
                        logging.warning("[+] Camera frames flowing again")
                        self.cuav.send_statustext(">>> CAMERA BACK", severity=4)
            except Exception:
                pass
            time.sleep(0.2)

    def stop(self):
        self.running = False


def marker_offset_ned(tvec, roll_rad, pitch_rad, yaw_rad):
    """Where the drone is relative to a marker, in metres north and east, from the PnP translation.

    The camera looks straight down: +x right, +y toward the tail, +z down the optical axis. The
    old code turned that into north/east with the yaw alone. That is fine for a level drone but
    not for a swinging one: at 26 m a 10 degree roll shifts the marker h*sin(10 deg) = 4.5 m
    across the image, and yaw-only maths reads that as the drone having moved 4.5 m. The full attitude is used
    here so a tilt is recognised as a tilt.
    """
    m_ned = marker_ned(tvec, roll_rad, pitch_rad, yaw_rad)
    return -m_ned[0], -m_ned[1]                   # the drone relative to the marker


def marker_ned(tvec, roll_rad, pitch_rad, yaw_rad):
    """The marker relative to the drone, north-east-down, in metres. Down is the drone's height
    above the marker, which lets the pose be checked against the lidar."""
    tx, ty, tz = (float(v) for v in np.asarray(tvec).ravel()[:3])
    m_body = np.array([-ty, tx, tz])              # marker in body axes: forward, right, down
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    r_nb = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])
    m = r_nb @ m_body
    return float(m[0]), float(m[1]), float(m[2])


def marker_offset_yaw_only(tvec, yaw_rad):
    """The offset as the daemon has always computed it, kept for side-by-side logging."""
    tx, ty = float(np.asarray(tvec).ravel()[0]), float(np.asarray(tvec).ravel()[1])
    fwd, right = ty, -tx
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return fwd * c - right * s, fwd * s + right * c


# Marker map, stage 2: let the marker hold the position the EKF is given. Flight 37 measured the
# flow position wandering 5-6 m from the marker at 35-42 m while the marker itself jittered only
# 33 cm. Only the nested centre target is used - its three markers share one point on the ground;
# the twelve distributed ones sit elsewhere and need the full map before they can be mixed in.
MARKER_POS_IDS = frozenset({45, 16, 0})
MARKER_POS_MIN_H_M = 2.0
MARKER_POS_MAX_H_M = 100.0       # stage 3: ID 45 (3.6 m) still decodes at 11 px to ~130 m
MARKER_POS_HEIGHT_TOL = 0.25     # the pose's height must agree with the lidar to 25 %
MARKER_POS_MAX_TILT_DEG = 25.0
# Fastest the marker may move the sent position (m/s). Fading this to 1 m/s above 60 m was tried
# on 20 Sep to make the corrections gentler, and it broke the hover above 50 m: the flow position
# drifts against the marker at about 3 m/s at 50-60 m and 4.8 m/s at 60-80 m, so the limiter
# saturated (the 90th percentile of the correction sat exactly on the limit), the sent position
# ran away with the flow and the marker could only re-anchor after 6 m. Gentleness has to come
# from the error declared to the EKF, not from starving the correction.
MARKER_CORR_RATE_MS = 4.0


def marker_corr_rate_ms(height_m):
    """How fast the marker may move the sent position at this height (m/s)."""
    return MARKER_CORR_RATE_MS


MARKER_REANCHOR_M = 6.0          # a disagreement larger than this is not corrected, only re-anchored
# Both of these are flat on purpose. Letting them grow with height was tried on 21 Sep and cost
# two flights. Raising the rate (flight 59) slews the sent position faster than the IMU can
# follow. Raising the re-anchor threshold (flight 60) is worse in a quieter way: re-anchoring is
# what stops the correction pulling one way, and without it the position was dragged 9.6 m in
# five seconds at 70-95 m against 4.8 m before, the test ratio went 0.11 -> 1.07, and the FC
# dropped to ALT_HOLD on an EKF variance failsafe at 72 m. Giving up on an error is ugly but it
# is gentler on the EKF than chasing it. Flight 58, with both flat, is still the best flight:
# 101 m, roll rms 4.0-4.3 deg and test ratio 0.10-0.11 from 65 to 95 m.
FLOW_RESET_JUMP_M = 3.0          # a flow step larger than this is a reset, not motion
MARKER_POS_ERR_M = 0.5           # position error declared to the EKF while the marker holds it
# Stage 3: above 25 m the error is set by the attitude, not the pixels - 1 deg is 0.017*h metres, and
# flights 38/39 stepped 2-7 deg at every EKF lane switch. 2 % of height: 1 m at 50 m, 2 m at 100 m.
MARKER_POS_ERR_PER_M = 0.02
MARKER_FRESH_S = 0.5
# Flight 46: at 82 m the marker went quiet for half a second, the flow velocity came back, and the
# drone swung to 28 deg of roll within seconds - the flow velocity is 2-3 m/s wrong up there, so
# restoring it is worse than a gap in the marker fixes. The wait before the velocity returns
# therefore grows with height; the EKF test-ratio guard still brings it back if the EKF is in
# real trouble, and a marker lost for good still ends the wait.
MARKER_HOLD_NEAR_S = 2.0          # ...starting here the moment the flow velocity is switched off
MARKER_HOLD_FAR_S = 5.0
MARKER_HOLD_FULL_H_M = 70.0       # full patience at this height, MARKER_FRESH_S at 35 m and below


def marker_hold_window_s(height_m):
    """How long a marker fix keeps counting as 'the marker is holding the position'."""
    if height_m <= FLOW_VEL_OFF_H_M:
        return MARKER_FRESH_S
    a = min(1.0, (height_m - FLOW_VEL_OFF_H_M) / max(1e-6, MARKER_HOLD_FULL_H_M - FLOW_VEL_OFF_H_M))
    return MARKER_HOLD_NEAR_S + a * (MARKER_HOLD_FAR_S - MARKER_HOLD_NEAR_S)


def flow_shift_since(hist, t, max_age_s=1.0, max_jump_m=3.0):
    """How far the flow says the drone has moved since time t, from [(t, north, east)].

    A marker fix describes where the drone was when the frame was taken - 174 ms ago for the
    marker camera on 20 Sep, 60-80 ms for the flow camera. Carrying it forward removes that lag,
    but it pays the flow's own noise: flight 47 doubled the step between consecutive fixes at
    35-50 m (18 -> 36 cm) because the flow velocity is 1.5-2 m/s wrong up there, and the noisier
    position fed the EKF until the guard brought the flow velocity back and the drone swung. So
    this is behind --marker-lag-comp, off by default. Returns (dn, de), or None when the history
    does not cover t or the flow integrator was reset in between."""
    if not hist or t is None:
        return None
    t_new, n_new, e_new = hist[-1]
    if t < hist[0][0] or t_new - t > max_age_s or t > t_new:
        return None
    for i in range(len(hist) - 1, 0, -1):
        t0, n0, e0 = hist[i - 1]
        t1, n1, e1 = hist[i]
        if t0 <= t <= t1:
            a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
            n = n0 + a * (n1 - n0)
            e = e0 + a * (e1 - e0)
            dn, de = n_new - n, e_new - e
            return None if math.hypot(dn, de) > max_jump_m else (dn, de)
    return None


def marker_pos_err_m(height_m):
    """Position error (m) declared to the EKF while the marker holds the position at this height."""
    return max(MARKER_POS_ERR_M, MARKER_POS_ERR_PER_M * max(0.0, height_m))


# Flight 41: with the marker holding the position the drone still swung above 40 m, because the
# flow velocity was off by 1.4 m/s rms at 30-40 m and 2.3 m/s at 40-55 m while the EKF is told
# 0.7 m/s (VISO_VEL_M_NSE; the message covariance is ignored). At 50 m one pixel is 13 cm, so the
# flow cannot resolve velocity there. Above this height, while the marker holds the position, the
# flow velocity is not sent and the EKF runs on the marker position and its IMU - as with a GPS.
# EKF3 (PosVelFusion/Control) only sets velTimeout on loss of position aiding, so stopping and
# resuming the velocity causes no velocity reset while the marker position keeps fusing.
FLOW_VEL_OFF_H_M = 35.0
FLOW_VEL_ON_H_M = 33.0           # hysteresis: back on below this
FLOW_VEL_OFF_HOLD_S = 1.0        # the marker must have held this long before the velocity stops
# Flight 42A: 8 s after the velocity stopped the drone really moved 10 m in 3 s. The EKF, on
# position and IMU alone, saw 3 m, then rejected the (correct) marker position as an outlier and
# dead-reckoned to -4.8 m/s until it reset itself 8 s later and the drone lurched +25/-27 deg.
# So the EKF is watched while the velocity is off, and the velocity comes back when it is in
# trouble. The trigger is the EKF's own position test ratio (EKF_STATUS_REPORT
# pos_horiz_variance = sqrt(test ratio); it rejects at 1.0): 0.8 or more. In 42A that fired at
# 240.8 s, before the rejection began and 9 s before the reset. Flight 43 showed that a plain
# position gap of max(1.5 m, 4 % h) also fires on ordinary climbs and stick moves (it brought
# the 2 m/s-wrong flow velocity back at 46 m and 90 m), so the gap is now only a backstop at
# max(3 m, 6 % h), measured against its value when the velocity stopped.
FLOW_VEL_SP_ON = 0.8
FLOW_VEL_SP_CALM = 0.5           # the velocity only stops while the test ratio is below this
# The gap is only a backstop for the case the test ratio somehow misses. At max(3 m, 6 % h) it
# fired on ordinary climbs at 46 m (flight 44), 51 m (46) and 43 m (47) and each time brought
# back a flow velocity that is 1.4-2.3 m/s wrong there. In the 42A divergence the test ratio
# reached 0.8 at 240.8 s while the gap only passed 3 m at 242 s, so the test ratio leads anyway.
FLOW_VEL_GAP_MIN_M = 6.0
FLOW_VEL_GAP_PER_M = 0.10
FLOW_VEL_CALM_S = 3.0            # the EKF must have been steady this long before the velocity stops
FLOW_VEL_EKF_FRESH_S = 0.5       # GLOBAL_POSITION_INT, 10 Hz
FLOW_VEL_STATUS_FRESH_S = 1.0    # EKF_STATUS_REPORT, 5 Hz
FLOW_VEL_REF_TAU_S = 2.0
FLOW_VEL_GAP_LOCKOUT_S = 10.0    # after the EKF has been in trouble once, keep the velocity on this long


def flow_vel_gap_limit(height_m):
    return max(FLOW_VEL_GAP_MIN_M, FLOW_VEL_GAP_PER_M * max(0.0, height_m))


# Modes where the pilot is flying by hand. Flight 67: after an EKF failsafe dropped the drone
# into ALT_HOLD at 56 m, both lanes' POSITION test ratios recovered within five seconds and the
# drone was calm - but the VELOCITY ratio kept bouncing 1.03-1.37 against FS_EKF_THRESH of 1.20,
# because the marker had been lost and the guard had brought the flow velocity back, and up there
# the flow velocity is 1.4-2.3 m/s wrong. The failsafe needs the variances to stay down to clear,
# so it never did: "EKF Failsafe Cleared" came at 0.3 m, forty-eight seconds later, on the ground.
# Until then ArduPilot refuses LOITER, which is exactly what the pilot sees - ALT_HOLD taken by
# hand, the drone steady again, and no way back into LOITER short of landing.
# So while the pilot is flying by hand we send no velocity at all. The EKF then has nothing to
# disagree with, the variance decays and LOITER comes back. Position keeps going, so there is an
# estimate ready for it. This cannot disturb a position hold: it only applies when none is running.
VEL_MANUAL_MODES = frozenset({"STABILIZE", "ALT_HOLD", "ACRO", "SPORT", "DRIFT", "FLIP", "THROW"})


# Warning only, like the camera watchdog: it sends a message and touches nothing else.
# Flight 68 climb 1 drifted from the marker while the RC quietly showed PAD# going 16 -> 25 -> 30
# -> 36 -> 39 m; the marker was lost thirteen times and the flow velocity came back ten times.
# Climb 2, flown straight up over the marker, never lost it once and reached 78 m. The number was
# always on the screen - it just had to be read. So say it out loud, with the way to fly.
MARKER_FAR_WARN_M = 12.0          # say it past here
MARKER_FAR_CLEAR_M = 8.0          # and stop once back inside here (hysteresis)
MARKER_FAR_REPEAT_S = 4.0         # keep saying it while it lasts
_COMPASS_8 = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def bearing_home_to_marker(north_m, east_m):
    """Which way to fly to get back over the marker, as an 8-point compass word.

    north_m/east_m is the drone relative to the marker, so the way home is the opposite."""
    brg = math.degrees(math.atan2(-float(east_m), -float(north_m))) % 360.0
    return _COMPASS_8[int((brg + 22.5) // 45) % 8]


def marker_far_step(st, north_m, east_m, now, airborne, have_fix,
                    warn_m=MARKER_FAR_WARN_M, clear_m=MARKER_FAR_CLEAR_M,
                    repeat_s=MARKER_FAR_REPEAT_S):
    """The warning to say now, or None. st: dict, empty on the first call."""
    if not (airborne and have_fix):
        st.clear()
        return None
    d = math.hypot(float(north_m), float(east_m))
    if st.get("on"):
        if d <= clear_m:
            st.clear()
            return f">>> MARKER OK {d:.0f}m"
    elif d < warn_m:
        return None
    else:
        st["on"] = True
        st["said"] = -1e9
    if now - st.get("said", -1e9) < repeat_s:
        return None
    st["said"] = now
    return f">>> MARKER {d:.0f}m | GO {bearing_home_to_marker(north_m, east_m)}"


def flow_velocity_blocked_by_mode(flight_mode, airborne):
    """True when the pilot has taken manual control and the flow velocity must not be sent."""
    return bool(airborne) and str(flight_mode or "").strip().upper() in VEL_MANUAL_MODES


def flow_velocity_wanted(st, now, height_m, marker_holds, ekf_gap=None, ekf_sp=None,
                         off_h=FLOW_VEL_OFF_H_M, on_h=FLOW_VEL_ON_H_M, hold_s=FLOW_VEL_OFF_HOLD_S,
                         calm_s=FLOW_VEL_CALM_S):
    """True when the flow velocity should be sent.

    ekf_gap: (north, east) of the EKF's own position minus the position sent; ekf_sp: the EKF's
    horizontal position test ratio (sqrt). Either None means it is not fresh - then the velocity
    is always sent. It stops only above off_h after the marker has held for hold_s and the EKF
    has been steady (test ratio under FLOW_VEL_SP_CALM, gap not moving) for calm_s, and comes
    back below on_h, the moment the marker stops holding, when the test ratio reaches
    FLOW_VEL_SP_ON, or when the gap moves more than flow_vel_gap_limit(h) from where it was
    when the velocity stopped. st["why"] says why it last came back."""
    lim = flow_vel_gap_limit(height_m)
    if not marker_holds or ekf_gap is None or ekf_sp is None:
        if st.get("off"):
            st["why"] = "marker lost" if not marker_holds else "no EKF status"
        st.update(off=False, since=None, ref=None, calm=None)
        return True
    gn, ge = float(ekf_gap[0]), float(ekf_gap[1])
    sp = float(ekf_sp)
    if st.get("since") is None:
        st["since"] = now
    if st.get("off"):
        b = st["base"]
        moved = math.hypot(gn - b[0], ge - b[1])
        why = None
        if sp >= FLOW_VEL_SP_ON:
            why = f"EKF test ratio {sp:.2f}"
        elif moved > lim:
            why = f"EKF off by {moved:.1f} m"
        if why is not None:
            st.update(off=False, why=why, ref=(gn, ge), calm=None, t=now,
                      lock_until=now + FLOW_VEL_GAP_LOCKOUT_S)
        elif height_m < on_h:
            st.update(off=False, why="below", ref=(gn, ge), calm=None, t=now)
        return not st["off"]
    # Velocity on: follow the gap slowly and wait for the EKF to be steady.
    dt = max(0.0, now - st.get("t", now))
    st["t"] = now
    if st.get("ref") is None:
        st["ref"] = (gn, ge)
    else:
        a = min(1.0, dt / FLOW_VEL_REF_TAU_S)
        st["ref"] = (st["ref"][0] + a * (gn - st["ref"][0]), st["ref"][1] + a * (ge - st["ref"][1]))
    steady = (sp < FLOW_VEL_SP_CALM
              and math.hypot(gn - st["ref"][0], ge - st["ref"][1]) < 0.5 * lim)
    if not steady:
        st["calm"] = None
    elif st.get("calm") is None:
        st["calm"] = now
    if (height_m >= off_h and now - st["since"] >= hold_s and now >= st.get("lock_until", 0.0)
            and st.get("calm") is not None and now - st["calm"] >= calm_s):
        st.update(off=True, base=st["ref"])
    return not st.get("off", False)


def marker_fix_usable(mid, height_m, marker_down_m, roll_deg, pitch_deg, confirmed, airborne, armed,
                      min_h=MARKER_POS_MIN_H_M, max_h=MARKER_POS_MAX_H_M, tol=MARKER_POS_HEIGHT_TOL,
                      max_tilt=MARKER_POS_MAX_TILT_DEG):
    """(ok, reason). Every condition is one that, if false, has a known way of making the pose wrong."""
    if mid not in MARKER_POS_IDS:
        return False, "not a centre marker"
    if not (airborne and armed):
        return False, "not flying"
    if not confirmed:
        return False, "not confirmed"
    if not (min_h <= height_m <= max_h):
        return False, f"height {height_m:.1f} m outside {min_h:.0f}-{max_h:.0f} m"
    if abs(roll_deg) > max_tilt or abs(pitch_deg) > max_tilt:
        return False, "tilt"
    if height_m <= 0 or abs(marker_down_m - height_m) > tol * height_m:
        return False, f"pose height {marker_down_m:.1f} m vs lidar {height_m:.1f} m"
    return True, ""


def marker_fix_step(st, flow_n, flow_e, mk, dt, rate=MARKER_CORR_RATE_MS,
                    reanchor=MARKER_REANCHOR_M, reset_jump=FLOW_RESET_JUMP_M, source=None):
    """One update of the horizontal position sent to the EKF once the marker has taken part.

    The sent position is its own integrator: it moves with the flow between marker fixes, and each
    fix pulls it toward where the marker says the drone is. Three things make it safe to switch on
    in the air:
      * the first fix sets an anchor so that nothing moves - no jump when the marker locks;
      * a single fix can move it by at most rate*dt, so one bad detection costs centimetres;
      * a flow reset (the daemon zeroes its integral in several places) is absorbed, and a
        disagreement bigger than `reanchor` is re-anchored rather than chased.

    st: dict, empty on the first call. mk: (north, east) of the drone relative to the marker, or
    None when there is no usable fix. source: which camera the fix came from. The two cameras
    differ by their calibration (0.11-0.21 m on 20 Sep), so each keeps its own anchor: swapping
    between them then moves nothing and costs nothing, however often it happens.
    Returns (north, east, how).
    """
    if st.get("anchor") is None and mk is None:
        # Until the marker has locked once, change nothing at all: send the flow exactly as the
        # daemon always has, resets and all.
        st["sent"] = (float(flow_n), float(flow_e))
        st["flow"] = (float(flow_n), float(flow_e))
        return float(flow_n), float(flow_e), "flow"
    if st.get("sent") is None:
        st["sent"] = (float(flow_n), float(flow_e))
        st["flow"] = (float(flow_n), float(flow_e))
    dfn, dfe = flow_n - st["flow"][0], flow_e - st["flow"][1]
    st["flow"] = (float(flow_n), float(flow_e))
    if math.hypot(dfn, dfe) > reset_jump:
        dfn = dfe = 0.0                                  # the flow was reset, not moved
    sn, se = st["sent"][0] + dfn, st["sent"][1] + dfe
    how = "flow"
    anchors = st.setdefault("anchors", {})
    if mk is not None:
        if st.get("anchor") is None:
            st["anchor"] = anchors[source] = (sn - mk[0], se - mk[1])
            how = "anchor"
        elif source not in anchors:
            # First fix from this camera: anchor it where the position already is.
            st["anchor"] = anchors[source] = (sn - mk[0], se - mk[1])
            how = "source"
        else:
            st["anchor"] = anchors[source]
            en = mk[0] + st["anchor"][0] - sn
            ee = mk[1] + st["anchor"][1] - se
            err = math.hypot(en, ee)
            if err > reanchor:
                st["anchor"] = anchors[source] = (sn - mk[0], se - mk[1])
                how = "reanchor"
            else:
                lim = rate * max(dt, 0.0)
                k = 1.0 if err <= lim or err == 0.0 else lim / err
                sn, se = sn + k * en, se + k * ee
                how = "marker"
    st["sent"] = (sn, se)
    return sn, se, how


MARKER_SIZES_M = {
    45: 3.60, 16: 0.72, 0: 0.12,
    10: 1.20, 18: 1.20, 22: 1.20, 1: 1.20, 2: 1.20, 3: 1.20,
    4: 1.20, 5: 1.20, 6: 1.20, 11: 1.20, 12: 1.20, 13: 1.20,
    15: 1.13,
}
MARKER_MIN_PX = 11.0     # flight 30 decoded continuously down to 11.2 px


def marker_candidates(ids, corners, sizes=None):
    """[(id, mean side in px, corners)] for every known marker in the frame, biggest first.

    The daemon used to look only for ID 0 and assume one size for it. On the field target that
    would read the 0.12 m inner marker as if it were 1.13 m wide.
    """
    sizes = MARKER_SIZES_M if sizes is None else sizes
    if ids is None or corners is None:
        return []
    out = []
    for mid, c in zip(np.asarray(ids).ravel().tolist(), corners):
        if int(mid) not in sizes:
            continue
        q = np.asarray(c, dtype=float).reshape(4, 2)
        side = float(np.mean([np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]))
        out.append((int(mid), side, q))
    out.sort(key=lambda r: -r[1])
    return out


def pick_marker(cands, min_px=MARKER_MIN_PX, sizes=None):
    """The marker to take the pose from: the largest one that is big enough to trust.

    No upper limit is needed. A marker wider than the sensor loses its corners and the detector
    never returns it, so on the nested target the outer marker simply drops out as the drone
    descends and the next one down is already the largest left. An upper pixel cap would instead
    throw away a good detection that the detector had just proved was whole.
    """
    sizes = MARKER_SIZES_M if sizes is None else sizes
    usable = [c for c in cands if c[1] >= min_px]
    return usable[0] if usable else None


def ir_flood_target(on_now, height_m, height_valid=True, off_above_m=2.5, on_below_m=1.8):
    """Should the IR floodlight be on? Decided by height alone.

    The 700 mA flood reaches a couple of metres, so above ~2.5 m it only costs power and heat, and
    switching it there changes nothing in the image. Brightness cannot be used to detect daylight
    here: the camera sits 3.5 cm above ground, so with the flood on it reads its own light (100 us
    exposure) and with it off it reads the drone's shadow (1500 us) - the rule would oscillate.
    """
    if not height_valid:
        return on_now
    if height_m > off_above_m:
        return False
    if height_m < on_below_m:
        return True
    return on_now


# A TF03 reading can only rise as fast as the drone climbs. Flight 54 carried five upward spikes
# of 0.0-0.2 s, one of them to 174 m while the drone hovered at 52 m, all flagged good by the
# sensor. They went straight through the old filter, which only looked downward, and one of them
# dragged the barometer offset 18 m out - after which the height was 22 m too high, the flow was
# scaled by it and the marker's own pose was rejected for disagreeing with it.
LIDAR_MAX_RISE_MS = 6.0          # the drone climbs at 3 m/s; twice that is generous
LIDAR_RISE_ALLOW_M = 3.0         # plus a step for terrain


def tf03_glitch_hold(state, tf03_h, tf03_valid, airborne, armed, now, hold_s=0.5):
    """Change T: hold the last good TF03 height through a short glitch in flight.

    Log 24: one bad sample at 4 m (neighbours 3.97/4.05 m) muted vision for 2.5 s and wiped XY.
    In flight above 1 m, a reading that is invalid, below half the last good height, or above it
    by more than the drone could have climbed, is replaced by the last good height for up to
    hold_s. A change that outlasts the hold is accepted, so real ground still gets through.
    state: dict with 'h' and 't'. Returns (h, valid, held).
    """
    last_h, last_t = state.get("h", 0.0), state.get("t", 0.0)
    rise_room = LIDAR_RISE_ALLOW_M + LIDAR_MAX_RISE_MS * max(0.0, now - last_t)
    implausible = airborne and last_h > 1.0 and tf03_valid and (
        tf03_h < 0.5 * last_h or tf03_h > last_h + rise_room)
    if tf03_valid and not implausible:
        state["h"], state["t"] = tf03_h, now
        return tf03_h, True, False
    if airborne and armed and last_h > 1.0 and now - last_t <= hold_s:
        return last_h, True, True
    if tf03_valid:
        # The change outlasted the hold (e.g. flying over a roof or vehicle): accept the new height.
        state["h"], state["t"] = tf03_h, now
        return tf03_h, True, False
    return tf03_h, False, False


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


# Marker camera (--marker-cam): RGB centre crop, used for the marker position from MARKER_CAM_MIN_H_M up.
# 30, not 10: the RGB frame reached the fix 163-172 ms old in every flight logged, and a
# rolling-shutter sensor shears more the slower it reads out - the marker's skew tracked the
# drone's turn rate (r = +0.46 over seven flights, 1.045 at 3 deg/s against 1.128 at 6 deg/s).
# 15 fps was flown first and settled the question: skew 1.128 -> 1.046, reprojection error
# 4.84 -> 1.57 px, frame age 172 -> 147 ms, fixes used 89 % -> 98 %, and the saved frames stopped
# showing curved marker edges and wavy furrows. 30 fps takes the frame period to 33 ms. Detection
# stays capped at MARKER_CAM_DETECT_HZ, so the extra frames only make the chosen one fresher;
# the capture and crop run on the camera's own processor. Measured on the bench with the FC
# connected: flow loop 99-100 FPS, zero frame drops, latency 13.6 ms - no cost.
MARKER_CAM_FPS = 30
# Lens position. A ground sweep on 20 Sep peaked at 120 on a scene a few metres away, and with
# this lens (f=4.8 mm, f/2.0, 2.3 um pixels after the 2/3 ISP scale) the hyperfocal distance is
# about 5 m, so a lens focused there is sharp from ~2.5 m to infinity. Flight 45 flew at lens 0,
# far off that peak, and its images were soft: no marker was read above 41 m.
MARKER_CAM_FOCUS = 120
MARKER_CAM_ISP_NUM, MARKER_CAM_ISP_DEN = 2, 3     # 3840x2160 -> 2560x1440
MARKER_CAM_ISP_W, MARKER_CAM_ISP_H = 2560, 1440
MARKER_CAM_W, MARKER_CAM_H = 1280, 800
MARKER_CAM_MIN_H_M = 20.0
MARKER_CAM_PROC_H_M = 15.0        # frames are not even decoded below this (ground test: -10 % flow fps)
# Left at 10. Raising it to 20 with the camera at 30 fps was tried on the bench and taken back
# out: with the FC connected that run logged 345 dropped frames and five frame-drop events,
# where 10 Hz runs clean at zero. The detector is not the limit anyway - detectMarkers on a
# 1280x800 frame takes 28.6 ms - and what actually starved the fixes was frames failing to
# decode at all, which the higher capture rate already fixed (RGB fixes used 89 % -> 98 %).
MARKER_CAM_DETECT_HZ = 10.0
MARKER_CAM_CONFIRM_S = 1.0        # it detects at 10 Hz, so 3 hits need a longer window than the flow camera
MARKER_CAM_AF_H_M = 25.0          # autofocus once at this height (the scene there really is at
MARKER_CAM_AF_SETTLE_S = 2.5      # infinity and the marker is the only contrast) and lock the result
MARKER_CAM_AF_MIN, MARKER_CAM_AF_MAX = 90, 150   # outside this the result is not believed
# How recently the marker camera must have delivered before its fix is preferred over the flow
# camera's. This was 1 s, from before each source had its own anchor: in flight 48 the marker
# camera had 26 gaps longer than 0.6 s (51 s in total) during the first climb, the flow camera
# saw the marker through every one of them at 23 fixes/s, and all of those were thrown away. The
# position went undriven, the daemon called the marker lost at 36 m, the flow velocity came back
# and the drone swung. Swapping source costs nothing now, so this only breaks a tie.
MARKER_CAM_FRESH_S = 0.2
# The EEPROM extrinsics are not quite right, and flight 64 measured by how much. Comparing the
# two cameras' fixes on the same drone, the difference sits entirely on the drone's RIGHT axis
# and is a constant ANGLE, not a constant distance - 0.36 m at 40-60 m, 0.49 at 60-80, 0.68 at
# 80-95, which is h*tan(0.45 deg) all the way up. The nose axis shows nothing (0.01-0.04 m). A
# per-source anchor absorbs a fixed offset but not one that grows, so an anchor set on the climb
# is a third of a metre wrong by 90 m. Replaying the same 2549 fixes with this rotation applied
# leaves +0.02, +0.02 and -0.05 m over those three bands. It is a calibration number measured
# from flight, not a tuning knob: if it is right the two cameras agree at every height, and the
# next log says so directly.
MARKER_CAM_EXTRIN_FIX_DEG = -0.45          # about the marker camera's y axis (the drone's nose)


def _extrin_fix(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t), 0.0, math.sin(t)],
                     [0.0, 1.0, 0.0],
                     [-math.sin(t), 0.0, math.cos(t)]])


MARKER_CAM_EXTRIN_FIX = _extrin_fix(MARKER_CAM_EXTRIN_FIX_DEG)
MARKER_ATT_MAX_AGE_S = 0.15       # attitude history must reach this close to the frame time


def crop_intrinsics(K, x0, y0):
    """Camera matrix of a crop starting at pixel (x0, y0) of the image K belongs to."""
    Kc = np.array(K, dtype=np.float64).copy()
    Kc[0, 2] -= x0
    Kc[1, 2] -= y0
    return Kc


def to_flow_camera_frame(tvec, R, t_m):
    """A point in the marker camera's frame, expressed in the flow camera's (CAM_B) frame.

    R, t_m: the EEPROM extrinsics from the marker camera to CAM_B (translation in metres). On
    this unit that is 3.7 cm and 0.70 deg, of which 0.49 deg is tilt (0.37 about x, 0.33 about
    y) - 0.8 m at 95 m if it were ignored; the rest turns about the optical axis."""
    v = np.asarray(tvec, dtype=np.float64).ravel()[:3]
    return np.asarray(R, dtype=np.float64) @ v + np.asarray(t_m, dtype=np.float64).ravel()[:3]


def attitude_at_time(hist, t, max_age=MARKER_ATT_MAX_AGE_S):
    """(roll, pitch, yaw) deg at monotonic time t, interpolated from [(t, roll, pitch, yaw)].

    The marker camera's frames are ~60 ms old when they arrive and ~85 ms by the time the pose is
    solved; at 10 deg/s of swing the current attitude would be ~0.8 deg off, 1.3 m at 95 m.
    None when the history does not reach t (then the caller falls back to the current attitude)."""
    if not hist:
        return None
    if t <= hist[0][0]:
        return tuple(hist[0][1:4]) if hist[0][0] - t <= max_age else None
    if t >= hist[-1][0]:
        return tuple(hist[-1][1:4]) if t - hist[-1][0] <= max_age else None
    for i in range(len(hist) - 1, 0, -1):
        t0, r0, p0, y0 = hist[i - 1]
        t1, r1, p1, y1 = hist[i]
        if t0 <= t <= t1:
            if t1 - t0 > max_age:
                return None
            a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
            dy = (y1 - y0 + 180.0) % 360.0 - 180.0
            return r0 + a * (r1 - r0), p0 + a * (p1 - p0), y0 + a * dy
    return None


def marker_ground_heading(rvec, yaw_deg):
    """Heading of the marker's own X axis over the ground in degrees, plus the drone's heading
    relative to the marker.

    The marker never moves, so the first number should not change through a flight however much
    the compass drifts. On 20 Sep the compass swung 26 deg above 80 m, the EKF followed it, the
    drone turned with it, and the position frame ended up 15 deg away from the marker's - which
    is what sent the hover into a 17 m circle. This is logged first, to see how steady it is
    before anything is allowed to depend on it."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    # The marker's X axis in camera axes: +x is right, +y is toward the tail, so forward is -y.
    # Its bearing clockwise from the nose is atan2(right, forward); add the drone's yaw for the
    # bearing over the ground.
    rel = math.degrees(math.atan2(R[0, 0], -R[1, 0]))
    return (yaw_deg + rel + 180.0) % 360.0 - 180.0, rel


MARKER_HDG_MIN_PX = 40.0          # below this the corners are too coarse to average a heading from
MARKER_HDG_MAX_TILT_DEG = 8.0
MARKER_HDG_MIN_N = 50


def marker_heading_step(st, ground_hdg_deg, px, tilt_deg):
    """Running estimate of the marker's own heading over the ground, as a circular mean.

    The marker is laid out fresh for each day's flying, so its heading is not a constant anyone
    can write down: it has to be measured every flight. Only clean sightings count - a big marker
    and a level drone. Returns (heading, spread, n) once there are enough of them, else None."""
    if px < MARKER_HDG_MIN_PX or tilt_deg > MARKER_HDG_MAX_TILT_DEG or ground_hdg_deg != ground_hdg_deg:
        return None
    a = math.radians(ground_hdg_deg)
    st["s"] = st.get("s", 0.0) + math.sin(a)
    st["c"] = st.get("c", 0.0) + math.cos(a)
    st["n"] = st.get("n", 0) + 1
    if st["n"] < MARKER_HDG_MIN_N:
        return None
    r = math.hypot(st["s"], st["c"]) / st["n"]
    spread = math.degrees(math.sqrt(max(0.0, -2.0 * math.log(max(1e-12, r)))))   # circular sd
    return math.degrees(math.atan2(st["s"], st["c"])), spread, st["n"]


def centre_marker_pick(cands, min_px=None):
    """The largest centre-target marker (45/16/0) big enough to trust, or None."""
    min_px = MARKER_MIN_PX if min_px is None else min_px
    usable = [c for c in cands if c[0] in MARKER_POS_IDS and c[1] >= min_px]
    return usable[0] if usable else None


class ArucoWorker(threading.Thread):
    """
    ArUco detection off the flow loop. detectMarkers takes ~17 ms on the Orin Nano; run inline at
    100 FPS it cut the airborne loop to 58-82 FPS with frames 80-160 ms old (15 Sep flight).
    Only the newest submitted frame is kept; detections are rate-capped by min_interval_s.
    OpenCV releases the GIL inside detectMarkers, so the flow loop keeps running meanwhile.
    """
    def __init__(self, detector, min_interval_s=0.04):
        super().__init__(daemon=True)
        self.detector = detector
        self.min_interval_s = float(min_interval_s)
        self._cv = threading.Condition()
        self._job = None        # (image, frame timestamp s)
        self._result = None     # (seq, image, frame timestamp s, corners, ids)
        self._seq = 0
        self.running = True
        self.start()

    def run(self):
        last_start = 0.0
        while self.running:
            gap = self.min_interval_s - (time.monotonic() - last_start)
            if gap > 0:
                time.sleep(gap)
            with self._cv:
                if self._job is None:
                    self._cv.wait(timeout=0.2)
                job, self._job = self._job, None
            if job is None:
                continue
            last_start = time.monotonic()
            img, t_frame = job
            try:
                corners, ids, _ = self.detector.detectMarkers(img)
            except Exception:
                corners, ids = None, None
            with self._cv:
                self._seq += 1
                self._result = (self._seq, img, t_frame, corners, ids)

    def submit(self, img, t_frame):
        with self._cv:
            self._job = (img, t_frame)
            self._cv.notify()

    def result_after(self, seq):
        """Newest result with a sequence number above seq, else None."""
        with self._cv:
            r = self._result
        return r if r is not None and r[0] > seq else None

    def stop(self):
        self.running = False

# Only one daemon may talk to the camera and the autopilot. In flight 37 (19 Sep) two ran at once:
# one flying and sending the real position, the other gated on the ground and sending "position
# 0,0, not moving" - 7 495 vision messages of it. The autopilot fused both, its two EKF cores took
# turns believing each (24 lane switches in four minutes), and it ended in an EKF failsafe.
# An exclusive lock makes a second copy exit at once, whoever starts it. The kernel drops the
# lock when the holder dies, even on SIGKILL, so a crash can never leave it stuck.
LOCK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flight_logs", ".flight_daemon.lock")


def acquire_single_instance_lock(path=LOCK_PATH):
    """An open file holding an exclusive lock for this process's life, or None if another has it."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    return fd


def lock_holder(path=LOCK_PATH):
    """PID written by whoever holds the lock, for the message a second copy prints."""
    try:
        with open(path) as fh:
            return fh.read().strip() or "?"
    except OSError:
        return "?"


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

def build_oak_pipeline(max_exposure_us=3000, max_iso=1200, rgb_shadow=False, rgb_autofocus=False,
                       marker_cam=False, marker_cam_focus=MARKER_CAM_FOCUS):
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



    if marker_cam and not rgb_shadow:
        # Marker camera: the 12 MP RGB sensor (CAM_A), ISP-scaled to 2560x1440 (fx ~1943) and the
        # centre 1280x800 sent as grey at 10 fps. On the ground (19 Sep) the second mono could not
        # be used: a different resolution crashes the device and a different frame rate drags the
        # flow camera down with it (both monos are hardware-synced). The RGB sensor is separate;
        # the flow loop kept 95-97 fps with it at 10 fps (90-92 at 15 fps). ID 45 is ~74 px here
        # at 95 m vs 14 px on the 640x400 mono.
        rgb_m = pipeline.create(dai.node.ColorCamera)
        rgb_m.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        rgb_m.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        rgb_m.setIspScale(MARKER_CAM_ISP_NUM, MARKER_CAM_ISP_DEN)
        rgb_m.setFps(MARKER_CAM_FPS)
        rgb_m.setInterleaved(False)
        rgb_m.initialControl.setManualFocus(marker_cam_focus)
        xin_focus = pipeline.create(dai.node.XLinkIn)
        xin_focus.setStreamName("marker_ctrl")
        xin_focus.out.link(rgb_m.inputControl)
        rgb_m.initialControl.setAutoExposureEnable()
        rgb_m.initialControl.setAutoExposureLimit(max_exposure_us)
        iw, ih = rgb_m.getIspSize()
        fw, fh = MARKER_CAM_W / iw, MARKER_CAM_H / ih
        manip_m = pipeline.create(dai.node.ImageManip)
        manip_m.initialConfig.setCropRect(0.5 - fw / 2, 0.5 - fh / 2, 0.5 + fw / 2, 0.5 + fh / 2)
        manip_m.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
        manip_m.setMaxOutputFrameSize(MARKER_CAM_W * MARKER_CAM_H * 3)
        manip_m.inputImage.setBlocking(False)
        manip_m.inputImage.setQueueSize(1)
        rgb_m.isp.link(manip_m.inputImage)
        xout_hi = pipeline.create(dai.node.XLinkOut)
        xout_hi.setStreamName("mono_hi")
        manip_m.out.link(xout_hi.input)

    pipeline.setXLinkChunkSize(0)

    # 5. Onboard Bosch BNO086 9-Axis IMU (100 Hz)
    #
    # Every number here was 400/400/400 with a batch threshold of 5 until the
    # 3-5 Oct measurements, and all four were wrong:
    #
    #   * Rate. The three reports SHARE ONE PACKET, so over-asking is not
    #     neutral - it is harmful. Gyro at 400 dragged the whole stream down to
    #     57.8 Hz. A calibrated gyro requested at 200 against a documented
    #     100 Hz maximum delivered exactly 100 anyway. 100 is what the device
    #     actually gives, so 100 is what we ask for.
    #   * Batch threshold. At 5, 34.2% of camera frames arrived with NO IMU
    #     sample at all; at 1 it is 1.1%, for the same camera rate and the same
    #     gaps. Derotation needs a sample per frame, so this must stay 1.
    #   * GAME_ROTATION_VECTOR, not ROTATION_VECTOR. ROTATION_VECTOR fuses the
    #     magnetometer and shares the rotation pipe with it; taking it cost
    #     derotation outright and two flights. The GAME variant is gyro+accel
    #     only, which is all derotation needs. The magnetometer measured clean
    #     but stays OFF for this reason.
    #   * LINEAR_ACCELERATION, not ACCELEROMETER_RAW - gravity already removed,
    #     which is what the velocity path consumes.
    #
    # Camera yaw is NOT available from this: 4 Oct tested GAME_ROTATION_VECTOR's
    # yaw for exactly that and it cannot do it. Heading stays on the FC compass.
    IMU_HZ = 100
    imu = pipeline.create(dai.node.IMU)
    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_CALIBRATED, IMU_HZ)
    imu.enableIMUSensor(dai.IMUSensor.LINEAR_ACCELERATION, IMU_HZ)
    imu.enableIMUSensor(dai.IMUSensor.GAME_ROTATION_VECTOR, IMU_HZ)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(20)

    xout_imu = pipeline.create(dai.node.XLinkOut)
    xout_imu.setStreamName("imu")
    imu.out.link(xout_imu.input)

    if rgb_shadow:
        # Shadow only: 12MP sensor, centre 640x400 crop in native pixels (fx stays ~2900) so the
        # bandwidth stays small. Logged for comparison with the mono flow, never used for control.
        rgb = pipeline.create(dai.node.ColorCamera)
        rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        rgb.setFps(10); rgb.setInterleaved(False)   # 10 Hz: enough to compare, keeps the ISP off the mono path
        if rgb_autofocus:
            rgb.initialControl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)  # bench only
        else:
            rgb.initialControl.setManualFocus(0)              # infinity, as in flight
        rgb.initialControl.setAutoExposureEnable()
        rgb.initialControl.setAutoExposureLimit(max_exposure_us)
        w, h = rgb.getIspSize()
        fw, fh = 640.0 / w, 400.0 / h
        manip = pipeline.create(dai.node.ImageManip)
        manip.initialConfig.setCropRect(0.5 - fw / 2, 0.5 - fh / 2, 0.5 + fw / 2, 0.5 + fh / 2)
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
        manip.setMaxOutputFrameSize(640 * 400 * 3)
        rgb.isp.link(manip.inputImage)
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        manip.out.link(xout_rgb.input)

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
    parser.add_argument("--marker-size", type=float, default=0.242,
                        help="Black-square edge in metres for the ID given by --marker-size-id. "
                             "Every other marker uses the built-in table (MARKER_SIZES_M).")
    parser.add_argument("--marker-pos", action="store_true",
                        help="Marker map stage 2: let the centre marker (IDs 45/16/0) hold the position "
                             "sent to the EKF between 2 and 100 m.")
    parser.add_argument("--marker-cam", action="store_true",
                        help="Use the RGB camera (1280x800 centre crop, fx ~1943) for the centre marker's "
                             "position above 20 m. The flow camera is unchanged.")
    parser.add_argument("--marker-cam-focus", type=int, default=MARKER_CAM_FOCUS,
                        help=f"Lens position of the marker camera (0-255, default {MARKER_CAM_FOCUS}). "
                             "0 was out of focus on 20 Sep.")
    parser.add_argument("--marker-lag-comp", action="store_true",
                        help="Carry each marker fix forward by the flow movement since its frame. "
                             "Flight 47: doubles the fix-to-fix step at 35-50 m, so off by default.")
    parser.add_argument("--marker-cam-af-off", action="store_true",
                        help="Do not autofocus the marker camera once at 25 m; keep --marker-cam-focus.")
    parser.add_argument("--marker-vel-off", action="store_true",
                        help="With --marker-pos: above 35 m, while the marker holds the position, do not "
                             "send the flow velocity (it is off by 2 m/s up there).")
    parser.add_argument("--lb-velocity", action="store_true",
                        help="Send the long-baseline (87 ms) flow velocity below 30 m, fading to the "
                             "single-frame one by 40 m. Measured ~2x less noise low down, none above 40 m.")
    parser.add_argument("--marker-size-id", type=int, default=None,
                        help="Which marker ID --marker-size applies to. Without this the table is used unchanged.")
    parser.add_argument("--no-prec-loiter", dest="prec_loiter", action="store_false", default=True, help="Do not auto-enable ArduPilot Precision Loiter over ArUco #0 in LOITER")
    parser.add_argument("--save-images", action="store_true", default=True, help="Save camera and ArUco detection frames to disk (default: True)")
    parser.add_argument("--port", type=str, default="/dev/ttyTHS1", help="Primary serial/UART port for CUAV (default: /dev/ttyTHS1)")
    parser.add_argument("--no-save-images", dest="save_images", action="store_false", help="Disable saving images")
    parser.add_argument("--vel-delta", action="store_true", default=False, help="Send VISION_POSITION_DELTA (confidence scales the EKF velocity trust with height) instead of VISION_SPEED_ESTIMATE")
    parser.add_argument("--rgb-shadow", action="store_true", default=False, help="Also run LK flow on a centre crop of the RGB camera and log it (shadow only, never sent to the FC)")
    parser.add_argument("--rgb-af", action="store_true", default=False, help="Bench only: autofocus the RGB shadow camera (flight uses infinity focus)")
    parser.add_argument("--rc-drift-report", action="store_true", help="Enable RC input drift analysis reporting")
    parser.add_argument("--outdoor-gps", action="store_true", default=False, help="Enable automatic GPS switching outdoors if GPS fix is strong")
    args = parser.parse_args()

    instance_lock = acquire_single_instance_lock()
    if instance_lock is None:
        msg = (f"flight_daemon already running (PID {lock_holder()}); this copy is exiting so that "
               f"only one daemon talks to the camera and the autopilot.")
        print(msg, file=sys.stderr)
        sys.exit(0)

    setup_logging(test_mode=args.test)
    logging.info(f"[+] Single-instance lock held (PID {os.getpid()})")

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
    # Origin waits for SCR_USER1/2 (apply_cuav_parameters): the FC keeps the first origin it gets.
    cuav = CuavMavlinkClient(preferred_port=args.port, auto_origin=False)
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
    aruco_worker = ArucoWorker(aruco_detector)  # 25 Hz max, newest frame only
    # Marker camera frames get their own detector and thread (~21 ms each at 1280x800).
    # Subpixel corner refinement was tried on the marker camera on 20 Sep and made things worse
    # in the air: the step between consecutive fixes at 20-30 m doubled, 8 cm to 17 cm, and the
    # hover lost its edge from 28 m up. A rendered marker says refinement should help; a painted
    # one on grass, softened by the lens and sheared by the rolling shutter, evidently does not
    # give the refiner an edge to find. The plain corners are back.
    hi_worker = (ArucoWorker(aruco.ArucoDetector(aruco_dict, aruco_params),
                             min_interval_s=1.0 / MARKER_CAM_DETECT_HZ)
                 if args.marker_cam else None)
    camera_frame_clock = [None]                 # time of the newest frame, for the watchdog
    camera_watchdog = CameraWatchdog(cuav, camera_frame_clock)
    marker_size_m = float(args.marker_size)
    marker_sizes = dict(MARKER_SIZES_M)
    if args.marker_size_id is not None:
        marker_sizes[int(args.marker_size_id)] = marker_size_m
        logging.info(f"[+] Marker ID {int(args.marker_size_id)} size overridden to {marker_size_m:.3f} m")
    logging.info("[+] Marker table: " + ", ".join(f"{k}={v:.2f}m" for k, v in sorted(marker_sizes.items())))

    def _obj_pts(size_m):
        h = size_m / 2.0
        return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)

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
    if args.marker_cam and args.rgb_shadow:
        logging.warning("[!] --marker-cam and --rgb-shadow both need the RGB camera: marker camera OFF")
        args.marker_cam = False
    pipeline = build_oak_pipeline(args.max_exposure, args.max_iso, args.rgb_shadow, args.rgb_af,
                                  marker_cam=args.marker_cam, marker_cam_focus=args.marker_cam_focus)
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
        q_rgb = device.getOutputQueue(name="rgb", maxSize=2, blocking=False) if args.rgb_shadow else None
        fx_rgb = 2915.0
        if args.rgb_shadow:
            try:
                iw, ih = 3840, 2160
                K_rgb = np.array(device.readCalibration().getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, iw, ih))
                fx_rgb = float(K_rgb[0][0])
                logging.info(f"[+] RGB shadow flow ON: 4K centre crop 640x400, fx={fx_rgb:.0f} (mono fx={focal_length_px:.0f})")
            except Exception as e:
                logging.warning(f"[-] RGB shadow: calibration read failed ({e}), using fx={fx_rgb:.0f}")
        q_hi = q_hi_ctrl = None
        K_hi = dist_hi = R_hi = t_hi = None
        if args.marker_cam:
            try:
                _cal = device.readCalibration()
                _Ka = np.array(_cal.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A,
                                                        MARKER_CAM_ISP_W, MARKER_CAM_ISP_H))
                K_hi = crop_intrinsics(_Ka, (MARKER_CAM_ISP_W - MARKER_CAM_W) / 2.0,
                                       (MARKER_CAM_ISP_H - MARKER_CAM_H) / 2.0).astype(np.float32)
                dist_hi = np.array(_cal.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A),
                                   dtype=np.float32).reshape(-1, 1)
                _E = np.array(_cal.getCameraExtrinsics(dai.CameraBoardSocket.CAM_A,
                                                       dai.CameraBoardSocket.CAM_B))
                R_hi, t_hi = _E[:3, :3], _E[:3, 3] / 100.0          # EEPROM translation is in cm
                R_hi = MARKER_CAM_EXTRIN_FIX @ R_hi
                t_hi = MARKER_CAM_EXTRIN_FIX @ t_hi
                _ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R_hi) - 1.0) / 2.0))))
                q_hi = device.getOutputQueue(name="mono_hi", maxSize=2, blocking=False)
                q_hi_ctrl = device.getInputQueue(name="marker_ctrl")
                logging.info(f"[+] Marker camera ON: RGB {MARKER_CAM_W}x{MARKER_CAM_H} crop @ {MARKER_CAM_FPS} fps, "
                             f"fx={K_hi[0, 0]:.0f} cx={K_hi[0, 2]:.0f} cy={K_hi[1, 2]:.0f}; to CAM_B "
                             f"{np.linalg.norm(t_hi) * 100:.1f} cm, {_ang:.2f} deg; used above {MARKER_CAM_MIN_H_M:.0f} m")
            except Exception as e:
                logging.error(f"[!] Marker camera calibration failed ({e}): marker camera OFF")
                q_hi = q_hi_ctrl = None
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
        # Derotation scale. 1.2 came from log 24 with FC accelerometers as truth, but that log had
        # heavy vibration (VibeZ 40+) so the reference was unreliable. Fitting the per-frame residual
        # against the rotation shift on two later flights gives 0.95-1.06 (log 25) and 0.89/0.96
        # (log 26, clean 3 m hover, VibeZ 27): 1.2 over-corrects by 20-26 %. Back to 1.0.
        DEROT_GAIN = 1.0
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
                    gyro_hist.append((g.getTimestamp().total_seconds(), float(g.x), float(g.y), float(g.z)))
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
            """Integrate gyro x/y/z (rad) over [t0, t1] (host-synced s); None if samples do not cover it."""
            if t1 <= t0 or len(gyro_hist) < 2 or gyro_hist[0][0] > t0 or gyro_hist[-1][0] < t1:
                return None
            sx = sy = sz = 0.0
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
                        sz += 0.5 * ((prev[3] + (s[3] - prev[3]) * k0) + (prev[3] + (s[3] - prev[3]) * k1)) * (a1 - a0)
                prev = s
            return sx, sy, sz

        def shadow_tilt_delta(t0, t1):
            """(droll, dpitch, source) over [t0, t1] from the live flow's tilt source, else None."""
            if gyro_derot["ok"]:
                gd = gyro_tilt_delta(t0, t1)
                if gd is not None:
                    return gyro_derot["sign_r"] * gd[0], gyro_derot["sign_p"] * gd[1], "gyro"
            a0, ok0 = attitude_at(t0)
            a1, ok1 = attitude_at(t1)
            if a0 is None or a1 is None or not (ok0 and ok1):
                return None
            return a1[0] - a0[0], a1[1] - a0[1], "rotvec"

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
        marker_far_state = {}
        last_aruco_z = 0.0
        last_aruco_ang_x = 0.0
        last_aruco_ang_y = 0.0
        last_aruco_size_rad = 0.0
        aruco_hits = deque(maxlen=128)   # times of valid ArUco #0 PnP solutions
        ARUCO_CONFIRM_HITS = 3           # within 0.5 s before any LANDING_TARGET is sent
        PREC_LOITER_MIN_HITS = 8         # within 1.0 s before Precision Loiter is enabled
        # Change H: precision loiter oscillated above ~3.4 m (logs 20, 21, 23) while plain LOITER held
        # at 4.5 m. Enable only at or below 3.0 m; switch it off above 3.5 m.
        PREC_LOITER_MAX_ALT_M = 3.0
        PREC_LOITER_OFF_ALT_M = 3.5
        # 0.14 s (fitted from pad-estimate scatter, logs 20/21) made precision loiter oscillate
        # already at 2.5 m in log 22, vs calm at 2.9 m with 0.03 s in log 21. Keep 0.03 s.
        PLND_LAG_S = 0.03                # measured frame->wire 18.7 ms median / 28.8 p95, + serial/FC ~5 ms
        aruco_target_sent = 0
        t_mono_host = None
        ARUCO_LOST_S = 1.5               # no valid PnP detection for this long -> LOST
        ARUCO_NOTIFY_GAP_S = 1.0         # min gap between lock/lost RC messages
        aruco_notify = {"state": "lost", "sent": "lost", "since": 0.0, "held": 0.0, "last_tx": 0.0}

        def aruco_notice(now_t, hits, st, can_send):
            """Track the marker LOCK/LOST and return the state to announce ("locked"/"lost") or None.
            LOCK = >= ARUCO_CONFIRM_HITS valid detections within 0.5 s (same gate as LANDING_TARGET).
            LOST = no valid detection for ARUCO_LOST_S. Announcements are >= ARUCO_NOTIFY_GAP_S apart;
            a change during that gap (or while the FC link is down) is announced afterwards."""
            last_hit = hits[-1] if hits else None
            if st["state"] == "lost" and sum(1 for t in hits if now_t - t <= 0.5) >= ARUCO_CONFIRM_HITS:
                st["state"], st["since"] = "locked", now_t
                logging.info(f"[+] ArUco #{aruco_id} LOCK")
            elif st["state"] == "locked" and (last_hit is None or now_t - last_hit > ARUCO_LOST_S):
                st["state"], st["held"], st["since"] = "lost", now_t - st["since"], now_t
                logging.info(f"[-] ArUco #{aruco_id} LOST (held {st['held']:.1f} s)")
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
        # Appended, never truncated, so a daemon restart does not erase a flight's shadow data.
        shadow_log = None
        try:
            shadow_log = open(os.path.join(LOG_DIR, "flow_shadow.csv"), "a", buffering=65536)
            if shadow_log.tell() == 0:
                shadow_log.write("wall,t_frame,armed,airborne,vio_blind,h_opt,heading,vn_sent,ve_sent,vn_raw,ve_raw,"
                                 "vn_N,ve_N,inl_N,base_N,tilt_N,vn_J,ve_J,inl_J,span_J,ms_J\n")
        except Exception:
            shadow_log = None

        # Stage 1 of the marker map: the absolute position the marker gives, logged beside the
        # integrated flow position it would replace. Nothing here is sent to the autopilot.
        marker_log = None
        try:
            marker_log = open(os.path.join(LOG_DIR, "marker_shadow.csv"), "a", buffering=1)  # line-buffered: <=25 Hz, and a crash must not take the last seconds with it
            if marker_log.tell() == 0:
                marker_log.write("wall,t_frame,airborne,h,id,px,t_z,roll,pitch,yaw,"
                                 "flow_n,flow_e,mk_n,mk_e,mk_n_yawonly,mk_e_yawonly\n")
        except Exception:
            marker_log = None

        # Stage 2: the marker holds the position sent to the EKF (behind --marker-pos).
        mk2_state = {}
        mk2_fix = None               # (north, east, time) of the drone relative to the centre marker
        mk2_used_t = -1.0
        mk2_send_t = None
        mk2_drive_t = -1e9
        mk2_last_why = ""
        fv_state = {}                # --marker-vel-off hysteresis
        fv_sending = True
        _vel_mode_blocked = False
        mk2_why_log_t = 0.0
        # Marker camera state (--marker-cam).
        hi_seen_seq = 0
        hi_lens = -1
        hi_hdg_state = {}
        hi_hdg_log_t = 0.0
        hi_hdg_said = False
        flow_hist = deque(maxlen=400)     # (monotonic s, north, east) of the flow integrator
        hi_af_state = "off" if args.marker_cam_af_off else "wait"         # wait -> running -> locked
        hi_af_t = 0.0
        hi_hits = deque(maxlen=64)
        hi_fix_t = -1e9              # time of the last usable marker-camera fix
        hi_img_seq = 0
        hi_img_t = 0.0
        hi_log_t = 0.0
        mk2_src_last = None
        hi_log = None
        if args.marker_cam:
            try:
                hi_log = open(os.path.join(LOG_DIR, "marker_hi.csv"), "a", buffering=1)
                if hi_log.tell() == 0:
                    hi_log.write("wall,t_frame,age_ms,airborne,h,id,px,tx,ty,tz,roll_f,pitch_f,yaw_f,att_src,"
                                 "roll_now,pitch_now,mk_n,mk_e,down,used,lo_n,lo_e,lens,mrel,mground,sideratio,sqerr\n")
            except Exception:
                hi_log = None
        stage2_log = None
        if args.marker_pos:
            try:
                stage2_log = open(os.path.join(LOG_DIR, "marker_stage2.csv"), "a", buffering=1)
                if stage2_log.tell() == 0:
                    stage2_log.write("wall,h,flow_n,flow_e,sent_n,sent_e,how,marker_holds,perr\n")
            except Exception:
                stage2_log = None

        frame_count = 0
        vio_sent_count = 0
        aruco_detect_count = 0
        aruco_seen_seq = 0
        total_skipped_frames = 0   # older queued frames skipped to process the newest
        # Shadow flow (logged to flow_shadow.csv only): N long-baseline FeatureTracker, J Jetson LK.
        SHADOW_BASELINE_S = 0.08
        lb_tracks = LongBaselineTracks(cx_f, cy_f, baseline_s=SHADOW_BASELINE_S)
        jlk_worker = JetsonLKWorker(cx_f, cy_f)
        rgb_worker = JetsonLKWorker(320.0, 200.0, max_gap_s=0.30) if args.rgb_shadow else None
        rgb_win = deque(maxlen=64)
        rgb_last_logged_t = 0.0
        rgb_log = None
        if args.rgb_shadow:
            try:
                rgb_log = open(os.path.join(LOG_DIR, "flow_shadow_rgb.csv"), "a", buffering=65536)
                if rgb_log.tell() == 0:
                    rgb_log.write("wall,t_frame,armed,airborne,h_opt,heading,fx,vn_sent,ve_sent,vn_R,ve_R,inl_R,span_R,ms_R\n")
            except Exception:
                rgb_log = None
        jlk_win = deque(maxlen=64)
        lb_res = None
        aruco_locked = False
        aruco_cands = []
        aruco_seen_log_t = 0.0
        lb_vel = None                            # newest long-baseline velocity (vn, ve, t)
        lb_log_t = 0.0
        aruco_id = None
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
        _low_loiter_guard_t = 0.0   # change S: last ALT_HOLD request for LOITER without vision
        exp_hist = deque(maxlen=200)      # mono auto-exposure (us) and ISO, for the IR decision
        iso_hist = deque(maxlen=200)
        ir_flood_on = True                # matches the intensity set at startup
        ir_since = 0.0                    # when the opposite state was first wanted
        ir_last_change = 0.0
        # change V: body-frame deltas accumulated between sends for VISION_POSITION_DELTA
        delta_acc = {"fwd": 0.0, "right": 0.0, "r": 0.0, "p": 0.0, "y": 0.0, "dt": 0.0}
        tf03_hold_state = {"h": 0.0, "t": 0.0}   # change T: last good TF03 height in flight
        tf03_glitch_count = 0
        tf03_glitch_log_t = 0.0
        baro_agl_offset = None                   # lidar AGL - barometer height above home
        baro_off_prev_h = None                   # lidar height 0.5 s ago, to spot a climb
        baro_off_prev_t = 0.0
        baro_off_trust = False
        height_source_last = ""
        height_weight_last = None
        _vio_fresh = 0
        VIO_FRESH_FRAMES = 5
        # Loop runs at 100 FPS. Every 4th frame = 25 Hz VPE+VSE (~5 kB/s). At 50 Hz (~9.9 kB/s) the
        # 115200 baud link was near full and the FC forwarded it to the RC link (slow param download).
        VIO_SEND_EVERY_N_FRAMES = 4
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
                # The position gate is back at ArduPilot's own default of 500 (5 sigma). Flight 67
                # showed why 300 was too tight: with the flow velocity off above 35 m the EKF has
                # no velocity measurement, so its own estimate swings around the position we send
                # - the innovation bounced between 0.25 and 2.46 m for 75 s at 56 m with the
                # sticks centred, never trending. Then one swing crossed the gate at t=226.7 s,
                # and rejection is a trapdoor: free-running on the IMU makes the next innovation
                # bigger, so it can never come back. 6 -> 13 -> 45 m and an EKF failsafe. At 56 m
                # a 300 gate is about 3.4 m against a 2.46 m swing; 500 gives 5.6 m. The velocity
                # gate is left at 300 - nothing measured says to move it.
                gate_ok = (
                    cuav.set_parameter('EK3_VEL_I_GATE', 300.0)
                    and cuav.set_parameter('EK3_POS_I_GATE', 500.0)
                )
                if not gate_ok:
                    logging.warning("[!] EK3 VEL_I_GATE=300 / POS_I_GATE=500 not confirmed on FC")
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
                default_lat = float(getattr(args, 'home_lat', 30.895560) or 30.895560)
                default_lon = float(getattr(args, 'home_lon', 74.613858) or 74.613858)
                # Field site from the RC: SCR_USER1 = latitude, SCR_USER2 = longitude (decimal degrees).
                home_lat, home_lon, origin_src = pick_origin(
                    cuav.get_parameter('SCR_USER1'), cuav.get_parameter('SCR_USER2'),
                    default_lat, default_lon,
                )
                origin_ok = cuav.set_ekf_origin(lat=home_lat, lon=home_lon)
                # Bridge re-sends this same site every 2 s until the FC reports a global position.
                cuav.gps_lat, cuav.gps_lon = home_lat, home_lon
                cuav.auto_origin = True
                if origin_ok:
                    logging.info(
                        f"[+] EKF origin set {home_lat:.6f},{home_lon:.6f} from {origin_src} "
                        "(anchor only — not GPS)"
                    )
                    cuav.send_statustext(f"ORIGIN {origin_src} {home_lat:.5f},{home_lon:.5f}", severity=6)
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
            tf03_h, tf03_valid, tf03_held = tf03_glitch_hold(
                tf03_hold_state, tf03_h, tf03_valid, is_airborne, cuav.is_armed, now)
            if tf03_held:
                tf03_glitch_count += 1
                if now - tf03_glitch_log_t >= 2.0:
                    tf03_glitch_log_t = now
                    logging.warning(f"[!] TF03 dropout in flight: holding {tf03_h:.2f} m ({tf03_glitch_count} samples held)")
            # Subtract physical mounting clearance (measured 17 Sep: 12.5 cm)
            drone_agl_from_lidar = max(0.0, tf03_h - cuav.lidar_ground_clearance_m)
            was_airborne = is_airborne

            if tf03_valid:
                on_ground, _ = ground_state(drone_agl_from_lidar, cuav.is_armed, is_airborne)
            else:
                on_ground = (not cuav.is_armed) and (getattr(cuav, 'rel_alt_m', 0.0) <= 0.20)

            height_weight = 1.0      # share of the height taken from the lidar, for the log below
            if on_ground and not is_airborne:
                current_altitude_m = 0.00
                height_source = "GROUND_GATE"
                is_airborne = False
            elif tf03_valid and drone_agl_from_lidar > GROUND_AGL_M:
                baro_rel = max(0.0, getattr(cuav, 'rel_alt_m', 0.0))
                w = lidar_weight(drone_agl_from_lidar)
                if now - baro_off_prev_t >= 0.5:
                    baro_off_trust = baro_offset_trusted(
                        drone_agl_from_lidar, baro_off_prev_h, now - baro_off_prev_t,
                        w, is_airborne, cuav.is_armed)
                    baro_off_prev_h, baro_off_prev_t = drone_agl_from_lidar, now
                baro_agl_offset = track_baro_offset(
                    baro_agl_offset, drone_agl_from_lidar, baro_rel, trusted=baro_off_trust)
                baro_agl = None if baro_agl_offset is None else max(0.0, baro_rel + baro_agl_offset)
                if baro_agl is not None and not lidar_agrees_with_baro(drone_agl_from_lidar, baro_agl):
                    w = 0.0          # lidar spike: the barometer carries the height instead
                if w >= 1.0 or baro_agl is None:
                    height_m = drone_agl_from_lidar
                    height_source = f"TF03({getattr(cuav, 'tf03_msg_type', 'LIDAR')})"
                else:
                    height_m = w * drone_agl_from_lidar + (1.0 - w) * baro_agl
                    height_source = "BARO_AGL" if w <= 0.0 else f"TF03+BARO({w * 100:.0f}%)"
                current_altitude_m = float(np.clip(height_m, 0.0, MAX_FLOW_ALT_M))
                height_weight = w
                _, is_airborne = ground_state(drone_agl_from_lidar, cuav.is_armed, is_airborne)
            elif cuav.is_connected and cuav.is_armed and getattr(cuav, 'rel_alt_m', 0.0) > 0.18:
                baro_agl = max(0.0, cuav.rel_alt_m + (baro_agl_offset or 0.0))
                height_weight = 0.0
                current_altitude_m = float(np.clip(0.85 * current_altitude_m + 0.15 * baro_agl,
                                                   0.0, MAX_FLOW_ALT_M))
                height_source = "BARO_REL"
                is_airborne = True
            else:
                height_source = "HOLD"
                is_airborne = is_airborne and cuav.is_armed

            # The handover is invisible in the numbers, so say when it happens — but only once the
            # blend has really moved, or the 55 m boundary alone fills the log.
            _src_kind = height_source.split("(")[0]
            if is_airborne and height_source_changed(_src_kind, height_source_last,
                                                     height_weight, height_weight_last):
                logging.info(f"[+] Flow height source: {height_source} at {current_altitude_m:.1f} m"
                             f" (lidar {drone_agl_from_lidar:.1f} m, baro offset "
                             f"{'n/a' if baro_agl_offset is None else f'{baro_agl_offset:+.2f} m'})")
                height_weight_last = height_weight
            height_source_last = _src_kind

            if not is_airborne and hi_hdg_state:
                hi_hdg_state.clear()                 # each flight measures the marker afresh
                hi_hdg_said = False
            if not is_airborne and (mk2_state or mk2_fix is not None):
                mk2_state.clear()                    # landed: next flight anchors afresh
                mk2_fix = None
                mk2_src_last = None
                mk2_drive_t = -1e9

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
                # Change S: LOITER without vision near the ground lets the EKF velocity drift, and LOITER
                # then demands lean on the ground (log 23: ~20 deg demanded after mute, drone flipped).
                # ALT_HOLD demands zero lean with sticks centred.
                if (cuav.is_armed and getattr(cuav, 'flight_mode', '') == "LOITER"
                        and (on_ground or (tf03_valid and tf03_h < 0.40))
                        and now - _low_loiter_guard_t >= 2.0):
                    _low_loiter_guard_t = now
                    if cuav.set_flight_mode("ALT_HOLD"):
                        logging.warning(f"[!] LOITER with vision muted near ground (TF03 {tf03_h:.2f} m) -> ALT_HOLD")
                        cuav.send_statustext(">>> LOW ALT NO VIO | ALT_HOLD", severity=4)
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

            # Newest packet only: a backlog made frames 80-160 ms old. Tracker IDs persist across
            # skipped frames and dt comes from hardware timestamps, so skipping is safe for flow.
            ft_pkts = q_features.tryGetAll()
            in_ft = ft_pkts[-1] if ft_pkts else None
            ft_skipped = max(0, len(ft_pkts) - 1)
            if in_ft is not None:
                frame_count += 1
                total_skipped_frames += ft_skipped
                t_host_rx = time.monotonic()
                camera_frame_clock[0] = t_host_rx
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
                        dropped = seq_diff - 1 - ft_skipped
                        if dropped >= 2 or seq_diff > 20:
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
                mono_pkts = q_mono.tryGetAll()
                in_mono = mono_pkts[-1] if mono_pkts else None
                if in_mono is not None:
                    mono_f = in_mono.getCvFrame()
                    try:
                        exp_hist.append(in_mono.getExposureTime().total_seconds() * 1e6)
                        iso_hist.append(float(in_mono.getSensitivity()))
                    except Exception:
                        pass
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
                            logging.info(f"[+] Armed Flight Recording: Saving ~1 FPS images (max 3000) to {flight_record_dir}")
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

                # ~1 image/s: the 3000-image budget then covers a whole flight (1000 ran out at ~55 m on 19 Sep). At every 15th
                # frame it ran out after 3 minutes, so the high-altitude part was never recorded.
                if flight_record_dir and (frame_count % 80 == 0) and flight_img_seq < 3000:
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
                # Detection runs in ArucoWorker (<=25 Hz, newest frame); use each new result once,
                # with its own frame image and timestamp.
                corners, ids = None, None
                aruco_img, t_aruco_frame = None, None
                lb_res = None
                if q_rgb is not None:
                    rgb_pkts = q_rgb.tryGetAll()
                    if rgb_pkts:
                        rf = rgb_pkts[-1]
                        try:
                            rgb_worker.submit(rf.getCvFrame(), rf.getTimestamp().total_seconds())
                        except Exception:
                            pass
                if mono_f is not None:
                    aruco_worker.submit(mono_f, t_mono_host)
                    jlk_worker.submit(mono_f, t_mono_host)
                aruco_res = aruco_worker.result_after(aruco_seen_seq)
                aruco_pick = None
                if aruco_res is not None:
                    aruco_seen_seq, aruco_img, t_aruco_frame, corners, ids = aruco_res
                    aruco_cands = marker_candidates(ids, corners, marker_sizes)
                    aruco_pick = pick_marker(aruco_cands, sizes=marker_sizes)
                    if aruco_pick is not None:
                        aruco_detect_count += 1
                        last_aruco_lock_time = now
                    # Every known marker the camera can see, logged whatever we then use it for:
                    # this is how we learn how high each marker size really decodes.
                    if aruco_cands and (now - aruco_seen_log_t) >= 2.0:
                        aruco_seen_log_t = now
                        seen = " ".join(f"#{m}:{px:.0f}px" for m, px, _ in aruco_cands[:6])
                        used = f"#{aruco_pick[0]}" if aruco_pick is not None else "none (all below 11 px)"
                        logging.info(f"[+] Markers at {current_altitude_m:.1f} m: {seen} -> using {used}")

                # Robust Debounce Hold: Keep aruco_locked TRUE for 0.6s after detection
                # This completely eliminates the locked/unlocked flickering on RC!
                aruco_locked = (now - last_aruco_lock_time <= 0.6)
                if aruco_pick is not None:
                    aruco_id, aruco_side_px, m_corners = aruco_pick
                    aruco_obj_pts = _obj_pts(marker_sizes[aruco_id])

                    # Save annotated detection image when ArUco is locked (visual confirmation during flight)
                    if flight_record_dir and (aruco_detect_count % 5 == 1) and flight_img_seq < 3000:
                        flight_img_seq += 1
                        annotated = cv2.cvtColor(aruco_img, cv2.COLOR_GRAY2BGR)
                        cv2.polylines(annotated, [m_corners.astype(np.int32)], True, (0, 255, 0), 2)
                        cv2.putText(annotated, f"ArUco #{aruco_id} Locked ({aruco_detect_count})",
                                    (int(m_corners[0][0]), max(20, int(m_corners[0][1]) - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                        aruco_img_path = os.path.join(flight_record_dir, f"aruco_{flight_img_seq:05d}_ID{aruco_id}.jpg")
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
                        # Full attitude, not yaw alone: a swinging drone would otherwise see its
                        # own tilt as metres of movement (4.5 m at 26 m for 10 degrees of roll).
                        pad_offset_north, pad_offset_east = marker_offset_ned(
                            tvec, math.radians(roll_deg), math.radians(pitch_deg), yaw_rad)
                        last_aruco_lock_time = now
                        last_aruco_offset = (pad_offset_north, pad_offset_east)
                        if marker_log is not None and (cuav.is_armed or args.test):
                            try:
                                yn, ye = marker_offset_yaw_only(tvec, yaw_rad)
                                marker_log.write(
                                    f"{time.time():.4f},{(t_aruco_frame if t_aruco_frame is not None else math.nan):.4f},"
                                    f"{int(is_airborne)},{current_altitude_m:.3f},{aruco_id},{aruco_side_px:.1f},"
                                    f"{t_z:.3f},{roll_deg:.2f},{pitch_deg:.2f},{yaw_deg:.2f},"
                                    f"{pos_north:.4f},{pos_east:.4f},"
                                    f"{pad_offset_north:.4f},{pad_offset_east:.4f},{yn:.4f},{ye:.4f}\n")
                            except Exception:
                                pass
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
                        if args.marker_pos:
                            _conf = sum(1 for t in aruco_hits if now - t <= 0.5) >= ARUCO_CONFIRM_HITS
                            _down = marker_ned(tvec, math.radians(roll_deg), math.radians(pitch_deg),
                                               yaw_rad)[2]
                            _ok, _why = marker_fix_usable(aruco_id, current_altitude_m, _down, roll_deg,
                                                          pitch_deg, _conf, is_airborne, cuav.is_armed)
                            _hi_active = (now - hi_fix_t <= MARKER_CAM_FRESH_S
                                          and current_altitude_m >= MARKER_CAM_MIN_H_M)
                            if _ok and not _hi_active:
                                _sl = flow_shift_since(flow_hist, t_aruco_frame) if args.marker_lag_comp else None
                                mk2_fix = ((pad_offset_north + (_sl[0] if _sl else 0.0),
                                            pad_offset_east + (_sl[1] if _sl else 0.0), now, "lo"))
                            elif _why != mk2_last_why and (now - mk2_why_log_t) >= 3.0:
                                mk2_why_log_t = now
                                logging.info(f"[-] Marker position not used: {_why}")
                            mk2_last_why = _why
                        # Send only this fresh detection (never re-send stale angles), and only after the
                        # marker was seen repeatedly (rejects single-frame false positives).
                        if (cuav.is_connected and t_z > 0.08
                                and sum(1 for t in aruco_hits if now - t <= 0.5) >= ARUCO_CONFIRM_HITS):
                            if cuav.send_landing_target(
                                last_aruco_ang_x, last_aruco_ang_y, aruco_los_dist,
                                last_aruco_size_rad, last_aruco_size_rad,
                                time_usec=None if t_aruco_frame is None else int(t_aruco_frame * 1e6),
                            ):
                                aruco_target_sent += 1

                        # Use ArUco altitude ONLY as a fallback if TF03 LiDAR and Stereo Depth are unavailable
                        aruco_agl = max(0.0, t_z - 0.03)
                        if not tf03_valid and height_source in ["HOLD", "BARO_REL"]:
                            current_altitude_m = float(np.clip(0.80 * current_altitude_m + 0.20 * aruco_agl, 0.00, 15.00))
                            height_source = "ARUCO_Z"

                # 4b. Marker camera (RGB crop): the centre marker's position above MARKER_CAM_MIN_H_M.
                if q_hi is not None and q_hi_ctrl is not None and hi_af_state in ("wait", "running"):
                    # No focus pinned: let the camera focus itself once, high up where the marker is
                    # the only thing in view, then hold that lens position for the rest of the flight.
                    if hi_af_state == "wait" and current_altitude_m >= MARKER_CAM_AF_H_M and is_airborne:
                        try:
                            _c = dai.CameraControl()
                            _c.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
                            _c.setAutoFocusTrigger()
                            q_hi_ctrl.send(_c)
                            hi_af_state, hi_af_t = "running", now
                            logging.info(f"[+] Marker camera: autofocus triggered at {current_altitude_m:.1f} m")
                        except Exception as e:
                            logging.warning(f"[-] Marker camera autofocus failed ({e})")
                            hi_af_state = "off"
                    elif hi_af_state == "running" and now - hi_af_t >= MARKER_CAM_AF_SETTLE_S:
                        _found = int(hi_lens)
                        _keep = MARKER_CAM_AF_MIN <= _found <= MARKER_CAM_AF_MAX
                        _lens = _found if _keep else int(args.marker_cam_focus)
                        try:
                            _c = dai.CameraControl()
                            _c.setManualFocus(_lens)
                            q_hi_ctrl.send(_c)
                            logging.info(f"[+] Marker camera: autofocus found lens {_found}, "
                                         + (f"locked (use --marker-cam-focus {_found} next time)" if _keep
                                            else f"not believed - back to {_lens}"))
                            cuav.send_statustext(f">>> MARKER CAM FOCUS {_lens}", severity=6)
                        except Exception as e:
                            logging.warning(f"[-] Marker camera focus lock failed ({e})")
                        hi_af_state = "locked"
                if q_hi is not None:
                    try:
                        _hp = q_hi.tryGetAll()             # always drain, so the queue never backs up
                        if _hp and (current_altitude_m >= MARKER_CAM_PROC_H_M or args.test):
                            hi_lens = _hp[-1].getLensPosition()
                            hi_worker.submit(_hp[-1].getCvFrame(), _hp[-1].getTimestamp().total_seconds())
                    except Exception:
                        pass
                    _hr = hi_worker.result_after(hi_seen_seq)
                    if _hr is not None:
                        hi_seen_seq, _himg, _ht, _hc, _hids = _hr
                        _hpick = centre_marker_pick(marker_candidates(_hids, _hc, marker_sizes))
                        if _hpick is not None:
                            _hid, _hpx, _hcor = _hpick
                            _hsides = [float(np.linalg.norm(_hcor[i] - _hcor[(i + 1) % 4])) for i in range(4)]
                            # How far the quad is from a square, in degrees of corner angle. A
                            # painted marker that is not square biases both the pose and the
                            # heading, so it is measured on every fix rather than guessed at.
                            _hsq = 0.0
                            for _i in range(4):
                                _a = _hcor[(_i - 1) % 4] - _hcor[_i]
                                _b = _hcor[(_i + 1) % 4] - _hcor[_i]
                                _c = float(np.dot(_a, _b) / (np.linalg.norm(_a) * np.linalg.norm(_b) + 1e-9))
                                _hsq = max(_hsq, abs(math.degrees(math.acos(max(-1.0, min(1.0, _c)))) - 90.0))
                            _hok = False
                            if min(_hsides) >= 8.0 and max(_hsides) <= 2.5 * min(_hsides):
                                _hok, _hrv, _htv = cv2.solvePnP(_obj_pts(marker_sizes[_hid]),
                                                                _hcor.astype(np.float32), K_hi, dist_hi,
                                                                flags=cv2.SOLVEPNP_IPPE_SQUARE)
                            if _hok:
                                _tb = to_flow_camera_frame(_htv, R_hi, t_hi)
                                _mgh = _mrel = float("nan")
                                _att = attitude_at_time(getattr(cuav, 'att_hist', None), _ht)
                                _asrc = "frame"
                                if _att is None:
                                    _att, _asrc = (roll_deg, pitch_deg, yaw_deg), "now"
                                _mn, _me, _md = marker_ned(_tb, math.radians(_att[0]), math.radians(_att[1]),
                                                           math.radians(_att[2]))
                                _hn, _he = -_mn, -_me
                                try:
                                    _mgh, _mrel = marker_ground_heading(_hrv, _att[2])
                                except Exception:
                                    pass
                                _hdg = marker_heading_step(hi_hdg_state, _mgh, _hpx,
                                                           math.hypot(_att[0], _att[1]))
                                if _hdg is not None and (now - hi_hdg_log_t) >= 15.0:
                                    hi_hdg_log_t = now
                                    logging.info(f"[+] Marker heading {_hdg[0]:+.1f} deg over the ground "
                                                 f"(spread {_hdg[1]:.1f} deg, {_hdg[2]} sightings)")
                                    if not hi_hdg_said and _hdg[2] >= 150 and _hdg[1] < 3.0:
                                        hi_hdg_said = True
                                        cuav.send_statustext(f">>> MARKER HDG {_hdg[0]:+.0f} +-{_hdg[1]:.1f}", severity=6)
                                _sh = flow_shift_since(flow_hist, _ht) if args.marker_lag_comp else None
                                _lagc = _sh is not None
                                if _lagc:
                                    _hn, _he = _hn + _sh[0], _he + _sh[1]
                                hi_hits.append(now)
                                _hconf = sum(1 for t in hi_hits if now - t <= MARKER_CAM_CONFIRM_S) >= ARUCO_CONFIRM_HITS
                                _hu, _hwhy = marker_fix_usable(_hid, current_altitude_m, _md, _att[0], _att[1],
                                                               _hconf, is_airborne, cuav.is_armed)
                                _used = bool(args.marker_pos and _hu and current_altitude_m >= MARKER_CAM_MIN_H_M)
                                if _used:
                                    mk2_fix = (_hn, _he, now, "hi")
                                    hi_fix_t = now
                                if now - hi_log_t >= 5.0:
                                    hi_log_t = now
                                    logging.info(f"[+] Marker camera #{_hid}: {_hpx:.0f}px at {current_altitude_m:.1f} m "
                                                 f"(attitude {_asrc}, {'used' if _used else 'not used: ' + (_hwhy or 'below ' + str(int(MARKER_CAM_MIN_H_M)) + ' m')})")
                                if hi_log is not None and (cuav.is_armed or args.test):
                                    try:
                                        _lo_fresh = now - last_aruco_lock_time <= 0.2
                                        hi_log.write(
                                            f"{time.time():.4f},{_ht:.4f},{(time.monotonic() - _ht) * 1000:.0f},{int(is_airborne)},"
                                            f"{current_altitude_m:.3f},{_hid},{_hpx:.1f},{_tb[0]:.3f},{_tb[1]:.3f},{_tb[2]:.3f},"
                                            f"{_att[0]:.2f},{_att[1]:.2f},{_att[2]:.2f},{_asrc},{roll_deg:.2f},{pitch_deg:.2f},"
                                            f"{_hn:.4f},{_he:.4f},{_md:.3f},{int(_used)}{'' if _lagc else '*'},"
                                            f"{last_aruco_offset[0] if _lo_fresh else math.nan:.4f},"
                                            f"{last_aruco_offset[1] if _lo_fresh else math.nan:.4f},{hi_lens},"
                                            f"{_mrel:.2f},{_mgh:.2f},{max(_hsides) / max(1e-6, min(_hsides)):.4f},"
                                            f"{_hsq:.2f}\n")
                                    except Exception:
                                        pass
                                if flight_record_dir and now - hi_img_t >= 1.0 and hi_img_seq < 1500:
                                    hi_img_t = now
                                    hi_img_seq += 1
                                    _ann = cv2.cvtColor(_himg, cv2.COLOR_GRAY2BGR)
                                    # Draw the box a little outside the marker: drawn on the
                                    # border it spoils the very corners these images are kept to
                                    # check (20 Sep: the saved frames could not answer whether
                                    # the painted marker is square).
                                    _ctr = _hcor.mean(axis=0)
                                    _out = (_ctr + (_hcor - _ctr) * (1.0 + 10.0 / max(8.0, _hpx)))
                                    cv2.polylines(_ann, [_out.astype(np.int32)], True, (0, 255, 0), 1)
                                    cv2.putText(_ann, f"HI #{_hid} {_hpx:.0f}px h{current_altitude_m:.0f}m",
                                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                                    image_logger.log_image(os.path.join(flight_record_dir, f"hi_{hi_img_seq:05d}_ID{_hid}.jpg"),
                                                           _ann, is_bytes=False)

                # 5. OAK-D Myriad-X FeatureTracker Optical Flow (Zero Jetson CPU!)
                if in_ft is not None:
                    cur_features = {f.id: (f.position.x, f.position.y) for f in in_ft.trackedFeatures}
                    if t_frame_host is not None:
                        lb_res = lb_tracks.update(t_frame_host, cur_features)
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
                                    tilt_shift_x = -focal_length_px * math.tan(DEROT_GAIN * delta_roll)
                                    tilt_shift_y = focal_length_px * math.tan(DEROT_GAIN * delta_pitch)  # flight logs: ty ≈ +f·dPitch
                                    _tsx, _tsy = tilt_shift_x, tilt_shift_y
                                    trans_dx -= tilt_shift_x
                                    trans_dy -= tilt_shift_y
                            prev_roll_rad = der_roll_rad
                            prev_pitch_rad = der_pitch_rad
                            prev_der_ts = t_frame_host if imu_bracketed else None

                            pure_trans_dx = trans_dx if abs(trans_dx) >= 0.08 else 0.0
                            pure_trans_dy = trans_dy if abs(trans_dy) >= 0.08 else 0.0

                            cos_tilt = max(0.70, math.cos(oak_pitch_rad) * math.cos(oak_roll_rad))
                            cam_lever_m = cuav.camera_ground_clearance_m
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
                                # Sampled against the camera clock so a marker fix can be carried
                                # forward from its own frame to now.
                                flow_hist.append((time.monotonic(), pos_north, pos_east))
                                if args.vel_delta:
                                    delta_acc["fwd"] += dy_body
                                    delta_acc["right"] += dx_body
                                    delta_acc["r"] += _droll
                                    delta_acc["p"] += _dpitch
                                    delta_acc["y"] += (_gd[2] if _gd is not None and len(_gd) > 2 else 0.0)
                                    delta_acc["dt"] += dt
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

                # Shadow flow estimators (logged only, never sent): N = long-baseline FeatureTracker,
                # J = Jetson LK on the mono stream. Same tilt source, height and heading as the live flow.
                if shadow_log is not None and (cuav.is_armed or args.test):
                    try:
                        cos_tilt_sh = max(0.70, math.cos(math.radians(oak_pitch_deg)) * math.cos(math.radians(oak_roll_deg)))
                        h_sh = max(0.03, current_altitude_m * cos_tilt_sh
                                   + cuav.camera_ground_clearance_m)
                        hdg_sh = cuav.get_flow_heading_deg()
                        hdg_sh = yaw_deg if hdg_sh is None else hdg_sh
                        n_v = (math.nan, math.nan, 0, 0.0, "")
                        if lb_res is not None and t_frame_host is not None:
                            td = shadow_tilt_delta(lb_res[0], t_frame_host)
                            if td is not None:
                                vn_s, ve_s = px_to_ned_velocity(lb_res[1], lb_res[2], DEROT_GAIN * td[0],
                                                                DEROT_GAIN * td[1], focal_length_px,
                                                                h_sh, hdg_sh, t_frame_host - lb_res[0])
                                n_v = (vn_s, ve_s, lb_res[3], t_frame_host - lb_res[0], td[2])
                        for jt0, jt1, jtx, jty, jn, jms in jlk_worker.pop_results():
                            td = shadow_tilt_delta(jt0, jt1)
                            if td is not None and jt1 > jt0:
                                vn_s, ve_s = px_to_ned_velocity(jtx, jty, DEROT_GAIN * td[0], DEROT_GAIN * td[1],
                                                                focal_length_px, h_sh, hdg_sh, jt1 - jt0)
                                jlk_win.append((jt1, vn_s * (jt1 - jt0), ve_s * (jt1 - jt0), jt1 - jt0, jn, jms))
                        j_v = (math.nan, math.nan, 0, 0.0, 0.0)
                        if jlk_win:
                            w = [x for x in jlk_win if x[0] > jlk_win[-1][0] - SHADOW_BASELINE_S]
                            span = sum(x[3] for x in w)
                            if span > 0.02:
                                j_v = (sum(x[1] for x in w) / span, sum(x[2] for x in w) / span,
                                       int(np.median([x[4] for x in w])), span, float(np.median([x[5] for x in w])))
                                lb_vel = (j_v[0], j_v[1], now)
                        shadow_log.write(
                            f"{time.time():.4f},{(t_frame_host if t_frame_host is not None else math.nan):.4f},"
                            f"{int(cuav.is_armed)},{int(is_airborne)},{int(_vio_blind)},{h_sh:.3f},{hdg_sh:.2f},"
                            f"{filtered_vel_north:.4f},{filtered_vel_east:.4f},{raw_vel_north:.4f},{raw_vel_east:.4f},"
                            f"{n_v[0]:.4f},{n_v[1]:.4f},{n_v[2]},{n_v[3]:.4f},{n_v[4]},"
                            f"{j_v[0]:.4f},{j_v[1]:.4f},{j_v[2]},{j_v[3]:.4f},{j_v[4]:.2f}\n")
                    except Exception:
                        pass

                if rgb_log is not None and (cuav.is_armed or args.test):
                    try:
                        cos_tilt_r = max(0.70, math.cos(math.radians(oak_pitch_deg)) * math.cos(math.radians(oak_roll_deg)))
                        h_r = max(0.03, current_altitude_m * cos_tilt_r
                                  + cuav.camera_ground_clearance_m)
                        hdg_r = cuav.get_flow_heading_deg()
                        hdg_r = yaw_deg if hdg_r is None else hdg_r
                        for rt0, rt1, rtx, rty, rn, rms in rgb_worker.pop_results():
                            td = shadow_tilt_delta(rt0, rt1)
                            if td is not None and rt1 > rt0:
                                vn_r, ve_r = px_to_ned_velocity(rtx, rty, DEROT_GAIN * td[0], DEROT_GAIN * td[1],
                                                                fx_rgb, h_r, hdg_r, rt1 - rt0)
                                rgb_win.append((rt1, vn_r * (rt1 - rt0), ve_r * (rt1 - rt0), rt1 - rt0, rn, rms))
                        if rgb_win and rgb_win[-1][0] != rgb_last_logged_t:
                            rgb_last_logged_t = rgb_win[-1][0]
                            w = [x for x in rgb_win if x[0] > rgb_win[-1][0] - 0.30]
                            span = sum(x[3] for x in w)
                            if span > 0.02:
                                rgb_log.write(
                                    f"{time.time():.4f},{rgb_win[-1][0]:.4f},{int(cuav.is_armed)},{int(is_airborne)},"
                                    f"{h_r:.3f},{hdg_r:.2f},{fx_rgb:.1f},{filtered_vel_north:.4f},{filtered_vel_east:.4f},"
                                    f"{sum(x[1] for x in w)/span:.4f},{sum(x[2] for x in w)/span:.4f},"
                                    f"{int(np.median([x[4] for x in w]))},{span:.4f},{float(np.median([x[5] for x in w])):.2f}\n")
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
                if cuav.is_connected and (frame_count % VIO_SEND_EVERY_N_FRAMES == 0):
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
                            if args.lb_velocity:
                                fresh = lb_vel is not None and (now - lb_vel[2]) <= 0.25
                                vn, ve, lb_w = blend_velocity(
                                    vn, ve, lb_vel[0] if fresh else None, lb_vel[1] if fresh else None,
                                    long_baseline_weight(current_altitude_m))
                                if lb_w > 0.0 and (now - lb_log_t) >= 5.0:
                                    lb_log_t = now
                                    logging.info(f"[+] Long-baseline velocity {lb_w * 100:.0f}% "
                                                 f"at {current_altitude_m:.1f} m")
                            px, py = pos_north, pos_east
                            if args.marker_pos:
                                _fresh = mk2_fix is not None and mk2_fix[2] > mk2_used_t
                                _mk = (mk2_fix[0], mk2_fix[1]) if _fresh else None
                                if _fresh:
                                    mk2_used_t = mk2_fix[2]
                                _dt = 0.04 if mk2_send_t is None else max(0.0, min(0.2, now - mk2_send_t))
                                mk2_send_t = now
                                _switch = bool(_fresh and mk2_src_last is not None and mk2_fix[3] != mk2_src_last)
                                if _fresh:
                                    mk2_src_last = mk2_fix[3]
                                px, py, _how = marker_fix_step(mk2_state, pos_north, pos_east, _mk, _dt,
                                                               rate=marker_corr_rate_ms(current_altitude_m),
                                                               source=mk2_src_last)
                                if _switch:
                                    logging.info(f"[+] Marker position source -> {'marker camera' if mk2_src_last == 'hi' else 'flow camera'} "
                                                 f"at {current_altitude_m:.1f} m")
                                if _how in ("anchor", "source", "marker"):
                                    mk2_drive_t = now
                                if _how in ("anchor", "source", "reanchor"):
                                    logging.info(f"[+] Marker position {_how} at {current_altitude_m:.1f} m "
                                                 f"(correction {math.hypot(px - pos_north, py - pos_east):.2f} m)")
                                    if _how == "anchor":
                                        cuav.send_statustext(">>> MARKER POS ON", severity=6)
                                if (latched or ready_to_fuse) and now - mk2_drive_t <= MARKER_FRESH_S:
                                    # Only a trusted estimate is touched; an untrusted one is never
                                    # promoted. The figure is the marker's own error, honestly, even
                                    # where that is looser than the flow's fixed 1.22 m: above ~60 m
                                    # the marker is good to about 2 % of height and saying so lets the
                                    # EKF lean on its IMU and answer with a gentle correction instead
                                    # of chasing every centimetre of attitude noise.
                                    pos_cov = marker_pos_err_m(current_altitude_m) ** 2 / 3.0
                                if stage2_log is not None and cuav.is_armed:
                                    try:
                                        stage2_log.write(
                                            f"{time.time():.4f},{current_altitude_m:.3f},{pos_north:.4f},{pos_east:.4f},"
                                            f"{px:.4f},{py:.4f},{_how},{int(now - mk2_drive_t <= MARKER_FRESH_S)},"
                                            f"{math.sqrt(3.0 * pos_cov):.3f}\n")
                                    except Exception:
                                        pass
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
                        if suppress_vio or _vio_blind:
                            for _k in delta_acc:
                                delta_acc[_k] = 0.0
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
                            if args.vel_delta:
                                # One velocity path only: both would be fused twice.
                                if delta_acc["dt"] > 0.005:
                                    conf = flow_confidence(current_altitude_m)
                                    sent = cuav.send_vision_position_delta(
                                        delta_acc["dt"],
                                        (delta_acc["r"], delta_acc["p"], delta_acc["y"]),
                                        (delta_acc["fwd"], delta_acc["right"], 0.0),
                                        conf)
                                else:
                                    sent = False
                                for _k in delta_acc:
                                    delta_acc[_k] = 0.0
                            else:
                                _want_v = True
                                if args.marker_vel_off and args.marker_pos:
                                    _holds = (trust_vision and (latched or ready_to_fuse)
                                              and now - mk2_drive_t
                                              <= marker_hold_window_s(current_altitude_m))
                                    _gap = None
                                    _en = getattr(cuav, 'ekf_ne', None)
                                    if (_en is not None and now - getattr(cuav, 'ekf_ne_time', 0.0)
                                            <= FLOW_VEL_EKF_FRESH_S):
                                        _gap = (_en[0] - px, _en[1] - py)
                                    _sp = None
                                    if now - getattr(cuav, 'ekf_status_time', 0.0) <= FLOW_VEL_STATUS_FRESH_S:
                                        _sp = float(getattr(cuav, 'ekf_horiz_var', 0.0))
                                    _want_v = flow_velocity_wanted(fv_state, now, current_altitude_m,
                                                                   _holds, _gap, _sp)
                                    if _want_v != fv_sending:
                                        fv_sending = _want_v
                                        _msg = (f">>> FLOW VEL {'ON' if _want_v else 'OFF'} "
                                                f"{current_altitude_m:.0f}m")
                                        logging.info(f"[+] Flow velocity {'resumed' if _want_v else 'stopped'} at "
                                                     f"{current_altitude_m:.1f} m (marker holds={_holds}"
                                                     f"{', ' + fv_state.get('why', '') if _want_v else ''})")
                                        cuav.send_statustext(_msg, severity=6 if _want_v else 5)
                                _mode_block = flow_velocity_blocked_by_mode(
                                    getattr(cuav, 'flight_mode', ''), is_airborne)
                                if _mode_block != _vel_mode_blocked:
                                    _vel_mode_blocked = _mode_block
                                    logging.info(
                                        f"[+] Flow velocity {'held back' if _mode_block else 'allowed'}: "
                                        f"mode {getattr(cuav, 'flight_mode', '?')} at "
                                        f"{current_altitude_m:.1f} m"
                                        f"{' (hand flying - let the EKF variance settle so LOITER comes back)' if _mode_block else ''}")
                                if _want_v and not _mode_block:
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

            # Change IR: daylight/height aware floodlight. Only toggled after the condition holds 5 s
            # and at most every 10 s, so the image brightness never jumps mid-manoeuvre.
            if (now - ir_last_change) > 10.0:
                exp_med = float(np.median(exp_hist)) if exp_hist else 0.0
                iso_med = float(np.median(iso_hist)) if iso_hist else 0.0
                ir_height = 0.0 if on_ground else current_altitude_m
                want_ir = ir_flood_target(ir_flood_on, ir_height,
                                          height_valid=bool(tf03_valid or on_ground))
                if want_ir == ir_flood_on:
                    ir_since = 0.0
                else:
                    if ir_since == 0.0:
                        ir_since = now
                    elif now - ir_since >= 5.0:
                        try:
                            device.setIrFloodLightIntensity(0.7 if want_ir else 0.0)
                            ir_flood_on = want_ir
                            ir_last_change = now
                            ir_since = 0.0
                            logging.info(f"[+] IR floodlight {'ON' if want_ir else 'OFF'} at {ir_height:.1f} m "
                                         f"(exposure {exp_med:.0f} us, ISO {iso_med:.0f})")
                        except Exception as e:
                            ir_since = 0.0
                            logging.warning(f"[-] IR floodlight switch failed: {e}")

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
                cam_str = (f"Exp:{np.median(exp_hist):.0f}us ISO:{np.median(iso_hist) if iso_hist else 0:.0f} "
                           f"IR:{'ON' if ir_flood_on else 'OFF'}") if exp_hist else "Exp:n/a"
                aruco_badge = f"[ARUCO#{aruco_id} LOCKED dN:{last_aruco_offset[0]:+.2f}m dE:{last_aruco_offset[1]:+.2f}m]" if (now - last_aruco_lock_time <= 1.0) else "[ARUCO: SEARCHING]"
                vel_status = f"Vel: N:{filtered_vel_north:+.2f} E:{filtered_vel_east:+.2f}m/s"
                logging.info(
                    f"[VIO {fps:.1f} FPS | Drops:{total_dropped_frames} Skip:{total_skipped_frames}] Pos: N:{pos_north:+05.2f}m E:{pos_east:+05.2f}m Z:{current_altitude_m:04.2f}m (Yaw:{yaw_deg:+.1f}°) | "
                    f"{vel_status} | {aruco_badge} | {tf03_log} | {cam_str} | "
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
                    cuav.send_statustext(f">>> ARUCO LOCK #{aruco_id} | z{last_aruco_z:.1f}m off{off_m:.2f}m", severity=6)
                elif aruco_event == "lost":
                    cuav.send_statustext(f">>> ARUCO LOST #{aruco_id} | held {aruco_notify['held']:.0f}s", severity=4)

                # Precision Loiter over ArUco #0: only in LOITER, airborne, with a confirmed marker.
                # ArduPilot itself falls back to normal LOITER on stick input (>0.5 m/s^2) or 2 s target loss.
                marker_confirmed = sum(1 for t in aruco_hits if now - t <= 1.0) >= PREC_LOITER_MIN_HITS
                in_loiter = getattr(cuav, 'flight_mode', '') == "LOITER"
                pl_too_high = current_altitude_m > PREC_LOITER_OFF_ALT_M
                want_precloit = bool(args.prec_loiter and in_loiter and is_airborne and marker_confirmed
                                     and current_altitude_m <= PREC_LOITER_MAX_ALT_M)
                prec_on = getattr(cuav, '_prec_loiter_on', False)
                last_pl_cmd = getattr(cuav, '_prec_loiter_cmd_t', 0.0)
                marker_lost = (not aruco_hits) or (now - aruco_hits[-1] > 3.0)
                _far = marker_far_step(marker_far_state, last_aruco_offset[0], last_aruco_offset[1],
                                       now, is_airborne, now - last_aruco_lock_time <= 2.0)
                if _far is not None:
                    logging.info(f"[+] {_far} (h {current_altitude_m:.0f} m)")
                    cuav.send_statustext(_far, severity=4)
                if want_precloit and ((not prec_on) or now - last_pl_cmd >= 2.0):
                    if cuav.set_precision_loiter(True):
                        cuav._prec_loiter_cmd_t = now
                        if not prec_on:
                            cuav._prec_loiter_on = True
                            logging.info(f"[+] Precision Loiter ON (ArUco #{aruco_id} confirmed in LOITER)")
                            cuav.send_statustext(">>> PREC LOITER ON | PAD #0", severity=6)
                elif prec_on and ((not in_loiter) or (not is_airborne) or marker_lost or pl_too_high):
                    if cuav.set_precision_loiter(False):
                        cuav._prec_loiter_on = False
                        cuav._prec_loiter_cmd_t = now
                        if pl_too_high and in_loiter and is_airborne and not marker_lost:
                            logging.info(f"[*] Precision Loiter OFF (above {PREC_LOITER_OFF_ALT_M:.1f} m: {current_altitude_m:.2f} m)")
                            cuav.send_statustext(f">>> PREC LOITER OFF | >{PREC_LOITER_OFF_ALT_M:.1f}m", severity=4)
                        else:
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
                            rc_msg = f"PAD#{aruco_id} N:{last_aruco_offset[0]:+.2f} E:{last_aruco_offset[1]:+.2f} | {fps_val}FPS"
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

            if in_ft is None:
                time.sleep(0.001)  # idle only; a fixed sleep per frame cost 2.1 ms of the 10 ms budget

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
    camera_watchdog.stop()
    aruco_worker.stop()
    if hi_worker is not None:
        hi_worker.stop()
    aruco_worker.join(timeout=1.0)  # exiting while detectMarkers runs aborts the interpreter
    try:
        jlk_worker.stop()
        jlk_worker.join(timeout=1.0)
        shadow_log.close()
        if rgb_worker is not None:
            rgb_worker.stop(); rgb_worker.join(timeout=1.0)
        if rgb_log is not None:
            rgb_log.close()
    except Exception:
        pass
    try:
        if marker_log is not None:
            marker_log.close()
        if stage2_log is not None:
            stage2_log.close()
    except Exception:
        pass
    cuav.close()
    # CQ-5: Close vel_log_file in finally-equivalent block to flush buffered CSV data
    try:
        vel_log_file.flush()
        vel_log_file.close()
    except Exception:
        pass
    logging.info("[+] Navigation Daemon exited cleanly.")

def announce_crash(cuav):
    """Tell the pilot the daemon is dying. The camera watchdog lives in this same process and dies
    with it, so without this a crash in flight is silent until the EKF failsafe - nine seconds in
    flight 36. Returns True if the message went out."""
    try:
        if cuav is not None and cuav.is_connected and getattr(cuav, "is_armed", False):
            return bool(cuav.send_statustext(">>> VIO CRASHED - TAKE ALT_HOLD", severity=2))
    except Exception:
        pass
    return False


if __name__ == "__main__":
    try:
        main()
    except Exception:
        announce_crash(_cuav_ref)
        logging.exception("[!] flight_daemon crashed")
        raise
