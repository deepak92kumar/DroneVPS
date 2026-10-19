#!/usr/bin/env python3
"""The whole marker-free position pipeline as one runnable thing.

Until now the pieces existed but nothing joined them: FlowTracker, MapCache,
MapFixWorker and MapFuser were each proved on their own in a separate script.
This is the node that runs them together, and it runs identically against the
live camera or against a recorded flight, so the same code that flies can be
checked on the ground over data we already have.

    replay:  python3 live_node.py --replay flights/flight_XXXX
    live:    python3 live_node.py --live

Position comes out as latitude/longitude and as North/East metres from the
origin. Nothing is sent to the flight controller unless --send-fc is passed,
and that is refused unless the aircraft is disarmed, so the first transmission
can only ever happen on a bench.
"""
from __future__ import annotations
import argparse, csv, io, os, sys, time
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flow_odom import FlowTracker, shift_to_ne, attitude_shift_px
from map_cache import MapCache
from map_fix_worker import MapFixWorker
from map_fuse import MapFuser
from velocity_bridge import VelocityBridge
import single_instance

M_PER_DEG_LAT = 111_320.0
# The only map on this machine after the SSD was lost. It was
# /data/DRONE/ap/site_2km.tif, which no longer exists.
#
# BEWARE: this sheet does not cover the point the FC last flew from. It spans
# 30.88411-30.90332 N, 74.59762-74.63224 E, while SCR_USER1/2 in log
# 00000195.BIN reads 30.7844, 74.6072 - 11.1 km south of it. Going by the
# 13 Oct report ("today's ground - 12.7 km north of it"), this is the 13 Oct
# site and SCR_USER holds the newer southern one. A take-off point outside the
# sheet means no map fixes for the whole flight, which that report lists as its
# first item to confirm before flying. Pass --map explicitly per site.
DEFAULT_MAP = "/data/DRONE/bhh.tif"


# --------------------------------------------------------------------------- #
#  sources: the loop does not care which one it is given
# --------------------------------------------------------------------------- #
class ReplaySource:
    """Recorded flight: frames from disk, attitude and height from index.csv."""

    def __init__(self, folder: str, every: int = 3, min_alt: float = 0.0):
        self.folder = folder
        raw = open(os.path.join(folder, "index.csv"), "rb").read().replace(b"\x00", b"")
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf8", "replace"))))
        self.rows = [r for r in rows
                     if r.get("alt_rel_m") and r.get("yaw_deg") and r.get("roll_deg")
                     and float(r["alt_rel_m"]) >= min_alt][::every]
        self.i = 0
        self.focal = self._focal()

    def _focal(self):
        with open(os.path.join(self.folder, "camera.txt")) as f:
            for t in f.readline().split():
                if t.startswith("fx="):
                    return float(t[3:])
        raise RuntimeError("camera.txt mein fx= nahi mila")

    def read(self):
        while self.i < len(self.rows):
            r = self.rows[self.i]; self.i += 1
            img = cv2.imread(os.path.join(self.folder, "mono", r["mono_file"]),
                             cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            gps = None
            if r.get("lat") and r["lat"] not in ("", "0.0"):
                gps = (float(r["lon"]), float(r["lat"]))
            return dict(img=img, alt=float(r["alt_rel_m"]), yaw=float(r["yaw_deg"]),
                        roll=float(r["roll_deg"]), pitch=float(r["pitch_deg"]),
                        gps=gps, t=float(r["wall_s"]))
        return None


class LiveSource:
    """OAK-D mono plus read-only FC telemetry."""

    def __init__(self, fc_port="/dev/ttyTHS1", fc_baud=921600, *,
                 fps: int | None = None, stereo_depth: bool | None = None,
                 crop_left: int | None = None):
        import depthai as dai
        from record_flight import Telemetry
        from config import VpsConfig
        # NOT oak_rig. That module imports cuvslam, which is not installed and
        # was REJECTED on 28 Sep: the OAK-D's 7.5 cm baseline gives under 0.3 px
        # of disparity above ~25 m and the mission is 100-150 m, so it
        # contributed exactly 0 over 1680 frames. Importing it for one integer
        # made this constructor raise ModuleNotFoundError outright.
        cfg = VpsConfig()
        fps = cfg.mono_fps if fps is None else int(fps)
        stereo_depth = cfg.stereo_depth if stereo_depth is None else bool(stereo_depth)
        # 921600, not 115200. TELEM2 flies at 921600 - SERIAL2_BAUD reads 921
        # on the FC. At the wrong baud the link produces nothing, alt and yaw
        # stay None, and VpsNode.step returns None on every frame: the node
        # runs, finds its origin, and silently never estimates anything.
        self.tel = Telemetry(fc_port, fc_baud).start()

        p = dai.Pipeline()
        mono = {}
        # CAM_B alone unless depth is wanted; depth needs the pair.
        sockets = [("mono", dai.CameraBoardSocket.CAM_B)]
        if stereo_depth:
            sockets.append(("mono_right", dai.CameraBoardSocket.CAM_C))
        for name, sock in sockets:
            m = p.create(dai.node.MonoCamera)
            m.setBoardSocket(sock)
            m.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
            m.setFps(fps)
            m.initialControl.setAutoExposureLimit(8000)
            mono[name] = m
            if name == "mono":          # only the left frame leaves the device
                xo = p.create(dai.node.XLinkOut)
                xo.setStreamName(name)
                m.out.link(xo.input)

        # The camera's own gyro. 28 Sep moved derotation onto it because flow
        # error DOUBLES when the aircraft tilts - 0.81 m/step under 0.5 deg of
        # attitude change against 1.75 m over 1.5 deg - and the FC's attitude
        # arrives over MAVLink at a few Hz, interpolated and late. The gyro is
        # on the same device as the frames.
        imu = p.create(dai.node.IMU)
        # 100 Hz is not a choice here, it is the ceiling. Luxonis' own IMU
        # table gives GYROSCOPE_CALIBRATED a maximum of 100 Hz on the BNO086 -
        # the raw gyro goes to 1000 but is only recommended at or below 400,
        # because above that the part jitters. So this is already as fast as
        # the calibrated report can be asked for, and the BNO rounds a request
        # UP to its next supported rate, which is how asking for more than the
        # part supports turns into a slower packet rather than a faster one.
        imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_CALIBRATED, 100)
        # LINEAR_ACCELERATION, not ACCELEROMETER_RAW: the BNO086's own
        # fusion has already taken gravity out, and that is what the
        # velocity bridge's push_accel is specified to take. Raw accel
        # would hand it 9.8 m/s^2 of gravity to integrate the moment the
        # aircraft tilts, which is the opposite of a velocity estimate.
        # LINEAR_ACCELERATION's own maximum is 400 Hz, so there is headroom
        # here that the gyro does not have. It is deliberately matched to the
        # gyro instead, for two measured reasons: all three reports share one
        # packet, and 3 Oct measured asking for 400 on the gyro dragging the
        # whole stream down rather than speeding anything up; and the job this
        # does is filling a ~60 ms gap between flow frames, where six samples
        # already integrate to well under the sensor's own noise. Raising it
        # would cost packet rate to buy precision nothing downstream can use.
        imu.enableIMUSensor(dai.IMUSensor.LINEAR_ACCELERATION, 100)
        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(20)
        xg = p.create(dai.node.XLinkOut)
        xg.setStreamName("imu")
        imu.out.link(xg.input)

        if stereo_depth:
            # Depth is computed on the OAK's Myriad X VPU, so this costs the
            # Jetson nothing but the USB bytes - measured 19 Oct: no fps or IMU
            # change, +2 C. It feeds the VIO scale, which is what 5 Oct kept it
            # for (flow noise down 21% at 20-40 m and 53% at 60-80 m). It is
            # NOT a height source; see config.stereo_depth_as_height.
            # The PRESET is the whole cost. Measured 19 Oct on this camera,
            # mono pair at 1280x800/30 with nothing else running:
            #   HIGH_ACCURACY  30.7 fps, worst gap 44.3 ms   (deprecated)
            #   HIGH_DENSITY   30.7 fps, worst gap 51.8 ms
            #   ROBOTICS       16.3 fps, worst gap 72.2 ms
            #   DEFAULT        15.6 fps, worst gap 80.1 ms
            # DEFAULT halving the frame rate reproduces 9 Oct's "fps 25 ->
            # 12.5" almost exactly, so what was reverted that day may have been
            # the preset rather than stereo depth itself.
            # HIGH_DENSITY: holds 30 fps, is not deprecated, and fill matters
            # more than millimetre accuracy for a VIO scale aid.
            st = p.create(dai.node.StereoDepth)
            st.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            st.setLeftRightCheck(True)
            st.setSubpixel(False)
            st.setDepthAlign(dai.CameraBoardSocket.CAM_B)
            mono["mono"].out.link(st.left)
            mono["mono_right"].out.link(st.right)
            xd = p.create(dai.node.XLinkOut)
            xd.setStreamName("depth")
            st.depth.link(xd.input)

        self.dev = dai.Device()
        cal = self.dev.readCalibration()
        # The IMU is not bolted to the camera's axes, and the documented body
        # mapping (config.R_vio_to_ned) is written in the CAMERA frame - the
        # 27 Sep note says so explicitly: the gravity vector was "rotated into
        # the camera frame by the IMU->CAM_B extrinsics" before it landed on
        # cam_z at 99.8%. DepthAI hands out IMU-frame vectors, so the same
        # rotation has to be applied here or the mapping is being used on the
        # wrong frame. On this device the two frames may well be close, but
        # "may well be" is not a basis for integrating acceleration.
        self.R_imu2cam = None
        try:
            E = np.array(cal.getImuToCameraExtrinsics(dai.CameraBoardSocket.CAM_B))
            R = np.array(E)[:3, :3]
            # A rotation and nothing else: orthonormal, det +1. Anything that
            # fails this is a calibration we do not understand, and silently
            # multiplying by it would be worse than not rotating at all.
            if (np.allclose(R @ R.T, np.eye(3), atol=1e-3)
                    and abs(np.linalg.det(R) - 1.0) < 1e-3):
                self.R_imu2cam = R
                off = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
                print(f"IMU->CAM_B rotation read from calibration ({off:.1f} deg "
                      f"off the camera frame)")
            else:
                print("*** IMU->CAM_B extrinsics is not a rotation - IGNORED ***")
        except Exception as e:
            print(f"*** IMU->CAM_B extrinsics padha nahi gaya ({e}) - "
                  f"IMU axes ko camera axes maan raha hoon. Accel/gyro ka "
                  f"mapping is frame par UNVERIFIED hai. ***")

        K = np.array(cal.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, 1280, 800))
        self.focal = float(K[0][0])
        # The WHOLE of K, and the distortion, not just fx. Measured on this
        # device 20 Oct: cx 624.89 / cy 391.80, which is 15.11 px and 8.20 px
        # off the geometric centre, and an 8-coefficient rational distortion
        # with k1 24.09 / k2 -39.14 / k3 18.88. Both were being thrown away
        # here - fx was kept and the rest discarded - and both turn aircraft
        # rotation into phantom translation. See FlowTracker for the numbers.
        # The leg comes off here, before anything looks at the image, so flow,
        # the map matcher and the recorder all get the same clean frame. cx
        # shifts with the crop; fx, fy and cy do not. See config.crop_left_px.
        self.crop_left = int(crop_left if crop_left is not None
                             else getattr(cfg, "crop_left_px", 0))
        if self.crop_left > 0:
            K = K.copy()
            K[0][2] -= self.crop_left
            print(f"crop: left {self.crop_left} px (drone ka leg), "
                  f"cx {K[0][2] + self.crop_left:.1f} -> {K[0][2]:.1f}")
        self.K = K
        self.focal_y = float(K[1][1])
        try:
            self.D = np.array(list(cal.getDistortionCoefficients(
                dai.CameraBoardSocket.CAM_B)), float)
        except Exception as e:
            self.D = None
            print(f"*** distortion coefficients padhe nahi gaye ({e}) - "
                  f"flow raw pixels par chalega ***")
        print(f"K: fx {K[0][0]:.2f} fy {K[1][1]:.2f} cx {K[0][2]:.2f} cy {K[1][2]:.2f}"
              + (f", D[{len(self.D)}] k1 {self.D[0]:.2f}" if self.D is not None else ""))
        self.dev.startPipeline(p)
        self.q = self.dev.getOutputQueue("mono", 4, False)
        # Non-blocking and shallow on purpose: a depth frame that is late is
        # dropped rather than delaying the mono frame the flow loop runs on.
        self.q_depth = (self.dev.getOutputQueue("depth", 2, False)
                        if stereo_depth else None)
        self.q_imu = self.dev.getOutputQueue("imu", 50, False)
        # Gyro integrated between frames, in degrees. UNVERIFIED AXIS MAPPING:
        # which gyro axis is roll and which is pitch, and the sign of each, has
        # not been measured on this airframe. Both this and the FC's attitude
        # delta are logged every frame so the mapping can be settled from data
        # before anything depends on it. Derotation still uses the FC.
        self._gyro_roll = self._gyro_pitch = 0.0
        self._gyro_n = 0
        self._accel_n = 0
        self._t_accel = None
        self._t_gyro = None
        self.t0 = time.monotonic()

    def _to_cam(self, x, y, z):
        """IMU-frame vector -> camera frame, if the calibration told us how."""
        if self.R_imu2cam is None:
            return float(x), float(y), float(z)
        v = self.R_imu2cam @ np.array([x, y, z], float)
        return float(v[0]), float(v[1]), float(v[2])

    def _crop(self, img):
        return img if self.crop_left <= 0 else img[:, self.crop_left:]

    def read(self):
        fr = self.q.get()
        # tryGet, never get: the mono frame is what the flow loop is waiting on,
        # and depth is an aid to it. Missing depth means this frame has none,
        # not that the loop waits for it.
        # Integrate the gyro over the interval this frame covers.
        d_roll = d_pitch = 0.0
        accel = []
        pk = self.q_imu.tryGet()
        while pk is not None:
            for packet in pk.packets:
                # Flow alone cannot see what happens BETWEEN frames; the
                # accelerometer can, and that is the whole difference between
                # optical flow and visual-inertial odometry. Every sample is
                # carried out at its own device timestamp rather than one value
                # a frame, so the bridge integrates at the IMU's 100 Hz instead
                # of the flow loop's 17.
                #
                # The timestamp is the test of whether this report exists at
                # all. An IMU field that was never enabled is not absent from
                # the packet - it is present and full of zeros, with a
                # timestamp that never advances - so "is not None" would have
                # read a sensor that is not running as one reporting a perfect
                # standstill. It has to be caught here: the bridge cannot tell
                # a still aircraft from a dead accelerometer, and downstream
                # accel_n would have counted up either way.
                la = getattr(packet, "linearAcceleration", None)
                if la is not None:
                    ts = la.getTimestampDevice().total_seconds()
                    if self._t_accel is None or ts > self._t_accel:
                        if self._t_accel is not None:
                            ax, ay, az = self._to_cam(la.x, la.y, la.z)
                            accel.append((ts, ax, ay, az))
                            self._accel_n += 1
                        self._t_accel = ts
                g = getattr(packet, "gyroscope", None)
                if g is None:
                    continue
                ts = g.getTimestampDevice().total_seconds()
                if self._t_gyro is not None:
                    dt = ts - self._t_gyro
                    if 0.0 < dt < 0.1:
                        # Rotated into the camera frame first, for the same
                        # reason the acceleration is. x->roll, y->pitch after
                        # that is still a GUESS - but derotation runs off the
                        # FC attitude, so this pair is logged for comparison
                        # rather than flown on.
                        gx, gy, _gz = self._to_cam(g.x, g.y, g.z)
                        d_roll += np.degrees(gx) * dt
                        d_pitch += np.degrees(gy) * dt
                        self._gyro_n += 1
                self._t_gyro = ts
            pk = self.q_imu.tryGet()

        depth = None
        if self.q_depth is not None:
            d = self.q_depth.tryGet()
            if d is not None:
                depth = self._crop(d.getFrame())   # uint16 mm, aligned to CAM_B
        s = self.tel.snapshot()
        if s.get("alt_rel") is None or s.get("yaw") is None:
            return dict(img=self._crop(fr.getCvFrame()), alt=None, yaw=None, roll=None,
                        pitch=None, gps=None, depth=depth, rngfnd=s.get("rngfnd"),
                        gyro_d_roll=d_roll, gyro_d_pitch=d_pitch,
                        gyro_n=self._gyro_n,
                        accel=accel, accel_n=self._accel_n,
                        t=time.monotonic() - self.t0)
        gps = None
        if s.get("lat") not in (None, 0.0):
            gps = (s["lon"], s["lat"])
        return dict(img=self._crop(fr.getCvFrame()), alt=s["alt_rel"], yaw=s["yaw"],
                    alt_amsl=s.get("alt_amsl"), rngfnd=s.get("rngfnd"),
                    gps1_alt=s.get("gps1_alt"), gps1_fix=s.get("fix"),
                    roll=s["roll"], pitch=s["pitch"], gps=gps, depth=depth,
                    gyro_d_roll=d_roll, gyro_d_pitch=d_pitch,
                    gyro_n=self._gyro_n,
                    accel=accel, accel_n=self._accel_n,
                    t=time.monotonic() - self.t0, armed=bool(s.get("armed")))


class FcSender:
    """Transmit VISION_POSITION_ESTIMATE. The first thing here that writes to
    the aircraft, so it keeps its own guards rather than trusting the caller.

    Refuses while armed, full stop. The agreed order is bench, then props-off,
    then low hover, then altitude steps, and this is the bench.

    Sending is not the same as being believed: with VISO_TYPE=1 but the EK3
    sources left on GPS, ArduPilot parses and logs the messages and reports the
    vision sensor healthy, while the position solution stays exactly where it
    was. That is the whole point of the bench step - prove the link, change
    nothing about how the aircraft flies.
    """

    def __init__(self, port="/dev/ttyTHS1", baud=921600):
        from pymavlink import mavutil
        self.mavutil = mavutil
        self.m = mavutil.mavlink_connection(port, baud=baud, source_system=1,
                                            source_component=197)
        _announce(self.m)
        if self.m.wait_heartbeat(timeout=20) is None:
            raise RuntimeError("FC se heartbeat nahi aaya")
        # No armed check, at the pilot's instruction - same reason as
        # GpsInputSender: refusing to start while armed means the node can
        # never be brought back after an arm.
        self.t0 = time.monotonic()
        self.sent = 0

    def armed(self) -> bool:
        msg = self.m.recv_match(type="HEARTBEAT", blocking=False)
        if msg is None:
            return False
        return bool(msg.base_mode &
                    self.mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    def send(self, north_m, east_m, down_m, roll_deg, pitch_deg, yaw_deg):
        us = int((time.monotonic() - self.t0) * 1e6)
        self.m.mav.vision_position_estimate_send(
            us, float(north_m), float(east_m), float(down_m),
            float(np.radians(roll_deg)), float(np.radians(pitch_deg)),
            float(np.radians(yaw_deg)))
        self.sent += 1


class GpsInputSender:
    """Inject the estimate as a second GPS, the way the Stride unit does.

    ArduPilot's own notes mark GPS_INPUT "not recommended" and prefer ODOMETRY,
    and the reason is fair: on this path the flight controller applies no
    vision-specific health checks and treats what arrives with the same trust it
    gives a satellite fix. A wrong position is followed.

    It is still the right path here. Nothing about the aircraft's configuration
    has to change - no EK3 source switching, Loiter, Auto and RTL behave exactly
    as they do today, and missions stay in latitude and longitude. Fifty-four
    flights of experience are on the GPS path; this keeps them.

    This docstring used to claim that "with GPS_AUTO_SWITCH already set to use
    the best the aircraft moves onto this source by itself the moment the real
    GPS loses its fix". That is NOT TRUE on this aircraft: read live on 19 Oct,
    GPS_AUTO_SWITCH = 0, which is disabled - the primary receiver only. Nothing
    switches by itself. The channel 11 RC switch is what selects the source,
    which is the 28 Sep design on purpose: "never falls back to GPS1
    automatically - the pilot decides".

    What makes the trust defensible is that horiz_accuracy carries the fuser's
    own sigma. When the estimate is drifting the flight controller is told so,
    and GPS blending weighs it against the real receiver instead of guessing.
    """

    # GPS_INPUT_IGNORE_FLAGS
    IGN_VEL_HORIZ, IGN_VEL_VERT, IGN_SPEED_ACC = 8, 16, 32

    def __init__(self, port="/dev/ttyTHS1", baud=921600, gps_id: int = 1,
                 sats: int = 12, send_velocity: bool = True):
        from pymavlink import mavutil
        self.mavutil = mavutil
        self.m = mavutil.mavlink_connection(port, baud=baud, source_system=1,
                                            source_component=220)
        # Announce before waiting: through the router a client that has not
        # spoken is not registered, so its heartbeat can never arrive.
        _announce(self.m)
        if self.m.wait_heartbeat(timeout=20) is None:
            raise RuntimeError("FC se heartbeat nahi aaya")
        # No armed check here either, at the pilot's instruction. Refusing to
        # START while armed would mean the node could never be brought back up
        # after an arm - which is precisely when the aircraft needs it.
        self.gps_id = int(gps_id)
        self.sats = int(sats)
        # OFF by default, and it stays off until someone passes it. Velocity is
        # the channel that caused four EKF failsafes on 28 Sep by reporting
        # 47.7 m/s during a 106 m hover, and it is the 54% row that the 13 Oct
        # report still calls the largest open fault.
        self.send_velocity = bool(send_velocity)
        self.sent = 0
        self.sent_with_vel = 0

    def armed(self) -> bool:
        msg = self.m.recv_match(type="HEARTBEAT", blocking=False)
        return False if msg is None else bool(
            msg.base_mode & self.mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    def velocity_fields(self, vel):
        """Decide what to tell the FC about velocity, and say so explicitly.

        Returns (ignore_flags, vn, ve, speed_acc, why).

        IGN_VEL_VERT is ALWAYS set. The bridge is horizontal only, and the FC
        itself agrees there is no vertical source: EK3_SRC1_VELZ reads 0 in log
        00000195.BIN. Height comes from the lidar below 60 m and the barometer
        above it.

        Horizontal velocity is sent only when the bridge says valid AND the
        value is above the deadband. When the deadband bites we go back to
        IGNORE rather than sending zero, and that is a deliberate difference
        from asserting a stationary aircraft: 5 Oct measured the deadband
        discarding 49% of genuine motion below 25 m, so a zero there would be a
        confident false statement to the EKF, which follows what it is given on
        this path. "I do not know" is the honest flag, and "stationary" is not.
        """
        ign = self.IGN_VEL_VERT
        if not self.send_velocity:
            return ign | self.IGN_VEL_HORIZ | self.IGN_SPEED_ACC, 0.0, 0.0, 0.0, "off"
        if vel is None:
            return ign | self.IGN_VEL_HORIZ | self.IGN_SPEED_ACC, 0.0, 0.0, 0.0, "no estimate"
        if not vel.valid:
            return (ign | self.IGN_VEL_HORIZ | self.IGN_SPEED_ACC, 0.0, 0.0, 0.0,
                    vel.reason)
        if vel.deadbanded:
            return (ign | self.IGN_VEL_HORIZ | self.IGN_SPEED_ACC, 0.0, 0.0, 0.0,
                    f"deadbanded ({vel.speed_mps:.2f} < {vel.deadband_mps:.2f})")
        # Speed accuracy the EKF can weigh. Floored at the measured noise -
        # flow velocity error was 0.236 m/s at 0-30 m and phantom speed at a
        # hover ran 0.36-0.43 m/s - and widened with how stale the flow is,
        # because between flow updates this is dead reckoning on the camera
        # accelerometer. UNVERIFIED scaling; it wants a sweep against a flight.
        acc = max(0.4, 0.4 + 2.0 * vel.age_s)
        return ign, float(vel.sent_vn), float(vel.sent_ve), float(acc), "sent"

    def send(self, lon, lat, alt_amsl_m, sigma_m, yaw_deg=None, vel=None):
        t = time.time()
        week = int((t - 315964800) // (7 * 86400))
        week_ms = int(((t - 315964800) % (7 * 86400)) * 1000)
        # HDOP is what a GCS shows the pilot, so it should read like the real
        # thing: metres of error over the usual ~1.5 m per HDOP unit.
        hdop = float(np.clip(sigma_m / 1.5, 0.5, 99.0))
        ign, vn, ve, speed_acc, self.vel_why = self.velocity_fields(vel)
        self.m.mav.gps_input_send(
            int(t * 1e6), self.gps_id, ign,
            week_ms, week, 3,                      # fix_type 3 = 3D
            int(lat * 1e7), int(lon * 1e7), float(alt_amsl_m),
            hdop, hdop, vn, ve, 0.0,
            speed_acc, float(sigma_m), float(sigma_m * 2.0),
            self.sats)
        self.sent += 1
        if not (ign & self.IGN_VEL_HORIZ):
            self.sent_with_vel += 1


# --------------------------------------------------------------------------- #
class AmslRef:
    """The ground's AMSL altitude, for the one field GPS_INPUT cannot fake.

    The first attempt at this read GLOBAL_POSITION_INT.alt and sent it straight
    back as the GPS2 altitude. That is a FEEDBACK LOOP: GLOBAL_POSITION_INT is
    the EKF's altitude, the EKF's origin altitude came from our own injection,
    so the number we sent was the number we had just been told. Measured on the
    bench 20 Oct: the real receiver read 300.73 m and GPS2 read 2.43 m and
    climbing, having started at 0.01 m.

    So the reference comes from somewhere that is not the EKF:

      1. --site-alt, if the pilot knows the elevation and says so.
      2. the REAL receiver, the first time it has a 3D fix: its AMSL minus the
         relative altitude at that moment is the ground, and that is a constant.

    and the altitude sent is that reference plus the relative altitude, which
    is barometer-driven (EK3_SRC1_POSZ is the baro, not GPS) and so is not
    something we fed in.

    With neither available there is no honest AMSL, and it says so rather than
    inventing one.
    """

    def __init__(self, fixed_m=None):
        self.ref = None if fixed_m is None else float(fixed_m)
        self.source = None if fixed_m is None else "--site-alt"

    def observe(self, gps1_fix, gps1_alt_m, alt_rel_m) -> None:
        if self.ref is not None:
            return
        if (gps1_fix is not None and gps1_fix >= 3 and gps1_alt_m is not None
                and alt_rel_m is not None):
            self.ref = float(gps1_alt_m) - float(alt_rel_m)
            self.source = f"asli GPS1 3D fix ({self.ref:.1f} m)"

    def amsl(self, alt_rel_m):
        if self.ref is None or alt_rel_m is None:
            return None
        return self.ref + float(alt_rel_m)


# --------------------------------------------------------------------------- #
class SourceWatcher:
    """Watches GPS_PRIMARY while the node runs, not just once at startup.

    This is the hole behind two of the pilot's complaints. read_gps_primary was
    called ONCE, before the loop, and the node then injected GPS_INPUT forever -
    so flipping the transmitter to the real receiver changed which GPS the
    flight controller NAVIGATED on, but the VPS carried on pushing into GPS2
    regardless, and nothing was announced either way. Log 214 has 35 VPS
    heartbeats and not one source line.

    Two ways in, because one is not reliable on its own:
      * ArduPilot broadcasts PARAM_VALUE when a parameter changes, so a switch
        made by gps_rc_switch.py (or by the GCS, or by hand) shows up here
        within a frame or two.
      * and it is re-requested every period_s anyway, because a broadcast that
        arrives while we are not reading is a broadcast that is gone. Half a
        second, not two: the pilot has to be able to flip the switch and see
        the answer, and every link in that chain adds its own delay already -
        four stable RC samples in gps_rc_switch, then its parameter set and
        read-back, then this.

    Unknown is NOT treated as "on the receiver". If the parameter cannot be
    read the node keeps injecting, because the two failures are not
    symmetrical: injecting into GPS2 while the aircraft navigates on GPS1 is
    ignored by the flight controller - GPS_AUTO_SWITCH is 0, so the primary is
    the primary - whereas going silent while the aircraft is actually ON the
    VPS is what left it with no position at all on 20 Oct. So it fails towards
    sending, and says out loud that it does not know.
    """

    def __init__(self, link, period_s: float = 0.5):
        self.link = link
        self.period = float(period_s)
        self._next = 0.0
        self.value = None
        self.reads = 0

    def poll(self, t: float):
        """Returns GPS_PRIMARY as an int, or None if it has never been read."""
        if self.link is None:
            return self.value
        try:
            while True:
                msg = self.link.recv_match(type="PARAM_VALUE", blocking=False)
                if msg is None:
                    break
                if msg.param_id == "GPS_PRIMARY":
                    self.value = int(msg.param_value)
                    self.reads += 1
            if t >= self._next:
                self._next = t + self.period
                self.link.mav.param_request_read_send(
                    self.link.target_system, self.link.target_component,
                    b"GPS_PRIMARY", -1)
        except Exception:
            pass
        return self.value

    def on_vps(self) -> bool:
        """True when the VPS may inject. Unknown counts as yes - see above."""
        return self.value != 0


# --------------------------------------------------------------------------- #
def accel_to_ne(ax: float, ay: float, yaw_deg: float):
    """Camera-frame horizontal acceleration -> (north, east), m/s^2.

    The body mapping is config.py's, measured 27 Sep on this camera: cam_z is
    DOWN - the gravity vector lands on it at 99.8% - the nose points at the TOP
    of the image, and cam_y is image DOWN. So body forward is -cam_y and body
    right is +cam_x.

    UNVERIFIED on THIS airframe: the camera has been remounted since that
    measurement. A recording with a deliberate push along one axis would settle
    it, and until then the only protection is that flow stays the authority -
    a wrong mapping degrades what is carried between frames, not the answer.

    The yaw convention matches shift_to_ne, which is the point: the same
    rotation has to take flow pixels and acceleration into the same frame, or
    position and velocity will disagree about which way north is.
    """
    import numpy as _np
    fwd, right = -ay, ax
    t = _np.radians(yaw_deg)
    c, sn = _np.cos(t), _np.sin(t)
    return float(fwd * c - right * sn), float(fwd * sn + right * c)


def choose_origin(cli_origin, scr_origin, gps_primary, gps_fix):
    """Where the origin may come from. Returns ((lon, lat), why) or (None, why).

    Precedence, and the middle rule is the pilot's:

      1. --origin             an explicit instruction beats everything
      2. SCR_USER1/2          the point typed on the RC
      3. the live receiver    ONLY when GPS_PRIMARY == 0

    While the aircraft is on the VPS, the receiver is not an acceptable origin
    whatever it is reporting. A receiver that happens to have a fix sitting on
    the bench before arming must not quietly become the origin, because every
    position afterwards is measured from it.

    GPS_PRIMARY of None - we could not ask - is treated as "on the VPS", not as
    "on the receiver". Not knowing must never be the state that permits a
    GPS-derived origin.
    """
    if cli_origin is not None:
        return cli_origin, "--origin"
    if scr_origin is not None:
        return scr_origin, "SCR_USER1/2 (typed on the RC)"
    if gps_primary == 0:
        if gps_fix is not None:
            return gps_fix, "the live receiver (GPS_PRIMARY=0)"
        return None, "waiting for a receiver fix"
    if gps_primary == 1:
        return None, "the aircraft is on the VPS"
    return None, "GPS_PRIMARY could not be read"


def _announce(link) -> None:
    """Say hello so a UDP router knows where to send our replies.

    On a serial device this is harmless. Through the router it is REQUIRED:
    the router learns a client's address from the first packet it receives, so
    a client that only listens is never registered and waits for a heartbeat
    that can never arrive. That is inherent to UDP, not a router bug - but a
    helper that only reads still has to announce itself.
    """
    try:
        link.mav.heartbeat_send(6, 8, 0, 0, 0)   # MAV_TYPE_GCS, INVALID
    except Exception:
        pass


def read_gps_primary(endpoint: str, baud: int = 921600, timeout_s: float = 8.0):
    """Which source the aircraft is flying on: 0 = real GPS, 1 = the VPS.

    This is the value gps_rc_switch writes from channel 6, so it is the
    aircraft's own answer rather than our guess. None means we could not ask -
    and "could not ask" must never be read as "on the real GPS", because that
    is the direction that permits a GPS-derived origin.
    """
    try:
        from pymavlink import mavutil
    except Exception:
        return None
    m = None
    try:
        m = mavutil.mavlink_connection(endpoint, baud=baud, source_system=255,
                                       source_component=223)
        _announce(m)
        if m.wait_heartbeat(timeout=timeout_s) is None:
            return None
        m.mav.param_request_read_send(m.target_system, m.target_component,
                                      b"GPS_PRIMARY", -1)
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            msg = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=1.0)
            if msg is not None and msg.param_id == "GPS_PRIMARY":
                return int(msg.param_value)
        return None
    except Exception:
        return None
    finally:
        if m is not None:
            try:
                m.close()
            except Exception:
                pass


def read_scr_user(endpoint: str, baud: int = 921600, timeout_s: float = 8.0):
    """The take-off point the pilot typed on the RC, read off the FC.

    SCR_USER1 is latitude, SCR_USER2 is longitude, in decimal degrees. The
    13 Oct report records why this has to win over the live receiver: the pilot
    typed coordinates for one site, drove to a second, retyped nothing, and the
    aircraft picked up the NEW location by itself - because the receiver was
    winning whenever it had a fix, so the typed point only ever reached the
    flight controller's home. On the VPS the typed point must win.

    Returns (lon, lat) or None. The validation mirrors flight_daemon's
    pick_origin: 0, missing or out-of-range falls through rather than placing
    the origin in the Gulf of Guinea.

    Called BEFORE the camera source opens the serial port, so this works on a
    direct device as well as on the router's UDP endpoint.
    """
    try:
        from pymavlink import mavutil
    except Exception:
        return None
    m = None
    try:
        m = mavutil.mavlink_connection(endpoint, baud=baud, source_system=255,
                                       source_component=194)
        _announce(m)
        if m.wait_heartbeat(timeout=timeout_s) is None:
            return None
        want = {"SCR_USER1": None, "SCR_USER2": None}
        for name in want:
            m.mav.param_request_read_send(m.target_system, m.target_component,
                                          name.encode(), -1)
        t0 = time.time()
        while time.time() - t0 < timeout_s and any(v is None for v in want.values()):
            msg = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=1.0)
            if msg is not None and msg.param_id in want:
                want[msg.param_id] = float(msg.param_value)
        lat, lon = want["SCR_USER1"], want["SCR_USER2"]
        if lat is None or lon is None:
            return None
        if not (np.isfinite(lat) and np.isfinite(lon)):
            return None
        if abs(lat) < 1e-6 or abs(lon) < 1e-6:
            return None
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return None
        return float(lon), float(lat)
    except Exception:
        return None
    finally:
        if m is not None:
            try:
                m.close()
            except Exception:
                pass



class Notifier:
    """Short STATUSTEXT lines to whatever is listening on the radio.

    The pilot flies looking at a transmitter screen, not a terminal. The marker
    daemon has thirty-one of these; the VPS path had none, so from the air
    there was no way to tell a healthy VPS from a dead one.

    Two rules, because a flooded link is worse than a quiet one:

      * a heartbeat line every period_s, so silence means something is wrong
        rather than nothing is happening
      * state CHANGES immediately, and only once per change - tracking lost,
        map fixes stopped, velocity refused, the camera IMU gone quiet

    STATUSTEXT is 50 characters. Anything longer is cut by the protocol, so
    these are written short rather than truncated by accident.
    """

    SEV_CRIT, SEV_WARN, SEV_INFO = 2, 4, 6

    def __init__(self, link, period_s: float = 10.0, enabled: bool = True):
        self.link = link                    # anything with .mav.statustext_send
        self.period = float(period_s)
        self.enabled = bool(enabled) and link is not None
        self._next = 0.0
        self._state = {}
        self.sent = 0

    def _send(self, text: str, sev: int) -> None:
        if not self.enabled:
            return
        try:
            self.link.mav.statustext_send(sev, text[:50].encode())
            self.sent += 1
        except Exception:
            pass                            # the link is not the mission

    def event(self, key: str, value, text: str, sev: int = SEV_WARN) -> None:
        """Announce only when `value` for `key` actually changes."""
        if self._state.get(key) == value:
            return
        self._state[key] = value
        self._send(text, sev)

    def tick(self, t: float, est: dict, node) -> None:
        v = est["vel"]
        tracks = est["tracks"]

        # Changes first - these are what the pilot needs immediately.
        self.event("track", tracks >= 30,
                   f"VPS TRACK OK {tracks}" if tracks >= 30
                   else f"VPS TRACK LOST {tracks}",
                   self.SEV_INFO if tracks >= 30 else self.SEV_CRIT)
        self.event("vel", v.valid,
                   "VPS VEL OK" if v.valid else f"VPS VEL {v.reason}"[:50],
                   self.SEV_INFO if v.valid else self.SEV_WARN)
        self.event("imu", (est.get("gyro_n") or 0) > 0,
                   "VPS GYRO OK" if (est.get("gyro_n") or 0) > 0
                   else "VPS GYRO SILENT - CAM IMU DEAD",
                   self.SEV_INFO if (est.get("gyro_n") or 0) > 0 else self.SEV_CRIT)
        # The accelerometer gets its own line. Until 20 Oct nothing read it at
        # all, and a dead one is invisible in the gyro's: flow would carry on
        # looking healthy while the gap between frames went unfilled.
        a_n = est.get("accel_n") or 0
        self.event("accel", a_n > 0,
                   "VPS ACCEL OK" if a_n > 0
                   else "VPS ACCEL SILENT - FLOW ONLY, NO VIO",
                   self.SEV_INFO if a_n > 0 else self.SEV_CRIT)

        dm = est.get("depth_med_m")
        self.event("depth", dm is not None,
                   f"VPS STEREO OK {dm:.1f}m" if dm is not None
                   else "VPS STEREO OFF/OUT OF RANGE", self.SEV_INFO)

        # "NO MAP FIX" on its own reads as a failure when the usual cause is
        # simply being too low for the map to mean anything. Log 214 sat at
        # 0.0 m for 1260 s and sent 22 405 doomed submissions; the pilot saw
        # one warning with no number in it.
        min_alt = getattr(getattr(node, "worker", None), "min_alt_m", None)
        alt = est.get("alt")
        too_low = (min_alt is not None and alt is not None and alt < min_alt)
        if node.n_fix:
            self.event("fix", "yes", f"VPS MAP FIX {node.n_fix}", self.SEV_INFO)
        elif too_low:
            self.event("fix", "low",
                       f"VPS MAP NEEDS {min_alt:.0f}m - ab {alt:.0f}m",
                       self.SEV_INFO)
        else:
            self.event("fix", "none", "VPS NO MAP FIX", self.SEV_WARN)

        if t < self._next:
            return
        self._next = t + self.period
        # The heartbeat. Everything a pilot can act on, in 50 characters.
        self._send(f"VPS t{tracks} s{est['sigma']:.0f}m f{node.n_fix} "
                   f"a{est['alt']:.0f} v{v.speed_mps:.1f} "
                   f"{est.get('src') or '?'}", self.SEV_INFO)


class VpsNode:
    # How much faster the estimate doubts itself while dead reckoning. 3x is a
    # starting value, not a measured one: it wants a sweep against a recording
    # with deliberate flow dropouts.
    DR_TRUST = 3.0

    def __init__(self, cache: MapCache, focal_px: float, *,
                 fix_period_s: float = 1.0, radius_m: float = 150.0,
                 max_corners: int = 300, K=None, D=None, focal_y_px=None):
        self.cache = cache
        self.focal = float(focal_px)
        # fy is 791.58 against fx 791.72 on this camera - a ratio of 1.000177,
        # which is 1.8 cm per 100 m travelled. It is carried because it is free
        # and correct, NOT because it is a fix for anything.
        self.focal_y = float(focal_px if focal_y_px is None else focal_y_px)
        self.tracker = FlowTracker(max_corners=max_corners, K=K, D=D)
        self.fuser = MapFuser(flow_drift_frac=0.07, fix_sigma_m=12.0)
        self.worker = MapFixWorker(cache, focal_px=focal_px, radius_m=radius_m)
        self.fix_period = float(fix_period_s)
        self.origin = None            # (lon, lat)
        self.mlon = None
        self._next_fix = 0.0
        self.prev_att = None
        self.prev_t = None
        self.n_fix = self.n_flow = self.n_dr = 0
        # Flow AND acceleration on this path - step() pushes both. The bridge
        # still holds rather than dead-reckoning when no acceleration arrives,
        # which is what a replayed recording and a silent IMU both look like,
        # and says how stale it is either way.
        self.vel = VelocityBridge()

    def set_origin(self, lon: float, lat: float):
        self.origin = (float(lon), float(lat))
        self.mlon = M_PER_DEG_LAT * float(np.cos(np.radians(lat)))

    def to_lonlat(self, north: float, east: float):
        return (self.origin[0] + east / self.mlon,
                self.origin[1] + north / M_PER_DEG_LAT)

    def step(self, f: dict) -> dict | None:
        """One frame in, one position estimate out."""
        if f["alt"] is None or f["yaw"] is None or self.origin is None:
            return None
        att = (f["roll"], f["pitch"], f["yaw"])
        # What the FC says the tilt changed by, for comparison against what the
        # camera's own gyro integrated over the same interval. Derotation uses
        # the FC; this pair is logged so the gyro's axis mapping and sign can be
        # settled from a recording instead of guessed.
        def _wrap(a):
            return (a + 180.0) % 360.0 - 180.0
        fc_dr = fc_dp = None
        if self.prev_att is not None:
            fc_dr = _wrap(att[0] - self.prev_att[0])
            fc_dp = _wrap(att[1] - self.prev_att[1])
        # The camera's accelerometer, rotated into NED and handed to the
        # bridge. This is what makes it inertial rather than flow-only: flow
        # cannot see what happens BETWEEN frames, and at ~17 Hz that is 60 ms
        # of unobserved motion per gap.
        #
        # The body mapping is config.py's, measured 27 Sep: cam_z is DOWN (the
        # gravity vector lands on it at 99.8%), the nose points at the TOP of
        # the image and cam_y is image DOWN, so body forward = -cam_y and body
        # right = +cam_x. UNVERIFIED on THIS airframe - the camera has been
        # remounted since. accel_n is logged so a silent accelerometer reads as
        # silent, and the bridge treats flow as the authority regardless, so a
        # wrong mapping degrades the between-frame carry rather than the
        # answer.
        acc = f.get("accel")
        if acc and f.get("yaw") is not None:
            for ts, ax, ay, _az in acc:
                an, ae = accel_to_ne(ax, ay, f["yaw"])
                self.vel.push_accel(ts, an, ae)

        # The stereo pair, which until now was computed on the camera's own
        # VPU, carried all the way into this dict and then dropped. It is used
        # to keep the flow on ONE surface - see FlowTracker._depth_gate - and
        # not as a height, which 9 Oct already tried and reverted.
        d, ntrk = self.tracker.step(f["img"], depth=f.get("depth"))
        flowed = False
        if d is not None and self.prev_att is not None:
            d = d - attitude_shift_px(self.prev_att, att)
            dn, de = shift_to_ne(d, altitude_m=f["alt"], focal_px=self.focal,
                                 yaw_deg=f["yaw"], focal_y_px=self.focal_y)
            self.fuser.predict(dn, de)
            self.n_flow += 1
            # Velocity is the SAME displacement the fuser just consumed,
            # divided by the frame interval. Deriving it here rather than in a
            # second estimator means position and velocity cannot disagree
            # about what the camera saw - and disagreement between them is
            # what the EKF reads as a velocity innovation.
            if self.prev_t is not None:
                dt = f["t"] - self.prev_t
                if 0.0 < dt < 0.5:
                    self.vel.push_flow(f["t"], dn / dt, de / dt)
            flowed = True

        # Flow produced nothing this frame - tracking lost, too few corners, or
        # the first frame after a reset. THIS is the gap, and it is where the
        # accelerometer earns its place: the position carries on at the
        # integrated velocity instead of freezing where it was. Log 214 shows
        # why it matters - "TRACK LOST 17" inside the first second, and while
        # flow was down the estimate simply stopped moving, which the EKF reads
        # as a stationary aircraft rather than an unobserved one.
        #
        # The window is the bridge's own: once flow has been stale longer than
        # flow_timeout_s the velocity stops being valid and this stops with it.
        # Dead reckoning on a camera-grade IMU is good for a fraction of a
        # second, not for a flight, and the variance grows DR_TRUST times
        # faster while it lasts so the flight controller is told the estimate is
        # getting worse.
        if not flowed and self.prev_t is not None:
            dt = f["t"] - self.prev_t
            v = self.vel.value(f["t"], f["alt"])
            if 0.0 < dt < 0.5 and v.valid:
                self.fuser.predict(v.vn * dt, v.ve * dt, trust=self.DR_TRUST)
                self.n_dr += 1

        self.prev_att = att
        self.prev_t = f["t"]

        fix_used = False
        if f["t"] >= self._next_fix:
            lon, lat = self.to_lonlat(*self.fuser.pos)
            if self.worker.submit(f["img"], prior_lon=lon, prior_lat=lat,
                                  altitude_m=f["alt"], yaw_deg=f["yaw"]):
                self._next_fix = f["t"] + self.fix_period
        res = self.worker.poll()
        if res is not None and res.ok:
            fN = (res.lat - self.origin[1]) * M_PER_DEG_LAT
            fE = (res.lon - self.origin[0]) * self.mlon
            fix_used = self.fuser.update(fN, fE)
            self.n_fix += 1

        lon, lat = self.to_lonlat(*self.fuser.pos)
        return dict(north=float(self.fuser.pos[0]), east=float(self.fuser.pos[1]),
                    lon=lon, lat=lat, sigma=self.fuser.sigma_m, tracks=ntrk,
                    fix_used=fix_used, alt=f["alt"], yaw=f["yaw"], t=f["t"],
                    vel=self.vel.value(f["t"], f["alt"]),
                    fc_d_roll=fc_dr, fc_d_pitch=fc_dp,
                    gyro_d_roll=f.get("gyro_d_roll"),
                    gyro_d_pitch=f.get("gyro_d_pitch"),
                    gyro_n=f.get("gyro_n"), accel_n=f.get("accel_n"),
                    depth_med_m=self.tracker.depth_med_m,
                    depth_dropped=self.tracker.depth_dropped,
                    n_dr=self.n_dr)

    def stop(self):
        self.worker.stop()


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--replay", help="recorded flight folder")
    g.add_argument("--live", action="store_true", help="OAK-D + FC")
    ap.add_argument("--map", default=DEFAULT_MAP)
    ap.add_argument("--every", type=int, default=3, help="replay: frame stride (3 = 10 Hz)")
    ap.add_argument("--min-alt", type=float, default=60.0)
    ap.add_argument("--fix-period", type=float, default=0.5,
                    help="seconds between map fixes. 1.0 until 20 Oct, when "
                         "XFeat went from 1233 ms to 272 ms by being given "
                         "ONE torch thread instead of six - it could not "
                         "previously finish inside its own period. 0.5 s also "
                         "pairs with the 0.5 m cap on each correction: "
                         "together they give the map 1 m/s of authority to "
                         "pull the estimate back, with no single fix able to "
                         "move it more than half a metre.")
    ap.add_argument("--radius", type=float, default=150.0)
    ap.add_argument("--corners", type=int, default=300)
    ap.add_argument("--origin", help="lon,lat - GPS-denied start")
    ap.add_argument("--fc-port", default="/dev/ttyTHS1",
                    help="FC endpoint: a device, or udpout:127.0.0.1:14650 "
                         "through the router")
    ap.add_argument("--fc-baud", type=int, default=921600,
                    help="TELEM2 flies at 921600. Ignored for a udp endpoint.")
    ap.add_argument("--no-notify", action="store_true",
                    help="do not send STATUSTEXT to the radio")
    ap.add_argument("--notify-period", type=float, default=10.0,
                    help="seconds between VPS heartbeat lines on the radio")
    ap.add_argument("--no-scr-origin", action="store_true",
                    help="do not take the origin from SCR_USER1/2")
    ap.add_argument("--out", help="estimate log CSV")
    ap.add_argument("--send-fc", action="store_true",
                    help="transmit VISION_POSITION_ESTIMATE (disarmed only)")
    ap.add_argument("--send-gps", action="store_true",
                    help="inject as GPS2 via GPS_INPUT, Stride-style (disarmed only)")
    ap.add_argument("--liveness", default="",
                    help="file to touch each loop so the watchdog can tell "
                         "this node apart from a process that is merely alive. "
                         "systemd Restart= only catches a CRASH; a node stuck "
                         "waiting on the camera stays 'active' forever.")
    ap.add_argument("--no-send-vel", dest="send_vel", action="store_false",
                    help="do NOT send horizontal velocity in GPS_INPUT. It is "
                         "ON by default now, because a real receiver sends "
                         "velocity and the flight controller expects it. This "
                         "is the channel that reported 47.7 m/s at a 106 m "
                         "hover on 28 Sep and caused four EKF failsafes - "
                         "velocity_bridge.py exists for that, and its 23 tests "
                         "pin the collapse - but the first flight on it is "
                         "still the first flight on it, so this is the way off.")
    ap.set_defaults(send_vel=True)
    ap.add_argument("--crop-left", type=int, default=None,
                    help="pixels to cut off the LEFT edge, where the drone's "
                         "left leg sits in CAM_B's frame. Measured 20 Oct: "
                         "columns 0-63 are under 60%% of the frame median and "
                         "the taper ends by 120, so the default is 130. "
                         "0 disables it. cx moves with the crop.")
    ap.add_argument("--site-alt", type=float, default=None,
                    help="ground elevation AMSL in metres. GPS_INPUT needs a "
                         "real AMSL altitude and the EKF's own is partly our "
                         "doing - see AmslRef. Without this it is taken from "
                         "the real receiver's first 3D fix.")
    ap.add_argument("--gps-id", type=int, default=1, help="0 = GPS1, 1 = GPS2")
    a = ap.parse_args()

    # One copy, whoever started it. What was here asked systemctl whether a
    # unit called "drone-vio" was active; no such unit exists - they are
    # drone-router, drone-vps, drone-rc-switch and drone-watchdog - so it was
    # always inactive and the guard never fired once. See single_instance for
    # why a second copy is not harmless: the camera opens exclusively, so with
    # Restart=always in front of it the loser crash-loops, and every loop is a
    # hole in the GPS_INPUT stream. A hole is what takes the EKF's position
    # away and drops the aircraft into ALT_HOLD, which log 214 shows at 8.6 s.
    #
    # The fd is kept for the life of the process on purpose: closing it would
    # release the lock while the node was still flying.
    _lock_fd = single_instance.require("drone-vps")   # noqa: F841

    gps_sender = None
    if a.send_gps:
        gps_sender = GpsInputSender(a.fc_port, a.fc_baud, gps_id=a.gps_id,
                                    send_velocity=a.send_vel)
        print(f"GPS INPUT SENDER ON - disarmed confirmed. GPS{a.gps_id+1} ban kar jayega.\n"
              f"   (horiz_accuracy = fuser ka apna sigma)")

    sender = None
    if a.send_fc:
        sender = FcSender(a.fc_port, a.fc_baud)
        print(f"FC SENDER ON - disarmed confirmed. VISION_POSITION_ESTIMATE jayega.\n"
              f"   (EK3 sources ko haath nahi lagaya - FC padhega par istemal nahi karega)")

    # Read the typed take-off point BEFORE the camera source opens the serial
    # port, so this works whether --fc-port is a device or the router's UDP.
    scr_origin = None
    gps_primary = None
    if a.live and not a.no_scr_origin and not a.origin:
        scr_origin = read_scr_user(a.fc_port)
        gps_primary = read_gps_primary(a.fc_port)
        src_name = {0: "real GPS", 1: "VPS"}.get(gps_primary, "UNKNOWN")
        print(f"GPS_PRIMARY = {gps_primary} ({src_name})")
        if scr_origin is None:
            print("SCR_USER1/2: not readable or not set")
        else:
            print(f"SCR_USER1/2: {scr_origin[1]:.6f}, {scr_origin[0]:.6f}")

    src = (ReplaySource(a.replay, a.every, a.min_alt) if a.replay
           else LiveSource(a.fc_port, a.fc_baud, crop_left=a.crop_left))
    cache = MapCache(a.map)
    print(cache.summary())
    print(f"focal {src.focal:.1f} px, fix har {a.fix_period:.1f} s, window +-{a.radius:.0f} m, "
          f"{a.corners} corners\n", flush=True)

    node = VpsNode(cache, src.focal,
                   K=getattr(src, "K", None), D=getattr(src, "D", None),
                   focal_y_px=getattr(src, "focal_y", None),
                   fix_period_s=a.fix_period,
                   radius_m=a.radius, max_corners=a.corners)

    wr = None
    if a.out:
        fh = open(a.out, "w", newline="")
        wr = csv.writer(fh)
        # The velocity columns are not optional padding. The 8 Oct collapse -
        # 9.93 m/s in, 3.90 out - could not be diagnosed from any recording
        # because the state that caused it was never written down, and five
        # columns had to be added after the fact. Everything the bridge knows
        # goes in the row.
        wr.writerow(["t", "north_m", "east_m", "lon", "lat", "sigma_m",
                     "tracks", "fix_used", "alt_m", "yaw_deg",
                     "gps_north_m", "gps_east_m", "err_m",
                     "vel_vn", "vel_ve", "vel_speed", "vel_valid",
                     "vel_sent_vn", "vel_sent_ve", "vel_deadbanded",
                     "vel_deadband", "vel_age_s", "vel_flow_updates",
                     "vel_bias_n", "vel_bias_e", "vel_bias_frozen",
                     "vel_reason",
                     # Derotation: what the FC said the tilt did, and what the
                     # camera's own gyro integrated over the same interval.
                     # Logged side by side because the gyro's axis mapping and
                     # sign are a GUESS until a recording settles them.
                     "fc_d_roll", "fc_d_pitch",
                     "gyro_d_roll", "gyro_d_pitch", "gyro_n"])

    # Reuse a sender's link if one exists. If none does - a read-only ground
    # run - open a link only through the ROUTER. A second opener of the serial
    # device is precisely the contention the router exists to remove, and
    # LiveSource's Telemetry cannot be used: it is read-only by contract.
    _link = (gps_sender.m if gps_sender is not None
             else (sender.m if sender is not None else None))
    if _link is None and not a.no_notify:
        if a.fc_port.startswith(("udp", "tcp")):
            try:
                from pymavlink import mavutil as _mu2
                _link = _mu2.mavlink_connection(a.fc_port, source_system=255,
                                                source_component=222)
            except Exception as e:
                print(f"notifications off: {e}")
        else:
            print("notifications off: nothing is sending, and opening "
                  f"{a.fc_port} a second time would fight the recorder. "
                  "Use --fc-port udpout:127.0.0.1:14650 through the router.")
    notifier = Notifier(_link, period_s=a.notify_period,
                        enabled=not a.no_notify)
    if notifier.enabled:
        print(f"notifications ON, every {a.notify_period:.0f}s")

    # The link the watcher listens on is the sender's, because that is the one
    # that owns the FC endpoint. With no sender there is nothing to gate, so
    # the notification link will do.
    watcher = SourceWatcher(getattr(gps_sender, "m", None) or _link)
    amsl_ref = AmslRef(a.site_alt)

    _origin_refused = [False]
    errs, t_start, n = [], time.perf_counter(), 0
    try:
        while True:
            f = src.read()
            if f is None:
                break
            if node.origin is None:
                # Precedence, and the order is the 13 Oct fix: an explicitly
                # passed origin, then the point the PILOT TYPED on the RC, and
                # only then the live receiver. It used to be the receiver that
                # won whenever it had a fix, which is how the aircraft picked
                # up a stale site by itself after the pilot drove to a new one.
                #
                # Every start says which of the three it was and whether that
                # point is inside the loaded sheet, because a take-off point
                # outside it means no map fixes for the whole flight - the
                # first item on the 13 Oct list of things to confirm.
                cli = None
                if a.origin:
                    lo, la = (float(x) for x in a.origin.split(","))
                    cli = (lo, la)
                chosen, why = choose_origin(cli, scr_origin, gps_primary,
                                            f["gps"])
                if chosen is None:
                    if not _origin_refused[0]:
                        _origin_refused[0] = True
                        print(f"NO ORIGIN: {why}. The receiver cannot be used "
                              f"here.")
                        print("  Set SCR_USER1/2 from the RC, or pass "
                              "--origin lon,lat. Waiting.")
                        notifier.event("noorigin", True,
                                       "VPS NO ORIGIN - SET SCR_USER",
                                       Notifier.SEV_CRIT)
                    continue
                node.set_origin(*chosen)
                olon, olat = node.origin
                covered = cache.covers(olon, olat)
                print(f"origin from {why}: {olat:.6f}, {olon:.6f}")
                notifier.event("origin", why, f"VPS ORIGIN {why}"[:50],
                               Notifier.SEV_INFO)
                notifier.event("sheet", covered,
                               "VPS SHEET OK" if covered
                               else "VPS SHEET MISS - NO MAP FIX",
                               Notifier.SEV_INFO if covered else Notifier.SEV_CRIT)
                print(f"  sheet covers this point: "
                      f"{'YES' if covered else 'NO - there will be NO map fixes'}")
                if not covered:
                    print(f"  loaded sheet: {a.map}")
            est = node.step(f)
            if est is None:
                continue
            n += 1

            # Which source the aircraft is actually navigating on, NOW, not at
            # startup. The pilot's rule: on the real receiver nothing of ours
            # reaches the flight controller, and either way the change is
            # announced on the radio.
            watcher.poll(est["t"])
            on_vps = watcher.on_vps()
            est["src"] = ("VPS" if watcher.value == 1
                          else "GPS1" if watcher.value == 0 else "?")
            notifier.event(
                "source", est["src"],
                "VPS ON - FC ko GPS2 ja raha hai" if watcher.value == 1
                else "GPS1 ON - VPS ka data FC ko BAND" if watcher.value == 0
                else "SOURCE ? - GPS_PRIMARY padha nahi gaya",
                Notifier.SEV_INFO if watcher.value == 1 else Notifier.SEV_CRIT)
            gN = gE = err = None
            if f["gps"]:
                gN = (f["gps"][1] - node.origin[1]) * M_PER_DEG_LAT
                gE = (f["gps"][0] - node.origin[0]) * node.mlon
                err = float(np.hypot(est["north"] - gN, est["east"] - gE))
                errs.append(err)
            v = est["vel"]
            if wr:
                wr.writerow([f"{est['t']:.2f}", f"{est['north']:.2f}", f"{est['east']:.2f}",
                             f"{est['lon']:.7f}", f"{est['lat']:.7f}",
                             f"{est['sigma']:.1f}", est["tracks"], int(est["fix_used"]),
                             f"{est['alt']:.1f}", f"{est['yaw']:.1f}",
                             "" if gN is None else f"{gN:.2f}",
                             "" if gE is None else f"{gE:.2f}",
                             "" if err is None else f"{err:.2f}",
                             f"{v.vn:.3f}", f"{v.ve:.3f}", f"{v.speed_mps:.3f}",
                             int(v.valid), f"{v.sent_vn:.3f}", f"{v.sent_ve:.3f}",
                             int(v.deadbanded), f"{v.deadband_mps:.3f}",
                             f"{v.age_s:.3f}", v.flow_updates,
                             f"{v.bias_n:.5f}", f"{v.bias_e:.5f}",
                             int(v.bias_frozen), v.reason,
                             "" if est["fc_d_roll"] is None else f"{est['fc_d_roll']:.4f}",
                             "" if est["fc_d_pitch"] is None else f"{est['fc_d_pitch']:.4f}",
                             "" if est["gyro_d_roll"] is None else f"{est['gyro_d_roll']:.4f}",
                             "" if est["gyro_d_pitch"] is None else f"{est['gyro_d_pitch']:.4f}",
                             est["gyro_n"], est["accel_n"],
                             "" if est["depth_med_m"] is None
                             else f"{est['depth_med_m']:.2f}",
                             est["depth_dropped"], est["n_dr"],
                             est.get("src") or "",
                             "" if f.get("rngfnd") is None else f"{f['rngfnd']:.2f}",
                             "" if f.get("alt_amsl") is None else f"{f['alt_amsl']:.2f}"])
            notifier.tick(est["t"], est, node)

            # Gated on the source. GPS_PRIMARY=0 means the pilot asked for the
            # real M9N, so the VPS stops injecting entirely - not "sends and is
            # ignored", stops. Unknown keeps sending; see SourceWatcher for why
            # the two directions are not symmetrical.
            if gps_sender is not None and on_vps:
                amsl_ref.observe(f.get("gps1_fix"), f.get("gps1_alt"), f["alt"])
                amsl = amsl_ref.amsl(f["alt"])
                notifier.event("amsl", amsl_ref.source or False,
                               f"VPS AMSL {amsl_ref.source}"[:50]
                               if amsl_ref.source
                               else "VPS AMSL ANJAAN - --site-alt do",
                               Notifier.SEV_INFO if amsl_ref.source
                               else Notifier.SEV_WARN)
                gps_sender.send(est["lon"], est["lat"],
                                est["alt"] if amsl is None else amsl,
                                est["sigma"], est["yaw"], vel=est["vel"])
            if sender is not None:
                sender.send(est["north"], est["east"], -est["alt"],
                            f["roll"], f["pitch"], f["yaw"])
            if a.liveness:
                # Written, not just touched: the mtime says the loop ran and
                # the contents say what it was doing, so a watchdog log shows
                # WHY it stalled rather than only that it did.
                try:
                    tmp = a.liveness + ".tmp"
                    with open(tmp, "w") as lf:
                        lf.write(f"t={est['t']:.2f} n={n} tracks={est['tracks']} "
                                 f"sigma={est['sigma']:.1f} fixes={node.n_fix} "
                                 f"flow={node.n_flow} vel={v.reason}\n")
                    os.replace(tmp, a.liveness)   # atomic; never a half file
                except OSError:
                    pass
            if n % 200 == 0:
                print(f"  {est['t']:7.1f}s  N {est['north']:8.1f}  E {est['east']:8.1f}  "
                      f"sigma {est['sigma']:5.1f}  tracks {est['tracks']:4d}  "
                      f"fixes {node.n_fix:3d}"
                      + (f"  err {err:6.1f} m" if err is not None else ""), flush=True)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        node.stop()
        if wr:
            fh.close()

    dt = time.perf_counter() - t_start
    print(f"\n{n} frames in {dt:.1f} s = {n/max(dt,1e-9):.1f} Hz processed")
    print(f"flow steps {node.n_flow}, map fixes {node.n_fix} "
          f"(worker: {node.worker.submitted} submitted, {node.worker.dropped} dropped), "
          f"fuser accepted {node.fuser.n_fix} rejected {node.fuser.n_rejected}")
    if gps_sender is not None:
        print(f"GPS_INPUT bheje: {gps_sender.sent}")
    if sender is not None:
        print(f"VISION_POSITION_ESTIMATE bheje: {sender.sent}")
    if errs:
        e = np.array(errs)
        print(f"GPS ke against: median {np.median(e):.1f} m  p90 {np.percentile(e,90):.1f} m  "
              f"max {e.max():.1f} m  final {e[-1]:.1f} m")


if __name__ == "__main__":
    main()
