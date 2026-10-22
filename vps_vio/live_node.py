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
import argparse, collections, csv, io, math, os, sys, time
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flow_odom import (FlowTracker, shift_to_ne, attitude_shift_px,
                       gyro_shift_px, TiltValidator)
from map_cache import MapCache
from map_fix_worker import MapFixWorker
from map_fuse import MapFuser
from velocity_bridge import VelocityBridge
from frame_writer import FrameWriter, next_run_number
import single_instance

M_PER_DEG_LAT = 111_320.0

# WHICH MONO CAMERA EVERYTHING IS MEASURED IN. CAM_C since 22 Oct, at the
# pilot's instruction; it was CAM_B before that.
#
# The aircraft's landing leg hangs in CAM_B's left edge - photographed on
# 22 Oct with the airframe lifted, the leg and its foam grip fill x 0-75 and
# a bright strut runs to about x 165. That is what config.crop_left_px = 130
# existed to cut off, and the crop cost 10% of the frame's width to flow and
# moved cx 80 px off the image centre, which is 11 m of ground at 110 m and
# had to be carried through the whole map-query chain by hand. CAM_C's left
# edge is clear in the same photograph, so on CAM_C there is nothing to crop
# and cx is only 4 px off centre.
#
# Everything measured in the camera frame has to come from the SAME socket or
# the frames silently disagree: intrinsics, distortion, the IMU extrinsics the
# accelerometer is rotated by, the pixel the map fix is reported for, and the
# frame the depth gate indexes. They all read these two names.
PRIMARY_SOCKET_NAME = "CAM_C"
STEREO_PARTNER_NAME = "CAM_B"


def _socket(name: str):
    """Named here so a typo is an AttributeError at startup, not a wrong frame."""
    import depthai as dai
    return getattr(dai.CameraBoardSocket, name)
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
        raise RuntimeError("no fx= found in camera.txt")

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
                 crop_left: int | None = None, imu_rate: int | None = None):
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
        imu_rate = cfg.imu_rate if imu_rate is None else int(imu_rate)
        self.imu_rate = imu_rate
        # 921600, not 115200. TELEM2 flies at 921600 - SERIAL2_BAUD reads 921
        # on the FC. At the wrong baud the link produces nothing, alt and yaw
        # stay None, and VpsNode.step returns None on every frame: the node
        # runs, finds its origin, and silently never estimates anything.
        self.tel = Telemetry(fc_port, fc_baud).start()

        p = dai.Pipeline()
        mono = {}
        # The primary alone unless depth is wanted; depth needs the pair.
        sockets = [("mono", _socket(PRIMARY_SOCKET_NAME))]
        # IR LASER AND FLOOD OFF, at the pilot's instruction. The OAK-D Pro's
        # dot projector and flood LED do nothing for this mission - the ground
        # is tens of metres away and the stereo pair is only used as a
        # same-surface check below 30 m - and they draw power from the same
        # rail as the sensors. They default to 0 on this device and are set
        # explicitly so a firmware default cannot turn them on quietly.
        #
        # The RGB sensor (CAM_A, IMX378) is not in this pipeline at all, which
        # is the other half of what was asked: it is never created, so it is
        # never streamed, encoded or powered for our sake.
        self._ir_off = True

        if stereo_depth:
            sockets.append(("mono_partner", _socket(STEREO_PARTNER_NAME)))
        for name, sock in sockets:
            m = p.create(dai.node.MonoCamera)
            m.setBoardSocket(sock)
            # THE_800_P IS THIS SENSOR'S MAXIMUM, not a choice. Asked on
            # 22 Oct whether the monos could run at 1200 as an earlier build
            # was remembered doing: the device itself says no. CAM_B and CAM_C
            # are OV9282 and report exactly three sizes - 640x400, 1280x720
            # and 1280x800. THE_1200_P is in depthai's enum for other sensors,
            # and asking for it here crashes the Myriad X firmware with
            # RTEMS_FATAL_SOURCE_ASSERT 4.4 s after boot, every configuration,
            # even one stream with no depth and no IMU, while 800p runs at
            # 29.9 Hz.
            #
            # The only resolution headroom on this camera is CAM_A, the
            # IMX378, which offers up to 4056x3040 and has fx about 1943 at
            # 2560x1440 against this sensor's 791.72 - 2.45x the pixels on the
            # same ground. It is not in this pipeline.
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
        # The rate for all three reports; they share one packet. See
        # VpsConfig.imu_rate for why 100 is the default and what it costs.
        #
        # This comment used to say 100 was "not a choice, it is the ceiling",
        # on Luxonis' table giving GYROSCOPE_CALIBRATED a maximum of 100 on
        # the BNO086. Measured on THIS device 21 Oct with the IMU alone, a
        # request for 400 delivered 357.9 Hz on the calibrated gyro and on
        # LINEAR_ACCELERATION alike, so the table is not the ceiling here.
        # What remains true is the second half: the BNO rounds a request up to
        # its next supported rate, and 3 Oct measured 400 with the camera
        # streams running dragging the whole stream DOWN - so the cost is in
        # the packet, not in the part.
        imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_CALIBRATED, imu_rate)
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
        imu.enableIMUSensor(dai.IMUSensor.LINEAR_ACCELERATION, imu_rate)
        # GAME_ROTATION_VECTOR: the BNO's own fused orientation, WITHOUT the
        # magnetometer. At the pilot's instruction, 21 Oct, and it costs one of
        # the two free report slots and nothing else.
        #
        # depthai's IMUPacket has exactly four slots - acceleroMeter,
        # gyroscope, magneticField, rotationVector - and every sensor of a
        # family lands in its family's slot. That is why the device refuses
        # LINEAR_ACCELERATION beside ACCELEROMETER_RAW, and ROTATION_VECTOR
        # beside GAME_ROTATION_VECTOR: each pair shares a slot. So this takes
        # `rotationVector`, which was empty, and leaves `magneticField` free.
        #
        # GAME and not plain ROTATION_VECTOR, measured on this airframe 21 Oct
        # with the aircraft standing still:
        #
        #     ROTATION_VECTOR        yaw drifted 18.9 deg/min, and the BNO's
        #                            own accuracy estimate was 1.375 rad - 79
        #                            degrees. Its magnetometer sits beside the
        #                            motors and is unusable.
        #     GAME_ROTATION_VECTOR   yaw stable to 0.02 deg over 40 s
        #
        # Both settle rather than drift: the roll moved -0.71 deg in the first
        # half of a 40 s run and -0.01 in the second, so about five seconds of
        # start-up convergence and then flat.
        #
        # IT IS NOT WHAT DEROTATION USES, and that is deliberate. Derotation
        # needs the rotation BETWEEN two frames, 33 ms apart, and a slowly
        # accumulating offset cancels in a difference - so the gyro's unbounded
        # drift costs it nothing, while this adds nothing. It is enabled to be
        # logged beside the gyro and the FC's attitude, so the next flight
        # settles which to fly on with real numbers instead of argument.
        #
        # Cost, measured: four reports in the packet run at 95.1 Hz against
        # 99.6 for two. 4.5%.
        imu.enableIMUSensor(dai.IMUSensor.GAME_ROTATION_VECTOR, imu_rate)
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
            # Aligned to the PRIMARY, because FlowTracker's depth gate looks
            # depth up at the pixel coordinates of its own tracks. Aligned to
            # the other socket, every one of those lookups would be off by the
            # pair's disparity.
            st.setDepthAlign(_socket(PRIMARY_SOCKET_NAME))
            # left/right here are PHYSICAL eyes, not primary/partner: CAM_B is
            # the left eye on this board and CAM_C the right. Linked the other
            # way round the disparity changes sign and the depth comes back as
            # noise, so these are keyed by socket NAME - changing which camera
            # is primary must not be able to swap them.
            by_socket = {PRIMARY_SOCKET_NAME: mono["mono"],
                         STEREO_PARTNER_NAME: mono["mono_partner"]}
            by_socket["CAM_B"].out.link(st.left)
            by_socket["CAM_C"].out.link(st.right)
            # The leg costs depth its left strip, whichever camera is primary.
            # Stereo needs BOTH eyes on the same ground, and CAM_B's x 0-165 is
            # landing leg, so no match exists there and depth is invalid in that
            # strip. Bounded: it is 13% of the frame's width, and depth is only
            # used as a same-surface check below 30 m.
            xd = p.create(dai.node.XLinkOut)
            xd.setStreamName("depth")
            st.depth.link(xd.input)

        self.dev = dai.Device()
        # Both off, explicitly. See the note where _ir_off is set: neither does
        # anything for ground tens of metres away, and a firmware default must
        # not be able to turn them on quietly. Wrapped because the call does
        # not exist on every OAK model and a missing projector is not an error.
        for _setter in ("setIrLaserDotProjectorBrightness",
                        "setIrFloodLightBrightness"):
            try:
                getattr(self.dev, _setter)(0)
            except Exception:
                pass
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
            E = np.array(cal.getImuToCameraExtrinsics(_socket(PRIMARY_SOCKET_NAME)))
            R = np.array(E)[:3, :3]
            # A rotation and nothing else: orthonormal, det +1. Anything that
            # fails this is a calibration we do not understand, and silently
            # multiplying by it would be worse than not rotating at all.
            if (np.allclose(R @ R.T, np.eye(3), atol=1e-3)
                    and abs(np.linalg.det(R) - 1.0) < 1e-3):
                self.R_imu2cam = R
                off = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
                print(f"IMU->{PRIMARY_SOCKET_NAME} rotation read from calibration "
                      f"({off:.1f} deg off the camera frame)")
            else:
                print(f"*** IMU->{PRIMARY_SOCKET_NAME} extrinsics is not a "
                      f"rotation - IGNORED ***")
        except Exception as e:
            print(f"*** could not read IMU->{PRIMARY_SOCKET_NAME} extrinsics ({e}) "
                  f"- treating the IMU "
                  f"axes as the camera axes. The accel/gyro mapping is "
                  f"UNVERIFIED in that frame. ***")

        K = np.array(cal.getCameraIntrinsics(_socket(PRIMARY_SOCKET_NAME),
                                             1280, 800))
        self.focal = float(K[0][0])
        # The WHOLE of K, and the distortion, not just fx. Measured on this
        # device 22 Oct for CAM_C, the primary: cx 644.13 / cy 404.16, which is
        # 4.13 px and 4.16 px off the geometric centre (CAM_B, the old primary,
        # read cx 624.89 / cy 391.80 - 15.11 px and 8.20 px
        # off it, and 80 px off once its 130 px leg crop had moved cx). The
        # distortion is read per socket too and the two genuinely differ, so
        # this is a per-unit calibration and not a defaulted one: CAM_B's k1 is
        # 24.09 / k2 -39.14, CAM_C's 26.05 / -41.26. Both were being thrown away
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
            print(f"crop: left {self.crop_left} px (the aircraft leg), "
                  f"cx {K[0][2] + self.crop_left:.1f} -> {K[0][2]:.1f}")
        self.K = K
        self.focal_y = float(K[1][1])
        try:
            self.D = np.array(list(cal.getDistortionCoefficients(
                _socket(PRIMARY_SOCKET_NAME))), float)
        except Exception as e:
            self.D = None
            print(f"*** could not read the distortion coefficients ({e}) - flow will "
                  f"run on raw pixels ***")
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
        # The previous fused orientation, for the per-frame delta.
        self._q_prev = None
        self._quat_n = 0
        self.t0 = time.monotonic()

    def _quat_delta_rp(self, q0, q1):
        """Degrees of roll and pitch between two orientations, camera frame.

        The RELATIVE rotation, not the absolute attitude: q0^-1 * q1 as an
        axis-angle vector, which is the same quantity the gyro integral
        produces, so the two columns can be compared without any further
        conversion. Taking Euler angles of each quaternion and subtracting
        would not be - that goes wrong near gimbal lock and mixes the axes.

        The axis-angle vector is then put through _to_cam, exactly as the gyro
        sample is, so both carry the same mounting assumption and the same
        caveat: x->roll, y->pitch is a GUESS until a recording settles it.
        """
        x0, y0, z0, w0 = q0
        x1, y1, z1, w1 = q1
        # q0 conjugate times q1
        rw = w0 * w1 + x0 * x1 + y0 * y1 + z0 * z1
        rx = w0 * x1 - x0 * w1 - y0 * z1 + z0 * y1
        ry = w0 * y1 + x0 * z1 - y0 * w1 - z0 * x1
        rz = w0 * z1 - x0 * y1 + y0 * x1 - z0 * w1
        if rw < 0.0:                      # shortest arc
            rx, ry, rz, rw = -rx, -ry, -rz, -rw
        s = float(np.sqrt(max(0.0, 1.0 - rw * rw)))
        ang = 2.0 * float(np.arctan2(s, min(1.0, max(-1.0, rw))))
        if s < 1e-9:
            vx = vy = vz = 0.0
        else:
            vx, vy, vz = rx / s * ang, ry / s * ang, rz / s * ang
        cx, cy, _cz = self._to_cam(vx, vy, vz)
        return float(np.degrees(cx)), float(np.degrees(cy))

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
        # Every gyro sample kept, not just their integral. The integral is what
        # derotation needs; the samples are what a VIBRATION number needs, and
        # on 20 Oct there was no way to answer "how much was the camera itself
        # shaking" because only the per-frame integral had ever been written
        # down. 100 Hz is the BNO086's ceiling for GYROSCOPE_CALIBRATED, so
        # this band reaches 50 Hz: the 47 Hz prop fundamental sits right at the
        # edge of it and anything above folds in. Treat >40 Hz here as "there
        # is energy up there", never as a calibrated amplitude.
        gyro = []
        # The fused orientation, as a per-frame delta so it sits beside
        # gyro_d_roll and fc_d_roll and can be compared directly. The raw
        # quaternion goes to the IMU log.
        q_roll = q_pitch = 0.0
        quat = []
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
                # acceleroMeter, not linearAcceleration - there is no field
                # by that name. Every accelerometer-family report lands in
                # this one slot, and WHICH of them it carries is decided by
                # enableIMUSensor above: we ask for LINEAR_ACCELERATION, so
                # what arrives here has already had gravity removed.
                #
                # Reading a field that does not exist cost the whole feature
                # silently: getattr returned None on every packet, accel_n sat
                # at 0 for the entire run, and the velocity bridge went on
                # being flow-only while the code read as though it were not.
                # The CSV is what showed it.
                la = getattr(packet, "acceleroMeter", None)
                if la is not None:
                    ts = la.getTimestampDevice().total_seconds()
                    if self._t_accel is None or ts > self._t_accel:
                        if self._t_accel is not None:
                            ax, ay, az = self._to_cam(la.x, la.y, la.z)
                            accel.append((ts, ax, ay, az))
                            self._accel_n += 1
                        self._t_accel = ts
                # The fused orientation, read BEFORE the gyro's `continue`.
                # A packet can carry one report and not another, and putting
                # this after that continue is how it would silently never run.
                rv = getattr(packet, "rotationVector", None)
                if rv is not None:
                    ts_q = rv.getTimestampDevice().total_seconds()
                    q = (float(rv.i), float(rv.j), float(rv.k), float(rv.real))
                    if self._q_prev is not None:
                        dr, dp = self._quat_delta_rp(self._q_prev, q)
                        q_roll += dr
                        q_pitch += dp
                    self._q_prev = q
                    self._quat_n += 1
                    quat.append((ts_q,) + q)

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
                        gx, gy, gz = self._to_cam(g.x, g.y, g.z)
                        d_roll += np.degrees(gx) * dt
                        d_pitch += np.degrees(gy) * dt
                        self._gyro_n += 1
                        gyro.append((ts, gx, gy, gz))
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
                        quat_d_roll=q_roll, quat_d_pitch=q_pitch,
                        quat=quat, quat_n=self._quat_n,
                        accel=accel, accel_n=self._accel_n, gyro=gyro,
                        t=time.monotonic() - self.t0)
        gps = None
        if s.get("lat") not in (None, 0.0):
            gps = (s["lon"], s["lat"])
        return dict(img=self._crop(fr.getCvFrame()), alt=s["alt_rel"], yaw=s["yaw"],
                    alt_amsl=s.get("alt_amsl"), rngfnd=s.get("rngfnd"),
                    gps1_alt=s.get("gps1_alt"), gps1_fix=s.get("fix"),
                    gps1_hacc=s.get("gps1_hacc"),
                    # WHAT THE EKF THINKS OF ITS OWN SOLUTION. The recorder has
                    # read these since it was written and nothing ever logged
                    # them, so FS_EKF_THRESH has been argued about rather than
                    # measured. They are the normalised innovation test ratios
                    # - how far the EKF's prediction and each sensor disagree -
                    # and FS_EKF_THRESH is the threshold on the worst of them.
                    # ArduPilot documents 0.6 strict, 0.8 default, 1.0 relaxed;
                    # this aircraft flies 1.2, looser than the loosest. One
                    # flight on the VPS with these logged settles it.
                    ekf_vel_var=s.get("ekf_vel_var"),
                    ekf_pos_h_var=s.get("ekf_pos_h_var"),
                    ekf_comp_var=s.get("ekf_comp_var"),
                    ekf_flags=s.get("ekf_flags"),
                    # The RAW receiver, separate from `gps` above. `gps` is
                    # GLOBAL_POSITION_INT - the EKF's fused position - which on
                    # a VPS flight is partly our own injection coming back. This
                    # one is the receiver itself and is the only independent
                    # truth on board.
                    gps1_raw=((s["gps1_lon"], s["gps1_lat"])
                              if s.get("gps1_lat") not in (None, 0.0) else None),
                    roll=s["roll"], pitch=s["pitch"], gps=gps, depth=depth,
                    gyro_d_roll=d_roll, gyro_d_pitch=d_pitch,
                    gyro_n=self._gyro_n,
                    quat_d_roll=q_roll, quat_d_pitch=q_pitch,
                    quat=quat, quat_n=self._quat_n,
                    accel=accel, accel_n=self._accel_n, gyro=gyro,
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
            raise RuntimeError("no heartbeat from the flight controller")
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
    # m/s of velocity error per metre of altitude, fitted to three flights.
    # See velocity_fields for the table it comes from.
    # Speed accuracy: a floor plus a small altitude term. 0.0867 x altitude was
    # fitted on the RAW flow, before the 5 m/s^2 output slew was put back, and
    # the slew is what removes most of that error - so the old coefficient now
    # declares roughly three times the error the aircraft actually sends.
    #
    # Re-measured on the thirteen recorded VPS flights, sigma of what LEAVES the
    # bridge (hovers, where the true speed is about zero so the reported speed
    # IS the error):
    #
    #     altitude      5      15      25      38      52      70 m
    #     raw        0.07    0.87    2.16    4.57    4.30    5.46 m/s
    #     slewed     0.56    0.69    0.63    1.17    1.20    2.39 m/s
    #
    # The slewed column is nearly flat, because the slew bounds the rate of
    # change rather than the magnitude - it is not proportional to height the
    # way the raw flow error is. Four models against those six numbers:
    #
    #                             rms error   worst under-statement
    #     0.0867 x alt  (was)         2.305        +0.128 m/s
    #     0.0305 x alt                0.233        +0.259
    #     flat 1.20                   0.636        +1.194
    #     0.6 + 0.025 x alt           0.432        +0.044
    #
    # The last one is chosen on the under-statement column, not the rms. An
    # accuracy declared to an EKF is a promise, and the failure that matters is
    # claiming to be better than you are - 0.0305 x alt fits more tightly but
    # under-states by a quarter of a metre per second somewhere, and the flat
    # model under-states by 1.2. This one is never more than 0.044 m/s optimistic
    # across the whole range, and at 70 m it declares 2.35 m/s against a measured
    # 2.39 - against log 00000194's p50 of 0.93 on a quieter day.
    #
    # 22 OCT: THE PER_M TERM WAS NO LONGER TRUE AND THE PROMISE WAS BEING
    # BROKEN. Those six numbers came from hovers on thirteen flights that
    # mostly sat low. Log 226 flew to 82.7 m with the bridge's own inputs
    # recorded, and the PER-AXIS error against the raw receiver reads:
    #
    #     altitude        5-25    25-45    45-70    70-85 m
    #     real error sd   0.41     2.09     2.05     4.19 m/s
    #     declared        0.92     1.50     2.02     2.66 m/s
    #     ratio           0.44     1.40     1.02     1.57
    #
    # So at the heights the aircraft actually flies, the declaration was
    # optimistic by 0.7 to 1.5 m/s - not 0.044 - and XKF4.SV is exactly
    # error / declared. SV sat at p50 1.26 against FS_EKF_THRESH 1.2 and put
    # the aircraft in ALT_HOLD twice.
    #
    # The slewed column being "nearly flat" was an artefact of the altitude
    # range it was measured over. Fitted on log 226's per-band sigma, the
    # error is proportional to height after all - which it must be, because
    # velocity is pixels x altitude / focal, so one pixel of frame-to-frame
    # jitter is 0.10 m at 80 m and 3.03 m/s at 30 fps:
    #
    #     per-band sd = 0.0476 x altitude - 0.11
    #
    # Keeping the 0.6 floor (it is pessimistic below 25 m, which is the safe
    # direction) gives 0.6 + 0.0476 x alt: 2.27 m/s at 35 m against 1.48
    # before, and 4.55 at 83 m against 2.68. Predicted SV p50 1.26 -> 0.82.
    #
    # THIS DOES NOT MAKE THE VELOCITY BETTER. It stops the EKF being lied to
    # about how bad it is. The velocity itself is still noise proportional to
    # height, and the fix for that is a longer measurement baseline, not a
    # number here.
    VEL_ACC_FLOOR = 0.6
    VEL_ACC_PER_M = 0.048
    # The vertical accuracy the working flights declared: a constant 3.0 m
    # (GPA.VAcc is exactly 3.00 throughout logs 00000191 and 00000194). It had
    # been sigma_m * 2 here, which at a typical sigma of 21 m declares 42 m of
    # vertical error - and EK3_SRC1_POSZ is the barometer anyway, so the only
    # thing a wild number here can do is harm.
    VERT_ACC_M = 3.0

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
            raise RuntimeError("no heartbeat from the flight controller")
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
        # 4.3 Hz, which is what logs 00000191 and 00000194 actually carried.
        self.send_period_s = 1.0 / 4.3
        # Seeded on the first send, then advanced by exactly one period each
        # time - see the gate in send() for why that distinction is worth 0.4 Hz.
        self._next_send = None
        self.skipped_rate = 0
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
        # Speed accuracy the EKF can weigh, and it MUST grow with height.
        #
        # This used to be a flat 0.4 m/s widened only by staleness, and that
        # told the flight controller the velocity was good to 0.4 m/s while
        # the aircraft hovered at 70 m and the VPS reported up to 24.6 m/s.
        # The EKF believed it, because we said it was accurate.
        #
        # Measured across the three flights of 20 Oct, in hovers where the
        # true speed is about zero so the reported speed IS the error:
        #
        #     0-10 m   sigma 0.126 m/s      0.078 per metre of altitude
        #    10-20 m         0.811          0.049
        #    20-30 m         1.826          0.077
        #    30-45 m         4.569          0.118
        #    45-60 m         4.131          0.078
        #    60-80 m         5.713          0.082
        #
        # A straight line through those is sigma = 0.0867 x altitude, and the
        # per-metre figure barely moves across the whole range - which is what
        # you expect, because the flow's error is roughly constant in PIXELS
        # and metres per pixel is altitude / focal length.
        #
        # So the honest number is altitude-scaled. At 20 m it says 1.7 m/s, at
        # 70 m it says 6.1 - and the EKF then weighs it accordingly instead of
        # being lied to. The staleness term stays: between flow updates this is
        # dead reckoning on the camera accelerometer.
        acc = (self.VEL_ACC_FLOOR
               + self.VEL_ACC_PER_M * max(0.0, vel.altitude_m)
               + 2.0 * vel.age_s)
        # A CAPPED SAMPLE IS NOT A MEASUREMENT, so it is not declared as one.
        #
        # max_sent_mps bounds the magnitude the bridge may send. When it bites,
        # the estimate had left what the airframe can physically do - the cap
        # is 6.0 m/s against LOIT_SPEED_MS of 3.0 and an ATC_ANGLE_MAX of 15
        # degrees - so what goes out is the ceiling, not a reading, and the
        # only honest uncertainty for it is the whole range it could have been.
        #
        # Measured on log 231, the first flight with the cap, sent speed
        # against the receiver's own:
        #
        #     altitude   at the cap   real speed p50   we declared
        #      5-25 m       0.0%          0.75            1.30 m/s
        #     25-40 m       0.6%          0.88            1.95
        #     40-60 m       7.8%          1.52            2.69
        #     60-80 m      31.0%          2.30            4.27
        #
        # So at 60-80 m we were asserting 6.00 m/s to 4.27 m/s while the
        # aircraft did 2.30 - an innovation near 4 against a sigma of 4.3,
        # which is XKF4.SV at the gate. Those are the three switches above
        # 30 m that took ALT_HOLD; the two at 27.7 and 28.5 m did not.
        #
        # This keys off the CAP FIRING, not off height, and the cap fired on
        # 0.0% of samples below 25 m - so nothing below 30 m changes, which is
        # the pilot's condition and the measurement rather than a threshold
        # chosen to satisfy it.
        if getattr(vel, "capped", False):
            acc = max(acc, float(getattr(vel, "cap_mps", 0.0)))
        return ign, float(vel.sent_vn), float(vel.sent_ve), float(acc), "sent"

    def send(self, lon, lat, alt_amsl_m, sigma_m, yaw_deg=None, vel=None):
        t = time.time()
        # RATE GATE. The flight controller is told this is a GPS receiver, and
        # a receiver arrives at its own cadence - GPS2_RATE_MS is 200 ms on
        # this aircraft. The flights that worked sent GPS_INPUT at 4.28 and
        # 4.31 Hz (logs 00000191 and 00000194, 12 and 13 Oct), and the 5 Oct
        # report records the design as "GPS_INPUT at 4.2 Hz".
        #
        # Without this gate the node sends on every flow frame - 30 Hz, seven
        # times the rate - and that is not merely wasteful. Each message
        # carries the SAME fused estimate as the one before it, so seven
        # correlated samples arrive where the EKF expects one independent
        # measurement, and it weighs them as though they were independent.
        # The effect is to trust the VPS by roughly the square root of that
        # over-sampling, about 2.6x more than the sigma we are declaring.
        if self._next_send is not None and t < self._next_send:
            self.skipped_rate += 1
            return
        # Advance from the DEADLINE, not from now. `_next_send = t + period`
        # looks equivalent and is not: the gate can only fire on a frame, the
        # frames arrive every 30 ms, so each send lands a little after its
        # deadline and starting the next period from there loses that remainder
        # every time. Measured over the thirteen recorded VPS flights, on their
        # real frame timestamps:
        #
        #     _next_send = t + period        3.913 Hz
        #     _next_send += period           4.295 Hz      target 4.30
        #
        # and log 00000194, which flew, carried 4.31 Hz. The first form is not
        # a rounding difference, it is a systematic 9% shortfall.
        #
        # After a real gap - a stalled camera, a reconnect - the deadline would
        # be far in the past and the gate would then fire on every frame until
        # it caught up, which is the 30 Hz burst this gate exists to prevent.
        # So a deadline already behind is restarted from the present instead.
        if self._next_send is None:
            self._next_send = t + self.send_period_s
        else:
            self._next_send += self.send_period_s
            if self._next_send < t:
                self._next_send = t + self.send_period_s
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
            speed_acc, float(sigma_m), self.VERT_ACC_M,
            self.sats)
        self.sent += 1
        if not (ign & self.IGN_VEL_HORIZ):
            self.sent_with_vel += 1


# --------------------------------------------------------------------------- #
# The estimate log's columns, and the row that fills them, as one pair.
#
# 20 Oct: seven columns were added to the ROW and not to the header. The file
# still opened, every row was 39 fields wide under a 32-field header, and
# everything past the 32nd column was silently attributed to the wrong name -
# 1162 rows of it before anyone looked. A CSV that is wrong in that way is
# worse than one that fails to open.
#
# They live here as a constant and a function so a test can run both and
# compare their widths, instead of reading the source and hoping.
CSV_COLUMNS = ["t", "north_m", "east_m", "lon", "lat", "sigma_m",
    "tracks", "fix_used", "alt_m", "yaw_deg",
    "gps_north_m", "gps_east_m", "err_m",
    "vel_vn", "vel_ve", "vel_speed", "vel_valid",
    "vel_sent_vn", "vel_sent_ve", "vel_deadbanded",
    "vel_deadband", "vel_age_s", "vel_flow_updates",
    "vel_bias_n", "vel_bias_e", "vel_bias_frozen",
    "vel_reason",
    # The bridge's two INPUTS. Added 21 Oct because log 225 proved the
    # accel blend was inflating the speed and then could not be replayed
    # at a different tau_s: neither term was written down.
    "vel_flow_vn", "vel_flow_ve", "vel_accel_n", "vel_accel_e",
    # THE MAP FIX ITSELF. Added 22 Oct. fix_used said only that one was
    # applied; where it actually put the aircraft was never recorded, so the
    # matcher's error by height had to be read off an archived report whose
    # configuration carried an 8.61 m east bias this code never had. Now the
    # fix's own position is in the log and the dependence is ours to measure.
    "fix_lon", "fix_lat", "fix_score", "fix_inliers", "fix_alt_m",
    # THE STEP GATE'S INPUT. A refused step enters the position as zero and a
    # clipped one at the ceiling, so n_step_rej counts the bites but nothing
    # records what was bitten. 22 Oct.
    "step_px", "step_action",
    # Derotation: what the FC said the tilt did, and what the
    # camera's own gyro integrated over the same interval.
    # Logged side by side because the gyro's axis mapping and
    # sign are a GUESS until a recording settles them.
    "fc_d_roll", "fc_d_pitch",
    "gyro_d_roll", "gyro_d_pitch", "gyro_n",
    # The THIRD opinion on how far the aircraft turned across this frame, from
    # GAME_ROTATION_VECTOR. Logged beside the FC's and the gyro's so one flight
    # settles which derotation should use. Measured 21 Oct on the 20-21 Oct
    # recordings: the FC's reads 0.000 on 88% of frames, and replacing it with
    # the gyro - sign flipped - took the sent-velocity error from p95 3.58 to
    # 1.22 m/s. This column is what tests the same claim for the fused source.
    "quat_d_roll", "quat_d_pitch", "quat_n",
    # Which of the three actually derotated this frame, and - where both the
    # gyro and the fused orientation were present - how far they disagreed.
    # The disagreement is the measurement that says whether
    # GAME_ROTATION_VECTOR belongs above or below the FC attitude as a
    # fallback, and whether a gyro has started to go wrong.
    "derot_src", "derot_dis_roll", "derot_dis_pitch",
    # Added 20 Oct with the rows that carry them. The header
    # was NOT updated at the time and 1162 rows were written
    # 39 wide under a 32-wide header before it was noticed -
    # a file that opens fine and is wrong in every column
    # after the 32nd. test_csv_columns.py now pins the two
    # together so the next addition cannot do it again.
    "accel_n", "depth_med_m", "depth_dropped",
    "n_dr", "src", "rngfnd_m", "alt_amsl_m",
    # The saved image for THIS row, or empty on the rows where no
    # frame was saved (most of them - the gate saves a couple a
    # second out of 17). This column is the whole point of writing
    # frames from inside the navigating process: offline map
    # matching needs heading and height per image, and they are
    # already in this row. An index, not a timestamp, because the
    # Jetson's clock jumps at boot.
    "frame_file",
    # The sigma actually SENT: the origin's own error folded in with the
    # fuser's drift. sigma_m above is the drift alone, and on the three
    # stale-origin flights of 20 Oct it declared 14 m against a true 151 m.
    "sigma_abs_m",
    # THE RAW RECEIVER, added 21 Oct, and the reason matters. gps_north_m /
    # gps_east_m / err_m above are NOT the GPS: they come from
    # GLOBAL_POSITION_INT, which is the EKF's fused position. On every flight so
    # far that was harmless, because they all ran with src = GPS1 and the source
    # gate blocked our injection, so the EKF really was GPS-driven. The moment
    # the aircraft flies ON the VPS, the EKF is driven by what we send and those
    # columns measure us against ourselves.
    #
    # These three are GPS_RAW_INT - the receiver before the EKF touches it - and
    # they are the only independent truth in the file. gps1_err_m is the one to
    # read when judging a VPS flight.
    "gps1_north_m", "gps1_east_m", "gps1_err_m",
    # Position steps refused as physically impossible, cumulative. A flight
    # where this never moves saw no glitches; one where it climbs steadily is
    # the pilot flying faster than STEP_MAX_SPEED_MPS, not a camera fault.
    "n_step_rej",
    # The EKF's own innovation test ratios, and its status bitmask.
    # FS_EKF_THRESH trips on the WORST of the first three; it is 1.2 on this
    # aircraft against a documented 0.8 default, and nothing has ever measured
    # what they actually reach while flying on the VPS.
    "ekf_vel_var", "ekf_pos_h_var", "ekf_comp_var", "ekf_flags"]


# w is only used by the quaternion rows, so it is last and empty elsewhere.
IMU_LOG_COLUMNS = ["t_dev", "src", "x", "y", "z", "w"]


def imu_rows(f):
    """One row per camera IMU SAMPLE, not one a frame.

    The estimate CSV carries `gyro_d_roll` - the gyro integrated across the
    frame interval. That is the right thing for derotation and the wrong thing
    for vibration: a 33 ms integral cannot separate a 0.9 Hz airframe wobble
    from a 47 Hz prop one, and on 20 Oct the pilot's question "how much
    vibration was on the camera IMU" had no answer for exactly that reason.
    So every sample is written at its own device timestamp.

    `src` is "g" for gyro (rad/s), "a" for the accelerometer (m/s2, gravity
    already removed - LINEAR_ACCELERATION), and "q" for the fused orientation
    from GAME_ROTATION_VECTOR, written as the raw quaternion x/y/z/w in the
    IMU's OWN frame. The first two are in the CAMERA frame, so they are
    directly comparable with the FC's own IMU only after the 180 degrees
    between the two is accounted for; the quaternion is left exactly as the
    BNO reported it, because rotating a quaternion is not the same operation
    as rotating a vector and doing it here would bury a second assumption in
    the log. The camera-frame DELTA is in the estimate CSV instead, as
    quat_d_roll/quat_d_pitch, where it sits beside the gyro's and the FC's.
    """
    for s in (f.get("gyro") or ()):
        yield ("%.6f" % s[0], "g",
               "%.6f" % s[1], "%.6f" % s[2], "%.6f" % s[3], "")
    for s in (f.get("accel") or ()):
        yield ("%.5f" % s[0], "a",
               "%.5f" % s[1], "%.5f" % s[2], "%.5f" % s[3], "")
    for s in (f.get("quat") or ()):
        yield ("%.6f" % s[0], "q",
               "%.6f" % s[1], "%.6f" % s[2], "%.6f" % s[3], "%.6f" % s[4])


def csv_row(est, f, v, gN, gE, err):
    """One row, in CSV_COLUMNS order. Must be the same length as CSV_COLUMNS."""
    return [f"{est['t']:.2f}", f"{est['north']:.2f}", f"{est['east']:.2f}",
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
            f"{v.flow_vn:.3f}", f"{v.flow_ve:.3f}",
            f"{v.accel_n:.3f}", f"{v.accel_e:.3f}",
            "" if est.get("fix_lon") is None else f"{est['fix_lon']:.7f}",
            "" if est.get("fix_lat") is None else f"{est['fix_lat']:.7f}",
            "" if est.get("fix_score") is None else f"{est['fix_score']:.4f}",
            "" if est.get("fix_inliers") is None else f"{est['fix_inliers']:d}",
            "" if est.get("fix_alt_m") is None else f"{est['fix_alt_m']:.1f}",
            "" if est.get("step_px") is None else f"{est['step_px']:.2f}",
            est.get("step_action", ""),
            "" if est["fc_d_roll"] is None else f"{est['fc_d_roll']:.4f}",
            "" if est["fc_d_pitch"] is None else f"{est['fc_d_pitch']:.4f}",
            "" if est["gyro_d_roll"] is None else f"{est['gyro_d_roll']:.4f}",
            "" if est["gyro_d_pitch"] is None else f"{est['gyro_d_pitch']:.4f}",
            est["gyro_n"],
            "" if est.get("quat_d_roll") is None else f"{est['quat_d_roll']:.4f}",
            "" if est.get("quat_d_pitch") is None else f"{est['quat_d_pitch']:.4f}",
            est.get("quat_n") or 0,
            est.get("derot_src") or "",
            "" if est.get("derot_dis_roll") is None else f"{est['derot_dis_roll']:.4f}",
            "" if est.get("derot_dis_pitch") is None else f"{est['derot_dis_pitch']:.4f}",
            est["accel_n"],
            "" if est["depth_med_m"] is None else f"{est['depth_med_m']:.2f}",
            est["depth_dropped"], est["n_dr"],
            est.get("src") or "",
            "" if f.get("rngfnd") is None else f"{f['rngfnd']:.2f}",
            "" if f.get("alt_amsl") is None else f"{f['alt_amsl']:.2f}",
            est.get("frame_file") or "",
            f"{est['sigma_abs']:.1f}",
            "" if est.get("g1_north") is None else f"{est['g1_north']:.2f}",
            "" if est.get("g1_east") is None else f"{est['g1_east']:.2f}",
            "" if est.get("g1_err") is None else f"{est['g1_err']:.2f}",
            est.get("n_step_rej", 0),
            "" if f.get("ekf_vel_var") is None else f"{f['ekf_vel_var']:.4f}",
            "" if f.get("ekf_pos_h_var") is None else f"{f['ekf_pos_h_var']:.4f}",
            "" if f.get("ekf_comp_var") is None else f"{f['ekf_comp_var']:.4f}",
            "" if f.get("ekf_flags") is None else int(f["ekf_flags"])]


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


def should_handover(src_val, src_was, armed_was, armed_now,
                   have_origin) -> bool:
    """Is this the frame on which the aircraft is handed over to the VPS?

    Pulled out of the loop so the state machine can be exercised rather than
    read. Every clause earns its place:

      src_val == 1              the switch is now on the VPS
      src_was != 1              and it was not, so this is an EDGE
      src_was is not None       the previous source is KNOWN. On the first
                                frame it is not, and a boot with the switch
                                already on the VPS must not look like a
                                handover - that is a take-off, case one.
      armed_was is True         armed on the PREVIOUS frame. This is what "in
                                flight" means, and it is why flipping to the
                                VPS and arming in the same frame is case one:
                                the arm edge's SCR_USER1/2 has to stand.
      armed_now is True         still armed.
      have_origin               there is an origin to replace.
    """
    return bool(src_val == 1 and src_was != 1 and src_was is not None
                and armed_was is True and armed_now is True and have_origin)


def choose_origin(cli_origin, scr_origin, gps_primary, gps_fix):
    """Where the origin may come from. Returns ((lon, lat), why) or (None, why).

    Precedence, and the middle rule is the pilot's, restated 20 Oct:

      1. --origin             an explicit instruction beats everything
      2. the live receiver    but ONLY when GPS_PRIMARY == 0
      3. SCR_USER1/2          the point typed on the RC

    "arm karte samay agar gps par hai tho waha se lat long update kar lena,
     agar vps par hai tab scr 12 se data lo"

    The receiver and SCR_USER swapped places, and the reason is three flights
    of evidence. SCR_USER kept being left at a point 147-157 m from where the
    aircraft actually stood, because it is typed by hand and the aircraft
    moves. Every VPS position is measured from the origin, so switching to the
    VPS with the origin 150 m out hands the EKF a 150 m step and the aircraft
    flies off to "correct" it. The receiver, when the switch is on the
    receiver, is simply where the aircraft IS.

    This is not a return to the 13 Oct fault, which was the receiver winning
    UNCONDITIONALLY and quietly adopting a stale site. Two things stop that
    here: it wins only while GPS_PRIMARY is 0 - the pilot has the aircraft on
    the receiver and is therefore already trusting it - and the origin is
    re-taken at ARM (see reanchor_on_arm), when the aircraft is standing on
    the ground at the place it is about to fly from.

    On the VPS the receiver is still not an acceptable origin whatever it is
    reporting, and GPS_PRIMARY of None - we could not ask - is still treated as
    "on the VPS". Not knowing must never be the state that permits a
    GPS-derived origin.
    """
    if cli_origin is not None:
        return cli_origin, "--origin"
    if gps_primary == 0 and gps_fix is not None:
        return gps_fix, "the live receiver (GPS_PRIMARY=0)"
    if scr_origin is not None:
        return scr_origin, "SCR_USER1/2 (typed on the RC)"
    if gps_primary == 0:
        return None, "waiting for a receiver fix"
    if gps_primary == 1:
        return None, "the aircraft is on the VPS and SCR_USER1/2 is not set"
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

    # How long a depth median is believed after the last one arrived. The
    # depth queue is read non-blocking at its own rate, so a single missing
    # cycle means nothing; two seconds of silence does.
    DEPTH_SEEN_S = 2.0
    # Above this the stereo pair cannot contribute and the depth gate stands
    # itself down (FlowTracker.DEPTH_MAX_M). Kept here as the altitude the
    # PILOT is told about, which is a different thing from the gate's own
    # threshold and must not silently follow it.
    DEPTH_USELESS_ABOVE_M = 30.0
    # Dead band on that threshold, so climbing through it is not an event.
    DEPTH_HYST_M = 5.0

    def __init__(self, link, period_s: float = 10.0, enabled: bool = True):
        self.link = link                    # anything with .mav.statustext_send
        self.period = float(period_s)
        self._depth_seen_t = None
        self._depth_high = False
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

        # THE DEPTH LINE WAS THE NOISIEST THING ON THE RADIO. On log 229 it
        # sent 154 of the flight's 363 messages, flipping at a median 0.04 s
        # apart - 25 Hz, almost every frame.
        #
        # The event() mechanism was working; the PREDICATE was wrong. It asked
        # "did a depth frame arrive on THIS cycle", and the depth queue is
        # read with tryGet() at its own rate, so that flips as an accident of
        # timing rather than as anything the pilot can act on.
        #
        # It is a HELD state now: has a depth median arrived in the last
        # DEPTH_SEEN_S. And above FlowTracker.DEPTH_MAX_M the gate stands
        # itself down - the OAK-D's 7.5 cm baseline gives under 0.3 px of
        # disparity past about 25 m, which is why log 229 read a median of
        # 5.00 m while the aircraft sat at 32-86 m - so up there it says so
        # instead of calling it a fault.
        dm = est.get("depth_med_m")
        if dm is not None:
            self._depth_seen_t = t
        seen = (self._depth_seen_t is not None
                and (t - self._depth_seen_t) < self.DEPTH_SEEN_S)
        # Hysteresis on the altitude, or climbing through the threshold flaps
        # it: on log 229 the held-state fix alone took 154 messages to 15, and
        # eleven of those fifteen were the aircraft crossing 30 m.
        alt_now = est.get("alt")
        if alt_now is not None and np.isfinite(alt_now):
            if self._depth_high:
                self._depth_high = alt_now > self.DEPTH_USELESS_ABOVE_M - self.DEPTH_HYST_M
            else:
                self._depth_high = alt_now > self.DEPTH_USELESS_ABOVE_M
        too_high = self._depth_high
        if too_high:
            self.event("depth", "high",
                       "VPS STEREO IDLE - above %.0fm, baseline too short"
                       % self.DEPTH_USELESS_ABOVE_M, self.SEV_INFO)
        else:
            self.event("depth", bool(seen),
                       f"VPS STEREO OK {dm:.1f}m" if (seen and dm is not None)
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
                       f"VPS MAP NEEDS {min_alt:.0f}m - now {alt:.0f}m",
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

    # THE POSITION STEP'S OWN CEILING, IN PIXELS.
    #
    # Why this exists. The position is ACCUMULATED - fuser.predict() adds each
    # frame's displacement to a running total - so one bad flow frame does not
    # produce one bad reading, it produces a permanent offset. vps_00016 has a
    # single-sample 20.55 m step at 41.3 m altitude and it is still in that
    # flight's position at the end. The map and the velocity were both already
    # defended: a map fix beyond 3 sigma is dropped and an accepted one may
    # move the estimate by at most max_correction_m = 0.5 m, and the velocity
    # bridge has had a physical gate and an output slew since 20 Oct. The
    # accumulating position had nothing.
    #
    # WHY PIXELS AND NOT m/s, which is what the pilot first asked for ("sabhi
    # mode main 3-4 ms se jada ka speed"). A pixel is altitude / focal metres
    # on the ground, so a fixed SPEED ceiling is a shrinking PIXEL ceiling as
    # the aircraft climbs. 4 m/s works out at:
    #
    #     40 m   3.17 px/frame      90 m   1.41 px/frame
    #     70 m   1.81 px/frame     110 m   1.15 px/frame
    #
    # and the flow tracker's own frame-to-frame noise is several pixels. So at
    # the 80-90 m the next flight plans, a 4 m/s ceiling clips at barely over
    # one pixel - below what the camera can even resolve - and would refuse
    # 38-66% of steps above 20 m on the recordings. In pixels, which is the
    # unit the error is actually constant in, the separation is clean:
    #
    #     normal steps      p50 3.18, p90 13.54, p95 22.39 px
    #     the glitch        394 px in one frame
    #     tracker's reach   31 winSize x 2^4 maxLevel = 496 px
    #
    # 50 px refuses 1.4% of 22738 recorded steps and kills the 394 px glitch.
    #
    # WHAT THIS DOES NOT DO, and it matters: it does not stop noise-driven
    # drift at altitude. At 70-80 m the flow's own error is about 5.7 m/s, and
    # nothing in the position path removes that - the position is an integral
    # of flow. Correcting it is the MAP FIX's job, which is why MAP_MIN_ALT_M
    # is 60 and why a VPS hold at 40 m is dead reckoning with no absolute
    # reference at all. This gate catches glitches, not noise.
    STEP_MAX_PX = 50.0
    STEP_ESCAPE_N = 5

    def __init__(self, cache: MapCache, focal_px: float, *,
                 fix_period_s: float = 2.0, radius_m: float = 150.0,
                 max_corners: int = 300, K=None, D=None, focal_y_px=None,
                 derotate: str = "fc"):
        self.cache = cache
        self.focal = float(focal_px)
        # "auto" or "fc". There is deliberately NO option to force a source
        # that has not been validated - that is exactly the 21 Oct mistake.
        #
        # The gyro was made the default earlier that day on an offline replay
        # that scored |our_speed - truth_speed|. That metric is a SCALAR: it
        # rewards reporting too little exactly as much as reporting correctly,
        # and it never looks at direction at all. The gyro version reported
        # smaller speeds, so it won.
        #
        # Flown, and compared as VECTORS against the raw receiver over the
        # moving samples of each flight:
        #
        #                       north slope   east slope   angle to truth
        #   FC attitude          +0.795        +0.892        33 deg
        #   camera gyro          -0.153        -0.535       160 deg
        #
        # east r = -0.884. The gyro path points the velocity almost exactly
        # backwards, and the aircraft ran away in Loiter because of it: we told
        # the flight controller it was moving one way while it drifted the
        # other, so the controller pushed it further. Log 00000223, two VPS
        # windows, position error growing 0.5 -> 21.4 m and 18.2 -> 43.3 m.
        #
        # The sign derivation behind -14.80/-13.82 was wrong, and the matched-
        # interval measurement it rested on (gyro vs FC, scale -0.99) did not
        # survive contact with a directional test.
        #
        # "auto" now means: use the gyro if TiltValidator has verified it
        # against the FC's attitude, else GAME_ROTATION_VECTOR if that has
        # been verified, else the FC itself. Nothing is used on an assumption,
        # the sign comes from the verification rather than from a constant, and
        # a source that loses agreement is dropped again mid-flight. "fc" pins
        # it to the FC alone, for reproducing an older flight.
        self.derotate = str(derotate)
        # fy is 791.58 against fx 791.72 on this camera - a ratio of 1.000177,
        # which is 1.8 cm per 100 m travelled. It is carried because it is free
        # and correct, NOT because it is a fix for anything.
        self.focal_y = float(focal_px if focal_y_px is None else focal_y_px)
        self.tracker = FlowTracker(max_corners=max_corners, K=K, D=D)
        # fix_sigma_m 20, not 12. This line said 12 while MapFuser's own default,
        # its docstring and tests/test_restored_from_flown_logs.py all said 20 -
        # so the flying code was quietly using a number the evidence does not
        # support. The 9 Oct report settled it with the fix's own lat/lon logged
        # for the first time: "the map's median error is 20.0 m against our own
        # 6.6 m ... the code's fix_sigma_m = 20 was already right."
        #
        # It made no difference to the 20 Oct flights because they took no fixes
        # at all (replayed: sigma p50 11.15 m at 12, 11.00 m at 20), but it
        # decides how far each fix moves the aircraft the moment fixes return,
        # and 12 trusts the map almost three times as much as it has earned.
        self.fuser = MapFuser(flow_drift_frac=0.07, fix_sigma_m=20.0)
        # K's cx/cy go to the worker, so a map fix means the point under the
        # camera rather than the point under the middle of the image. The two
        # are 80 px apart on the leg-cropped frame = 11 m at 110 m, and
        # FlowTracker has always used cx/cy, so this is also what stops the
        # map path and the flow path disagreeing about where the aircraft is.
        self.worker = MapFixWorker(
            cache, focal_px=focal_px, radius_m=radius_m,
            principal_xy=(None if K is None
                          else (float(np.asarray(K).reshape(3, 3)[0, 2]),
                                float(np.asarray(K).reshape(3, 3)[1, 2]))))
        self.fix_period = float(fix_period_s)
        self.origin = None            # (lon, lat)
        self.origin_sigma_m = None    # how well that point is known, metres
        self.mlon = None
        self._next_fix = 0.0
        self.prev_att = None
        self.prev_t = None
        self.n_fix = self.n_flow = self.n_dr = 0
        # Steps refused as physically impossible, and how many in a row - the
        # second is what lets the gate stand down. See step_is_possible.
        self.n_step_rejected = 0
        self._step_refusals = 0
        # Which source actually derotated each frame, so a flight says so
        # rather than being assumed.
        self.n_derot_gyro = self.n_derot_quat = self.n_derot_fc = 0
        # Two candidates, both judged against the FC's attitude, which is the
        # only tilt source measured to track the real motion on this airframe.
        # Neither is used until it has earned it - see TiltValidator.
        self.val_gyro = TiltValidator("gyro")
        self.val_quat = TiltValidator("GAME_ROTATION_VECTOR")
        # How far the fused orientation and the gyro disagreed on the frames
        # where both were present. Logged per frame, not just counted.
        self.n_derot_both = 0
        self.derot_dis_roll = self.derot_dis_pitch = None
        # Flow AND acceleration on this path - step() pushes both. The bridge
        # still holds rather than dead-reckoning when no acceleration arrives,
        # which is what a replayed recording and a silent IMU both look like,
        # and says how stale it is either way.
        self.vel = VelocityBridge()
        # THE FLOW VELOCITY'S BASELINE. See VpsConfig.flow_vel_frames: one
        # frame was the worst choice available, because a pixel of jitter is
        # 3.03 m/s at 80 m and 30 fps. These hold the running sum of the
        # displacements the fuser accepted, so the velocity is measured over
        # the span of the last N of them and still comes from exactly what the
        # position came from.
        from config import VpsConfig as _Cfg
        _c = _Cfg()
        self._vel_frames = int(_c.flow_vel_frames)
        self._vel_frames_high = int(_c.flow_vel_frames_high)
        self._vel_alt_m = float(_c.flow_vel_alt_m)
        # The ramp's width. A step from one baseline to the other at a single
        # height would put a discontinuity in the reported velocity exactly
        # where the aircraft is climbing through it, so it is interpolated -
        # which is also what the archived pre-SSD engine did, though it ramped
        # the other way.
        self._vel_alt_span_m = 15.0
        self._vel_cum_n = self._vel_cum_e = 0.0
        self.step_px = None
        self.step_action = "ok"
        # Held long enough for the LONGEST baseline; how far back is actually
        # read is chosen per frame from the height.
        self._vel_hist = collections.deque(
            maxlen=max(self._vel_frames, self._vel_frames_high) + 1)

    def _vel_baseline_frames(self, altitude_m) -> int:
        """How many frames the velocity is measured over, at this height.

        See VpsConfig.flow_vel_frames: below flow_vel_alt_m a longer baseline
        stops helping and starts averaging real motion away, above it the
        error keeps falling with length.
        """
        lo, hi = self._vel_frames, self._vel_frames_high
        if altitude_m is None or not math.isfinite(float(altitude_m)):
            return lo
        a = float(altitude_m)
        if a <= self._vel_alt_m:
            return lo
        if a >= self._vel_alt_m + self._vel_alt_span_m:
            return hi
        f = (a - self._vel_alt_m) / self._vel_alt_span_m
        return int(round(lo + f * (hi - lo)))

    def set_origin(self, lon: float, lat: float, sigma_m: float | None = None):
        """Place the origin, and record how well that point is known.

        sigma_m is the ORIGIN'S OWN ERROR, and carrying it is the whole point
        of the argument. Everything this node reports is origin + displacement,
        and until 20 Oct the accuracy declared to the flight controller was the
        displacement's alone. Measured on the three VPS flights of 20 Oct, with
        the origin left at a stale SCR_USER1/2:

            declared sigma   p50 14.00 m    p95 22.30 m
            TRUE error       p50 151.11 m   p95 154.53 m

        and the true error exceeded the declared sigma in 76.3% of samples. Take
        the constant offset out and the drift is covered properly - p95 16.11 m
        against a declared 14.10, exceeded 8.9% of the time where an honest
        one-sigma claim is exceeded about 32%. So the shape was never wrong. The
        origin's error was simply absent, and it was the entire 151 m.

        None means unknown, which is not the same as zero: a point typed into
        SCR_USER1/2 by hand has no stated accuracy at all, and that is exactly
        the case that was 147-157 m out. absolute_sigma_m falls back to the
        map's own error there, because nothing better has measured it.
        """
        self.origin = (float(lon), float(lat))
        self.origin_sigma_m = None if sigma_m is None else float(sigma_m)
        self.mlon = M_PER_DEG_LAT * float(np.cos(np.radians(lat)))

    @property
    def absolute_sigma_m(self) -> float:
        """What to declare to the flight controller: origin error AND drift.

        The fuser's sigma is the uncertainty of the DISPLACEMENT - it starts at
        init_sigma_m and grows with distance travelled. The absolute position is
        origin + displacement, so the two uncertainties add in quadrature and
        the result can never be better than the origin.

        When the origin's accuracy is unknown the map's own error stands in. It
        is the best this system can ever know absolutely, so declaring better
        than it with nothing measured is a claim nothing supports. Log 00000194,
        which flew, declared 21.36 m p50 - the same order as this floor.
        """
        drift = float(self.fuser.sigma_m)
        origin = self.origin_sigma_m
        if origin is None:
            origin = float(np.sqrt(self.fuser.fix_var))
        return float(np.hypot(origin, drift))

    def gate_step(self, dn: float, de: float, dt: float | None,
                  altitude_m: float):
        """Bound one frame's displacement. Returns (dn, de, action).

        action is "ok" for a step to use as measured, "refused" for one to drop
        (the caller then treats the frame as a GAP), or "clipped" for one the
        escape hatch let through at the ceiling.

        WHY THE ESCAPE HATCH CLIPS instead of passing the raw step. A gate that
        can refuse for ever is worse than the glitch it catches - if the
        estimate itself has gone wrong, every correct measurement looks
        impossible and the error is locked in - so after STEP_ESCAPE_N
        consecutive refusals it has to concede. Passing the raw step at that
        point is the wrong concession, and the aircraft's own recordings show
        why: at 20-80 m altitude 38-66% of steps exceed the ceiling, because
        the flow's error is constant in PIXELS and becomes metres times
        altitude. Those are not glitches, they are noise, and under sustained
        noise an escape hatch that passes raw steps admits the LARGEST one
        every sixth frame - systematically the worst of them.

        Clipping concedes the direction and keeps the bound: sustained real
        motion still moves the position, at up to STEP_MAX_SPEED_MPS, while
        sustained noise contributes at most that instead of the 95 m/s seen in
        vps_00016. Nothing above the ceiling ever enters the position, which is
        the pilot's instruction taken literally: "sabhi mode main 3-4 ms se
        jada ka speed".

        The cost is honest and it is the pilot's own stick. Flying faster than
        the ceiling in AltHold or Stabilize makes the position under-report,
        and that accumulates. LOIT_SPEED_MS is 3 m/s and ATC_ANGLE_MAX is 15
        degrees, so nothing the position controller can command comes near it,
        and n_step_rej is in the CSV so a flight shows how often it bit.

        dt None, or outside a sane frame interval, means there is nothing to
        judge the step against, so it is used as measured. Refusing on no
        evidence would throw away the first flowed frame of every run.
        """
        self.step_px = None
        self.step_action = "ok"
        if dt is None or not (0.0 < dt < 0.5) or not (altitude_m > 0.0):
            return dn, de, "ok"
        mag = float(np.hypot(dn, de))
        # The step as the gate SAW it, before it was refused or clipped, in
        # pixels - which is the unit the ceiling is in and the only unit the
        # flow's error is constant in. Added 22 Oct: a refused step enters the
        # position as zero and a clipped one at exactly the ceiling, so after
        # the fact the log cannot tell a justified refusal from a lost real
        # step. Measured post-gate the accepted steps run p50 3.4-4.9 px and
        # p99 32-40 against this 50 px ceiling, and at 85 m a real 3 m/s is
        # 0.93 px per frame - so whether the top few percent were noise or
        # motion has never been checked.
        self.step_px = mag * self.focal / altitude_m
        # The ceiling in metres for THIS frame: pixels x metres-per-pixel.
        ceiling = self.STEP_MAX_PX * (altitude_m / self.focal)
        if mag <= ceiling:
            self._step_refusals = 0
            self.step_action = "ok"
            return dn, de, "ok"
        self._step_refusals += 1
        self.n_step_rejected += 1
        if self._step_refusals > self.STEP_ESCAPE_N:
            # Concede the direction, keep the bound. Re-armed so the next
            # isolated glitch still dies rather than the gate standing open.
            self._step_refusals = 0
            k = ceiling / mag
            self.step_action = "clipped"
            return dn * k, de * k, "clipped"
        self.step_action = "refused"
        return 0.0, 0.0, "refused"

    def to_lonlat(self, north: float, east: float):
        return (self.origin[0] + east / self.mlon,
                self.origin[1] + north / M_PER_DEG_LAT)

    def step(self, f: dict) -> dict | None:
        """One frame in, one position estimate out."""
        if f["alt"] is None or f["yaw"] is None or self.origin is None:
            return None
        # Counters as they stood on entry, so the row can name the source that
        # actually ran on THIS frame rather than the one that ran most often.
        _dg, _dq, _df = self.n_derot_gyro, self.n_derot_quat, self.n_derot_fc
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
            # DEROTATION. The camera's own gyro by default, the FC's
            # attitude only as a fallback and only because something has to
            # happen when the gyro reports nothing for a frame.
            #
            # The FC's attitude arrives at ~4 Hz against this 30 Hz loop, so on
            # the 21 Oct flight it said "no rotation" on 88% of frames and
            # several frames' worth on the rest - which then got subtracted
            # from one frame's shift. Measured on that flight, per-frame speed
            # error against the real receiver:
            #
            #     no derotation at all      p50 2.50  p95 24.01 m/s
            #     the FC's attitude         p50 2.97  p95 31.05
            #     the gyro, sign corrected  p50 1.15  p95  7.54
            #
            # The FC path is WORSE than doing nothing. Through the whole
            # velocity path the same change takes the sent error from p50 0.725
            # / p95 3.576 to p50 0.155 / p95 1.243, and samples over 5 m/s from
            # 1.9% to none - and it was the velocity that tripped XKF4.SV to
            # 7.95 and 10.92 against a 1.2 gate and dropped the aircraft into
            # ALT_HOLD on both VPS handovers.
            #
            # This is a RESTORATION, not an experiment. Gyro derotation was
            # built, flown and carried at 95% from 28 Sep until the NVMe went,
            # and 30 Sep names it "the real cure". The rebuild lost it.
            # DEROTATION, and which source runs is EARNED, not configured.
            #
            # On 21 Oct the gyro was made the default on a static measurement
            # and its sign hard-coded. Flown, the reported velocity came out
            # 160 degrees from the truth and the aircraft ran away in Loiter.
            # The pre-SSD-loss daemon never had that hole: it compared 0.4 s
            # windows of its candidate against a reference, derived the sign
            # from the comparison, and switched the candidate off again the
            # moment agreement was lost.
            #
            # The reference is the FC's attitude. It is poor per FRAME - ~4 Hz
            # against a 30 Hz loop, so it reads 0.000 on most of them - but it
            # is the only source measured to track: velocity slope +0.795 north
            # and +0.892 east against the raw receiver, 33 degrees from the
            # truth. Over a 0.4 s window it has published once or twice, which
            # is why the windows are that long.
            #
            # Order of preference once earned: gyro (100 Hz, every frame),
            # then GAME_ROTATION_VECTOR, then the FC itself.
            fc_r, fc_p = fc_dr, fc_dp
            gd_r, gd_p = f.get("gyro_d_roll"), f.get("gyro_d_pitch")
            qd_r, qd_p = f.get("quat_d_roll"), f.get("quat_d_pitch")
            dt_v = (None if self.prev_t is None else f["t"] - self.prev_t)
            if dt_v is not None and 0.0 < dt_v < 0.5:
                self.val_gyro.update(gd_r, gd_p, fc_r, fc_p, dt_v)
                self.val_quat.update(qd_r, qd_p, fc_r, fc_p, dt_v)
            if (self.derotate != "fc" and self.val_gyro.ok
                    and gd_r is not None and gd_p is not None):
                d = d - gyro_shift_px(gd_r, gd_p,
                                      self.val_gyro.sign_roll,
                                      self.val_gyro.sign_pitch)
                self.n_derot_gyro += 1
            elif (self.derotate != "fc" and self.val_quat.ok
                  and qd_r is not None and qd_p is not None):
                d = d - gyro_shift_px(qd_r, qd_p,
                                      self.val_quat.sign_roll,
                                      self.val_quat.sign_pitch)
                self.n_derot_quat += 1
            else:
                d = d - attitude_shift_px(self.prev_att, att)
                self.n_derot_fc += 1
            # What the two candidates disagreed with each other by, carried out
            # so a flight measures it rather than a comment asserting it.
            if None not in (gd_r, gd_p, qd_r, qd_p):
                self.derot_dis_roll = float(qd_r) - float(gd_r)
                self.derot_dis_pitch = float(qd_p) - float(gd_p)
                self.n_derot_both += 1
            else:
                self.derot_dis_roll = self.derot_dis_pitch = None
            dn, de = shift_to_ne(d, altitude_m=f["alt"], focal_px=self.focal,
                                 yaw_deg=f["yaw"], focal_y_px=self.focal_y)
            # ONE decision for both consumers. Until 21 Oct predict() ran here
            # unconditionally and push_flow() below could then refuse the same
            # displacement on its own physical gate - so the position swallowed
            # steps the velocity had rejected, and the comment below claiming
            # they "cannot disagree about what the camera saw" was not true.
            # Asking once fixes both.
            dt_step = (None if self.prev_t is None else f["t"] - self.prev_t)
            dn, de, action = self.gate_step(dn, de, dt_step, f["alt"])
            if action != "refused":
                self.fuser.predict(dn, de, dt=dt_step)
                self.n_flow += 1
                # Velocity is the SAME displacement the fuser just consumed,
                # divided by the frame interval. Deriving it here rather than
                # in a second estimator means position and velocity cannot
                # disagree about what the camera saw - and disagreement
                # between them is what the EKF reads as a velocity innovation.
                if dt_step is not None and 0.0 < dt_step < 0.5:
                    # Accumulate, then measure over the longest span held.
                    # The deque grows from empty, so the baseline lengthens
                    # from one frame to flow_vel_frames over the first ~0.7 s
                    # instead of there being no velocity at all until it
                    # fills - an aircraft that has just switched to the VPS
                    # cannot wait for that.
                    if not self._vel_hist:
                        # Seed with the CUMULATIVE sum as it stands, not with
                        # zero. Zero is only right on the very first frame of
                        # a run; after a mid-flight gap clears the history the
                        # running total is already large, and measuring it
                        # against a zero endpoint over one frame's span read
                        # 82 m/s in the test that caught this.
                        self._vel_hist.append(
                            (f["t"] - dt_step, self._vel_cum_n,
                             self._vel_cum_e))
                    self._vel_cum_n += dn
                    self._vel_cum_e += de
                    self._vel_hist.append(
                        (f["t"], self._vel_cum_n, self._vel_cum_e))
                    # How far back to read, from the height. The deque holds
                    # the longest baseline; a shorter one just reads nearer
                    # the end, and while it is still filling we use whatever
                    # is there rather than having no velocity at all.
                    want = self._vel_baseline_frames(f.get("alt"))
                    back = min(want, len(self._vel_hist) - 1)
                    t0, n0, e0 = self._vel_hist[-(back + 1)]
                    span = f["t"] - t0
                    if span > 0.0:
                        self.vel.push_flow(f["t"],
                                           (self._vel_cum_n - n0) / span,
                                           (self._vel_cum_e - e0) / span)
                flowed = True
            # else: flowed stays False, so the gap handling below carries the
            # position on the accelerometer and grows the variance DR_TRUST
            # times faster. A refused frame IS a lost frame.

        if not flowed:
            # THE BASELINE BREAKS ON A GAP, and it has to. The velocity is a
            # sum of displacements divided by its own span; if a frame is
            # refused or tracking is lost, the time passes but the
            # displacement is missing, so a span that straddles the gap
            # reports less motion than there was - the 8 Oct failure mode
            # exactly, a fraction of truth that nothing aboard can see.
            # Restarting costs a few noisy frames while it refills, which is
            # the honest trade. Log 226 refused 8 frames in 10406.
            self._vel_hist.clear()

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
                self.fuser.predict(v.vn * dt, v.ve * dt, trust=self.DR_TRUST,
                                   dt=dt)
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
        fix_lon = fix_lat = fix_score = fix_alt = None
        fix_inl = None
        if res is not None and res.ok:
            fN = (res.lat - self.origin[1]) * M_PER_DEG_LAT
            fE = (res.lon - self.origin[0]) * self.mlon
            fix_used = self.fuser.update(fN, fE)
            self.n_fix += 1
            # Where the MATCHER put us, before the fuser weighed and capped
            # it, and the height it was matched at. This is the raw number the
            # matcher's error-by-height has to be measured from; the fused
            # position cannot show it because max_correction_m caps every fix
            # at 0.5 m, so a fix 40 m out and a fix 2 m out move the estimate
            # by the same amount and leave the same trace.
            fix_lon, fix_lat = float(res.lon), float(res.lat)
            fix_score = float(getattr(res, "score", 0.0))
            fix_inl = int(getattr(res, "inliers", 0))
            fix_alt = float(getattr(res, "altitude_m", 0.0)) or None

        lon, lat = self.to_lonlat(*self.fuser.pos)
        return dict(step_px=self.step_px, step_action=self.step_action,
                    fix_lon=fix_lon, fix_lat=fix_lat, fix_score=fix_score,
                    fix_inliers=fix_inl, fix_alt_m=fix_alt,
                    north=float(self.fuser.pos[0]), east=float(self.fuser.pos[1]),
                    lon=lon, lat=lat, sigma=self.fuser.sigma_m,
                    # Two sigmas, and the difference is the whole 20 Oct
                    # finding. `sigma` is the fuser's own - the uncertainty of
                    # the DISPLACEMENT this node has measured. `sigma_abs` adds
                    # the origin's error, which is what makes it the uncertainty
                    # of the POSITION, and that is the one the flight controller
                    # is told. Both are logged so a flight can be audited.
                    sigma_abs=self.absolute_sigma_m, tracks=ntrk,
                    fix_used=fix_used, alt=f["alt"], yaw=f["yaw"], t=f["t"],
                    vel=self.vel.value(f["t"], f["alt"]),
                    fc_d_roll=fc_dr, fc_d_pitch=fc_dp,
                    gyro_d_roll=f.get("gyro_d_roll"),
                    gyro_d_pitch=f.get("gyro_d_pitch"),
                    quat_d_roll=f.get("quat_d_roll"),
                    quat_d_pitch=f.get("quat_d_pitch"),
                    quat_n=f.get("quat_n"),
                    derot_src=("gyro" if self.n_derot_gyro > _dg
                               else "quat" if self.n_derot_quat > _dq
                               else "fc" if self.n_derot_fc > _df else ""),
                    derot_dis_roll=self.derot_dis_roll,
                    derot_dis_pitch=self.derot_dis_pitch,
                    gyro_n=f.get("gyro_n"), accel_n=f.get("accel_n"),
                    n_step_rej=self.n_step_rejected,
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
    # 2.0 s, which is the 5 Oct report's value and MapFixWorker's own default.
    #
    # It had drifted to 0.5 here, and the reasoning written beside it was about
    # CPU: "1.0 until 20 Oct, when XFeat went from 1233 ms to 272 ms by being
    # given ONE torch thread instead of six - it could not previously finish
    # inside its own period", plus the observation that 0.5 s paired with the
    # 0.5 m correction cap gives the map 1 m/s of authority.
    #
    # Both of those are true and neither is why 2 s was chosen. 5 Oct set it
    # for a different reason entirely, and the number it replaced was 0.4:
    #
    #   "The map was treated as the thing that breaks the hold. It is, above
    #    80 m - 1.18 m/s of push on a stationary aircraft - but it is also the
    #    only thing that bounds drift over distance... It now runs every 2 s
    #    instead of every 0.4, which keeps the bound and removes most of the
    #    shove."
    #
    # So the fast period was measured to SHOVE a stationary aircraft, and 0.5
    # sits beside the 0.4 that was abandoned, not beside the 2.0 that replaced
    # it. Giving the map 1 m/s of authority is the shove, described approvingly.
    #
    # This matters now rather than academically: on 21 Oct the matcher was
    # measured working at 90-100 m for the first time - 8.3 m median error,
    # 228 of 477 frames clearing min_inliers - so fixes are about to start
    # arriving, and the period decides what they do when they do.
    ap.add_argument("--imu-rate", type=int, default=None,
                    help="camera IMU report rate in Hz for all three sensors "
                         "(default from VpsConfig.imu_rate, 100). They share "
                         "one packet, so this sets them together.")
    ap.add_argument("--fix-period", type=float, default=2.0,
                    help="seconds between map fixes. 1.0 was tried on 22 Oct "
                         "because the pilot's archived 11 Oct reports record "
                         "'0.500 m per fix at 1 fix/s' - later evidence than "
                         "the 5 Oct value of 2.0. Flown as log 227 and "
                         "reverted the same day: at 82-83 m, XKF4.SP max went "
                         "0.13 (log 226, 2.0 s) to 1.17 (log 227, 1.0 s), "
                         "nine times worse against a 1.2 gate. A map fix "
                         "moves the POSITION by up to max_correction_m and "
                         "the velocity not at all, so at 1 Hz the EKF is told "
                         "the aircraft jumped without moving. 5 Oct's caution "
                         "- 1.18 m/s of push on a stationary aircraft - was "
                         "the same fault seen from the outside.")
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
    # 3 s, not 10. The pilot confirmed on 21 Oct that companion STATUSTEXT does
    # reach the handset - closing the 13 Oct open item - but called it "thora
    # slow". Measured on the link that evening: exactly 10.02 s between lines,
    # 0.10 lines a second, which is this default and nothing else.
    #
    # WARNINGS WERE NEVER SLOW, and that is worth separating. Notifier.event()
    # fires the moment a value CHANGES, so LIDAR ZERO, FRAMES ON/OFF, the
    # origin line and VPS HANDOVER all go out at once. This period governs only
    # the periodic status line - tracks, sigma, fixes, source.
    #
    # The cost is bandwidth on the radio, and it is small: a STATUSTEXT is 50
    # bytes of payload plus about 12 of MAVLink 2 framing, so 10 s is 6 B/s and
    # 3 s is 21 B/s. A SiK radio at a 57600 air rate carries some 5-6 kB/s
    # usable, so this is well under half a percent either way. Raise it with
    # the flag if the link is ever the thing that is short.
    ap.add_argument("--notify-period", type=float, default=3.0,
                    help="seconds between VPS status lines on the radio "
                         "(warnings are immediate regardless; default 3)")
    # DEFAULT BACK TO "fc" ON 21 OCT, AFTER LOG 00000224.
    #
    # TiltValidator did what it was built to do - it compared the gyro's
    # rotation against the FC's over 0.4 s windows, found roll slope -0.84 corr
    # -0.80 and pitch -0.89/-0.79, adopted the gyro at t=39.7 s with sign -1 on
    # both axes - and the aircraft still ran away. Our reported motion came out
    # 176 degrees from the truth in the first VPS window and the error grew
    # 1.1 -> 22.3 m in ten seconds.
    #
    # So the premise was wrong, not the implementation. Validating that a
    # source measures the same ROTATION as the FC does not establish that
    # derotating with it produces the right TRANSLATION, and translation is the
    # only thing that matters here. "auto" stays reachable for offline work; it
    # must be scored on the direction of the resulting velocity before it is
    # ever the default again.
    ap.add_argument("--derotate", choices=("auto", "fc"), default="fc",
                    help="where the tilt to remove from the flow comes from. "
                         "'auto' uses the camera gyro, or GAME_ROTATION_VECTOR, "
                         "ONLY once that source has been verified against the "
                         "FC's attitude IN FLIGHT - matching sign and slope "
                         "over 0.4 s windows - and drops back the moment "
                         "agreement is lost. 'fc' pins it to the FC alone. "
                         "There is no option to force an unverified source: "
                         "that is what ran the aircraft away on 21 Oct")
    ap.add_argument("--no-scr-origin", action="store_true",
                    help="do not take the origin from SCR_USER1/2")
    ap.add_argument("--out", help="estimate log CSV, or a DIRECTORY - a "
                                  "directory gets a new numbered file per run, "
                                  "the way the flight controller numbers its own")
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
    ap.add_argument("--save-frames", type=float, default=0.0, metavar="HZ",
                    help="also save camera frames, this many a second, into "
                         "the run's own folder beside the estimate CSV. OFF "
                         "by default. The OAK-D permits one owner, so "
                         "record_flight.py cannot run while this node is "
                         "flying and the images the pilot wants for offline "
                         "map matching were never recorded; this is how they "
                         "get recorded. Encoding is on a worker thread behind "
                         "a bounded queue, so a disk that falls behind drops "
                         "frames and says so rather than stalling the flow "
                         "loop. Needs --out to be a DIRECTORY. Each saved "
                         "frame's name goes in that row's frame_file column.")
    ap.add_argument("--frames-any-time", action="store_true",
                    help="save frames whether or not the aircraft is armed. "
                         "OFF by default: the pilot's rule is that images "
                         "start at arm and stop at disarm, because the "
                         "services run for as long as the Jetson is powered "
                         "and the hour on the ground is 29 MB a minute of "
                         "pictures of the same patch of ground. Use this for "
                         "bench work, where nothing ever arms.")
    ap.add_argument("--imu-log", dest="imu_log", action="store_true",
                    default=True,
                    help="write every camera IMU sample to <run>_imu.csv "
                         "beside the estimate log (default ON). This is the "
                         "only way to measure what the CAMERA is doing rather "
                         "than what the airframe is doing: on 20 Oct the "
                         "question 'how much vibration on the camera IMU' had "
                         "no answer, because only the per-frame integral had "
                         "ever been recorded.")
    ap.add_argument("--no-imu-log", dest="imu_log", action="store_false",
                    help="do not write the camera IMU log.")
    ap.add_argument("--sync-every", type=float, default=1.0, metavar="SEC",
                    help="flush and fsync the logs this often. 0 disables. "
                         "Pulling the battery on 20 Oct left NUL bytes at the "
                         "tail of two runs: the rows were in the kernel's "
                         "page cache and the power went before they reached "
                         "the disk. Measured on this machine: 0.64 ms median, "
                         "2.8 ms worst idle and 14.1 ms worst under load, "
                         "once a second against a 59 ms frame budget - so the "
                         "amortised cost is 0.025 ms a frame and the worst "
                         "single frame still fits.")
    ap.add_argument("--jpeg-quality", type=int, default=90,
                    help="quality of the saved frames. JPEG, not PNG: 9.5 ms "
                         "a frame against 30 (measured 20 Oct), and that CPU "
                         "is shared with flow and XFeat.")
    ap.add_argument("--frame-queue", type=int, default=8, metavar="N",
                    help="frames allowed to be waiting on the encoder. Small "
                         "on purpose - a longer queue does not rescue a slow "
                         "disk, it only delays noticing, and each queued "
                         "frame is ~0.9 MB held away from the map cache.")
    a = ap.parse_args()

    if a.save_frames > 0:
        # A directory that does not exist yet is still a directory: make it,
        # so this works on a machine where flight_logs/vps has never been
        # written rather than refusing a path that is plainly a folder.
        if a.out and not os.path.exists(a.out) and not a.out.endswith(".csv"):
            os.makedirs(a.out, exist_ok=True)
        if not (a.out and (os.path.isdir(a.out) or a.out.endswith(os.sep))):
            ap.error("--save-frames needs --out to be a DIRECTORY: the images "
                     "and the estimate CSV go in one numbered folder per run, "
                     "so an image can be matched back to the row that holds "
                     "its heading and height.")

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
        print(f"GPS INPUT SENDER ON - the estimate will arrive as GPS{a.gps_id+1}.\n"
              f"   (horiz_accuracy carries the fuser's own sigma)")

    sender = None
    if a.send_fc:
        sender = FcSender(a.fc_port, a.fc_baud)
        print(f"FC SENDER ON - VISION_POSITION_ESTIMATE will be sent.\n"
              f"   (EK3 sources untouched - the FC will read it but not use it)")

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
           else LiveSource(a.fc_port, a.fc_baud, crop_left=a.crop_left,
                           imu_rate=a.imu_rate))
    cache = MapCache(a.map)
    print(cache.summary())
    print(f"focal {src.focal:.1f} px, fix every {a.fix_period:.1f} s, window +-{a.radius:.0f} m, "
          f"{a.corners} corners\n", flush=True)

    node = VpsNode(cache, src.focal,
                   K=getattr(src, "K", None), D=getattr(src, "D", None),
                   focal_y_px=getattr(src, "focal_y", None),
                   fix_period_s=a.fix_period,
                   radius_m=a.radius, max_corners=a.corners,
                   derotate=a.derotate)

    wr = None
    fwriter = None
    fh = None
    imu_fh = imu_wr = None
    if a.out:
        out_path = a.out
        if os.path.isdir(a.out) or a.out.endswith(os.sep):
            # A numbered file, not a timestamped one - see next_run_number for
            # why the clock cannot be trusted to name a run on this machine.
            os.makedirs(a.out, exist_ok=True)
            stem = f"vps_{next_run_number(a.out):05d}"
            if a.save_frames > 0:
                # With frames, the run gets a FOLDER of its own and the CSV
                # goes inside it, so one flight is one directory holding its
                # images and the rows that describe them. Without frames
                # nothing changes: the CSV sits in a.out exactly as before,
                # which is what the systemd unit has been writing all along.
                run_dir = os.path.join(a.out, stem)
                os.makedirs(run_dir, exist_ok=True)
                out_path = os.path.join(run_dir, stem + ".csv")
            else:
                out_path = os.path.join(a.out, stem + ".csv")
        fh = open(out_path, "w", newline="")
        print(f"recording to {out_path}")
        wr = csv.writer(fh)
        # The velocity columns are not optional padding. The 8 Oct collapse -
        # 9.93 m/s in, 3.90 out - could not be diagnosed from any recording
        # because the state that caused it was never written down, and five
        # columns had to be added after the fact. Everything the bridge knows
        # goes in the row.
        wr.writerow(CSV_COLUMNS)

        # The camera IMU at its own rate, beside the estimate log. 20 Oct's
        # flight could not answer "how much was the CAMERA shaking" because
        # only the per-frame integral of the gyro had ever been recorded, and
        # an integral over 33 ms cannot tell a 0.9 Hz airframe wobble from a
        # 47 Hz prop one. One row per sample, written inline: 100 rows a
        # second cost 0.007 ms each (measured), which is 0.7 ms a second
        # against a loop that has 59 ms a frame.
        if a.imu_log:
            imu_path = out_path[:-4] + "_imu.csv"
            imu_fh = open(imu_path, "w", newline="")
            imu_wr = csv.writer(imu_fh)
            imu_wr.writerow(IMU_LOG_COLUMNS)
            print(f"camera IMU log: {imu_path}  (gyro rad/s and accel m/s2, "
                  f"camera frame, device timestamps)")

        if a.save_frames > 0:
            fwriter = FrameWriter(os.path.dirname(os.path.abspath(out_path)),
                                  hz=a.save_frames, quality=a.jpeg_quality,
                                  maxsize=a.frame_queue)
            print(f"saving frames at {a.save_frames:g} Hz into "
                  f"{fwriter.folder} (queue {a.frame_queue}; a frame is "
                  f"dropped, never waited for)")
            # The intrinsics the frames were taken with, in the same shape
            # ReplaySource already reads, so the folder can be replayed and so
            # offline map matching is not left guessing at fx or at where cx
            # went. cx moves with the crop and the saved frames ARE cropped -
            # they are the same pixels flow ran on, leg removed.
            try:
                K = getattr(src, "K", None)
                with open(os.path.join(fwriter.folder, "camera.txt"), "w") as cf:
                    if K is not None:
                        cf.write(f"fx={K[0][0]:.4f} fy={K[1][1]:.4f} "
                                 f"cx={K[0][2]:.4f} cy={K[1][2]:.4f} "
                                 f"crop_left={getattr(src, 'crop_left', 0)}\n")
                    else:
                        cf.write(f"fx={src.focal:.4f}\n")
                    D = getattr(src, "D", None)
                    if D is not None:
                        cf.write("D=" + " ".join(f"{x:.6f}" for x in D) + "\n")
            except Exception as e:
                print(f"  (could not write camera.txt: {e})")

    # Reuse a sender's link if one exists. If none does - a read-only ground
    # run - open a link only through the ROUTER. A second opener of the serial
    # device is precisely the contention the router exists to remove, and
    # The notifier reuses the sender's connection rather than opening its own.
    # Two reasons: the router registers UDP clients by address, so every extra
    # socket is another client it has to keep and feed; and a connection that
    # only ever receives gets dropped after 30 s, which is what left the RC
    # switch deaf. Reusing a link that is already transmitting avoids both.
    #
    # LiveSource's Telemetry is deliberately not used here. It now sends a
    # heartbeat to stay registered, but it still sends nothing that commands
    # the aircraft, and STATUSTEXT from it would break that.
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
        print(f"notifications ON: status line every {a.notify_period:.1f}s, "
              f"warnings immediate")

    # The link the watcher listens on is the sender's, because that is the one
    # that owns the FC endpoint. With no sender there is nothing to gate, so
    # the notification link will do.
    watcher = SourceWatcher(getattr(gps_sender, "m", None) or _link)
    amsl_ref = AmslRef(a.site_alt)

    _origin_refused = [False]
    # The armed flag as of the last frame, so the disarmed -> armed EDGE can be
    # seen. None until the origin is first taken.
    _armed_was = [None]
    # GPS_PRIMARY as of the last frame, so the moment the pilot hands the
    # aircraft over to the VPS in flight can be seen. See the handover block.
    _src_was = [None]
    errs, t_start, n = [], time.perf_counter(), 0
    last_sync = 0.0
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
                                       "VPS NO ORIGIN - SET SCR_USER1/2",
                                       Notifier.SEV_CRIT)
                    continue
                # Only the live receiver states its own accuracy. --origin and
                # SCR_USER1/2 are both typed, so they get None and
                # absolute_sigma_m falls back to the map's own error.
                boot_sigma = (f.get("gps1_hacc")
                              if why.startswith("the live receiver") else None)
                node.set_origin(*chosen, sigma_m=boot_sigma)
                _armed_was[0] = f.get("armed")
                olon, olat = node.origin
                covered = cache.covers(olon, olat)
                print(f"origin from {why}: {olat:.6f}, {olon:.6f}")
                print(f"  this point is known to "
                      f"{'no stated accuracy' if boot_sigma is None else f'{boot_sigma:.1f} m'}"
                      f", so the FC is told {node.absolute_sigma_m:.1f} m at the start")
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
                "VPS ON - injecting as GPS2" if watcher.value == 1
                else "GPS1 ON - VPS data blocked" if watcher.value == 0
                else "SOURCE UNKNOWN - cannot read GPS_PRIMARY",
                Notifier.SEV_INFO if watcher.value == 1 else Notifier.SEV_CRIT)

            # The lidar's ZERO, announced on the radio while still on the
            # ground. ArduPilot will not declare the aircraft landed unless
            # the rangefinder reads under 2 m in absolute terms, and this
            # sensor's zero wanders by one to two metres between power cycles
            # (0.82, 2.53 and 0.96 m on the same ground on 20 Oct). On the
            # 20 Oct flight it sat at 2.53 m, the land detector never fired,
            # DISARM_DELAY never ran and the battery had to be pulled. This
            # changes nothing and blocks nothing - it only says so, before
            # take-off, while there is still time to run lidar_zero.py.
            rng = f.get("rngfnd")
            if rng is not None and f.get("armed") is False:
                notifier.event(
                    "lidarzero", rng < 1.5,
                    f"VPS LIDAR ZERO OK {rng:.2f}m" if rng < 1.5
                    else f"LIDAR ZERO {rng:.1f}m - NO AUTO-DISARM"[:50],
                    Notifier.SEV_INFO if rng < 1.5 else Notifier.SEV_WARN)

            # THE ORIGIN IS RE-TAKEN AT ARM. The pilot's instruction, and
            # three flights of evidence behind it: SCR_USER1/2 was left 147,
            # 148 and 157 m from where the aircraft actually stood, because it
            # is typed by hand and the aircraft moves. Every VPS position is
            # measured from the origin, so flipping to the VPS with the origin
            # 150 m out hands the flight controller a 150 m step.
            #
            # At the moment of arming the aircraft is standing still on the
            # ground at the place it is about to fly from, which is the one
            # moment its position is certainly known. So:
            #
            #   on the receiver (GPS_PRIMARY 0) -> take the receiver's fix
            #   on the VPS                      -> take SCR_USER1/2
            #
            # and the VIO's accumulated displacement goes back to zero,
            # because it was measured from the origin that just moved.
            # Anchoring is the VIO's job; the map only corrects it afterwards.
            armed_now = f.get("armed")
            if (armed_now is True and _armed_was[0] is not True
                    and node.origin is not None):
                src_now = watcher.value
                new_o = why_o = None
                # The receiver states its own accuracy and SCR_USER1/2 cannot:
                # one is a measurement, the other is typed. Measured in the
                # recent logs, GPS1 reported 1.08 to 3.01 m with a lock and
                # UINT32_MAX without one (the reader discards that), so taking
                # the origin from the receiver is worth about 2 m of declared
                # uncertainty. A hand-typed point gets None, which
                # absolute_sigma_m reads as the map's own error - and the three
                # flights that flew on a stale SCR_USER were 147-157 m out.
                new_sigma = None
                if src_now == 0 and f["gps"]:
                    new_o, why_o = f["gps"], "receiver at arm"
                    new_sigma = f.get("gps1_hacc")
                elif scr_origin is not None:
                    new_o, why_o = scr_origin, "SCR_USER1/2 at arm"
                if new_o is not None:
                    moved = float(np.hypot(
                        (new_o[1] - node.origin[1]) * M_PER_DEG_LAT,
                        (new_o[0] - node.origin[0]) * node.mlon))
                    node.set_origin(*new_o, sigma_m=new_sigma)
                    node.fuser.reset_displacement()
                    print(f"ARM: origin re-taken from {why_o}: "
                          f"{new_o[1]:.6f}, {new_o[0]:.6f} "
                          f"(moved {moved:.0f} m, origin known to "
                          f"{'unmeasured' if new_sigma is None else f'{new_sigma:.1f} m'}"
                          f", declaring {node.absolute_sigma_m:.1f} m)", flush=True)
                    notifier.event("reanchor", round(moved),
                                   f"VPS ORIGIN {why_o} {moved:.0f}m"[:50],
                                   Notifier.SEV_INFO if moved < 50
                                   else Notifier.SEV_WARN)
                else:
                    notifier.event("reanchor", "none",
                                   "VPS ORIGIN NOT RE-TAKEN AT ARM",
                                   Notifier.SEV_WARN)
            # THE HANDOVER, IN FLIGHT. The pilot's instruction, 21 Oct, and the
            # reasoning is his own: when the switch goes the other way, from the
            # VPS back to GPS, the flight controller does not drag the aircraft
            # across whatever error had built up - "man lo error 50 mtr hai tho
            # gps drone ko 50 mtr lekar correct nahi karta, wo direct apne
            # current position inject kar deta hai aur wahi stable ho jata hai".
            # Handing over TO the VPS has to behave the same way.
            #
            # It did not. The origin was only ever re-taken at the ARM edge, so
            # a flight that armed on GPS, flew, and then switched to the VPS at
            # altitude started sending `takeoff origin + every metre of VIO
            # drift accumulated during the whole GPS phase`. Measured on the
            # three flights with real travel below 60 m: 5.73 m and 5.24 m of
            # drift on two of them, and a single-sample 20.55 m jump on
            # vps_00016 at 41 m altitude. The EKF holds the GPS position at that
            # instant, so it sees a step of exactly that size.
            #
            # So on the transition INTO the VPS, while armed, the origin is
            # re-taken from where the flight controller currently believes it is
            # and the VIO's displacement goes back to zero. The first position
            # the VPS then sends equals what the EKF already holds, which is a
            # handover with no jump and no drift carried across it. Nothing is
            # corrected, nothing is flown - exactly as the pilot described.
            #
            # DISARMED is deliberately excluded. A switch flipped on the ground
            # is case one - take off on the VPS - and there the origin must come
            # from SCR_USER1/2, which the arm edge above already does.
            #
            # f["gps"] is GLOBAL_POSITION_INT, which is the EKF's own fused
            # position rather than the raw receiver. That is the right source
            # HERE and not a compromise: the whole point is to start from what
            # the flight controller believes, because that is what makes the
            # step zero. While the aircraft is on GPS that belief is GPS-driven.
            # It is read once, at this edge, so nothing loops.
            src_val = watcher.value
            # _armed_was[0] is True, not armed_now: the aircraft has to have
            # ALREADY been armed on the previous frame. That is what "in flight"
            # means here, and it also stops this colliding with the arm edge
            # above if the pilot flips to the VPS and arms in the same frame -
            # that is case one, and the arm edge's SCR_USER1/2 must stand.
            if should_handover(src_val, _src_was[0], _armed_was[0],
                               armed_now, node.origin is not None):
                if f["gps"]:
                    moved = float(np.hypot(
                        (f["gps"][1] - node.origin[1]) * M_PER_DEG_LAT,
                        (f["gps"][0] - node.origin[0]) * node.mlon))
                    carried = float(np.hypot(*node.fuser.pos))
                    node.set_origin(*f["gps"], sigma_m=f.get("gps1_hacc"))
                    node.fuser.reset_displacement()
                    print(f"HANDOVER to VPS in flight: origin re-taken from the "
                          f"FC's own position {f['gps'][1]:.6f}, {f['gps'][0]:.6f} "
                          f"(origin moved {moved:.1f} m, {carried:.1f} m of VIO "
                          f"displacement discarded, declaring "
                          f"{node.absolute_sigma_m:.1f} m)", flush=True)
                    notifier.event("handover", round(carried),
                                   f"VPS HANDOVER - dropped {carried:.0f}m drift"[:50],
                                   Notifier.SEV_INFO)
                else:
                    # No fix to hand over from - which is the jammed case this
                    # aircraft exists for. Keep the origin and the displacement
                    # we have: they carry whatever drift has built up, and that
                    # is far better than falling back to SCR_USER1/2, which on
                    # three flights sat 147-157 m from the aircraft.
                    notifier.event("handover", "noloc",
                                   "VPS HANDOVER - NO FIX, drift carried",
                                   Notifier.SEV_WARN)
                    print("HANDOVER to VPS in flight: the FC had no position to "
                          "hand over, so the existing origin and displacement "
                          "are kept (drift is carried across)", flush=True)
            _src_was[0] = src_val
            _armed_was[0] = armed_now

            gN = gE = err = None
            if f["gps"]:
                gN = (f["gps"][1] - node.origin[1]) * M_PER_DEG_LAT
                gE = (f["gps"][0] - node.origin[0]) * node.mlon
                err = float(np.hypot(est["north"] - gN, est["east"] - gE))
                errs.append(err)
            # And the same against the RAW receiver, which is a different
            # measurement and the only independent one. err above is against
            # GLOBAL_POSITION_INT, the EKF's own fused position; once the
            # aircraft flies on the VPS that is partly our own output returning,
            # and comparing against it flatters us. This pair does not.
            est["g1_north"] = est["g1_east"] = est["g1_err"] = None
            if f.get("gps1_raw"):
                g1 = f["gps1_raw"]
                est["g1_north"] = (g1[1] - node.origin[1]) * M_PER_DEG_LAT
                est["g1_east"] = (g1[0] - node.origin[0]) * node.mlon
                est["g1_err"] = float(np.hypot(est["north"] - est["g1_north"],
                                               est["east"] - est["g1_east"]))
            v = est["vel"]
            if fwriter is not None:
                # ARMED ONLY, at the pilot's instruction (20 Oct). Images are
                # wanted for the flight, not for the hour the aircraft sits on
                # the ground with the services up: that hour is where the 29 MB
                # a minute and the bench frames at alt 0.1 m come from, and
                # every one of them is a frame offline matching has to be told
                # to ignore.
                #
                # `armed` is the FC's own flag out of the telemetry snapshot.
                # Three states, and the direction of each is deliberate:
                # True keeps saving, False stops, and None - no FC, or a link
                # that never carried a heartbeat - does NOT save, because
                # "cannot tell" on the bench is the common case and a flight
                # always has a link. If the link dies mid-flight the snapshot
                # FREEZES rather than emptying (see Telemetry), so armed stays
                # True and the recording continues, which is the safe way for
                # that failure to fall.
                armed = f.get("armed")
                if armed is True:
                    # After n, so the name carries the index of the row about
                    # to be written, and before csv_row, so that row names its
                    # own image. offer() encodes nothing and waits for
                    # nothing: if the writer is behind, this returns None and
                    # the frame is gone.
                    est["frame_file"] = fwriter.offer(f["img"], n) or ""
                elif not a.frames_any_time:
                    est["frame_file"] = ""
                else:
                    est["frame_file"] = fwriter.offer(f["img"], n) or ""
                notifier.event(
                    "frames", bool(armed) or a.frames_any_time,
                    f"VPS FRAMES ON {fwriter.hz:g}Hz" if (bool(armed)
                                                          or a.frames_any_time)
                    else "VPS FRAMES OFF - DISARMED",
                    Notifier.SEV_INFO)
            if wr:
                wr.writerow(csv_row(est, f, v, gN, gE, err))
            if imu_wr is not None:
                imu_wr.writerows(imu_rows(f))
            # A row in the page cache is not a row on the disk. The battery
            # comes out of this aircraft by hand, and when it did on 20 Oct
            # the last block of two runs came back as NUL bytes.
            if a.sync_every > 0 and est["t"] - last_sync >= a.sync_every:
                last_sync = est["t"]
                for h_ in (fh, imu_fh):
                    if h_ is None:
                        continue
                    try:
                        h_.flush()
                        os.fsync(h_.fileno())
                    except OSError:
                        pass
            notifier.tick(est["t"], est, node)

            # Gated on the source. GPS_PRIMARY=0 means the pilot asked for the
            # real M9N, so the VPS stops injecting entirely - not "sends and is
            # ignored", stops. Unknown keeps sending; see SourceWatcher for why
            # the two directions are not symmetrical.
            if gps_sender is not None and on_vps:
                amsl_ref.observe(f.get("gps1_fix"), f.get("gps1_alt"), f["alt"])
                amsl = amsl_ref.amsl(f["alt"])
                notifier.event("amsl", amsl_ref.source or False,
                               f"VPS AMSL from {amsl_ref.source}"[:50]
                               if amsl_ref.source
                               else "VPS AMSL UNKNOWN - pass --site-alt",
                               Notifier.SEV_INFO if amsl_ref.source
                               else Notifier.SEV_WARN)
                gps_sender.send(est["lon"], est["lat"],
                                est["alt"] if amsl is None else amsl,
                                est["sigma_abs"], est["yaw"], vel=est["vel"])
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
        if fwriter is not None:
            # Before the CSV closes, and bounded: queued frames get a few
            # seconds to land, and then we stop regardless. A node that hangs
            # on a slow disk at shutdown is a node that does not come back.
            print(fwriter.close())
        if wr:
            fh.close()
        if imu_fh is not None:
            imu_fh.close()

    dt = time.perf_counter() - t_start
    print(f"\n{n} frames in {dt:.1f} s = {n/max(dt,1e-9):.1f} Hz processed")
    print(f"flow steps {node.n_flow}, map fixes {node.n_fix} "
          f"(worker: {node.worker.submitted} submitted, {node.worker.dropped} dropped), "
          f"fuser accepted {node.fuser.n_fix} rejected {node.fuser.n_rejected}")
    tot = node.n_derot_gyro + node.n_derot_quat + node.n_derot_fc
    if tot:
        print(f"derotation: {node.n_derot_gyro} frames from the camera gyro, "
              f"{node.n_derot_quat} from GAME_ROTATION_VECTOR, "
              f"{node.n_derot_fc} from the FC attitude "
              f"({100.0 * node.n_derot_gyro / tot:.1f}% gyro)")
        print("  " + node.val_gyro.describe())
        print("  " + node.val_quat.describe())
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
