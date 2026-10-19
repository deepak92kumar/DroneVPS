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
                 fps: int | None = None, stereo_depth: bool | None = None):
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
        imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_CALIBRATED, 100)
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
        K = np.array(cal.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, 1280, 800))
        self.focal = float(K[0][0])
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
        self._t_gyro = None
        self.t0 = time.monotonic()

    def read(self):
        fr = self.q.get()
        # tryGet, never get: the mono frame is what the flow loop is waiting on,
        # and depth is an aid to it. Missing depth means this frame has none,
        # not that the loop waits for it.
        # Integrate the gyro over the interval this frame covers.
        d_roll = d_pitch = 0.0
        pk = self.q_imu.tryGet()
        while pk is not None:
            for packet in pk.packets:
                g = getattr(packet, "gyroscope", None)
                if g is None:
                    continue
                ts = g.getTimestampDevice().total_seconds()
                if self._t_gyro is not None:
                    dt = ts - self._t_gyro
                    if 0.0 < dt < 0.1:
                        # x->roll, y->pitch is a GUESS. See __init__.
                        d_roll += np.degrees(g.x) * dt
                        d_pitch += np.degrees(g.y) * dt
                        self._gyro_n += 1
                self._t_gyro = ts
            pk = self.q_imu.tryGet()

        depth = None
        if self.q_depth is not None:
            d = self.q_depth.tryGet()
            if d is not None:
                depth = d.getFrame()        # uint16, millimetres, aligned to CAM_B
        s = self.tel.snapshot()
        if s.get("alt_rel") is None or s.get("yaw") is None:
            return dict(img=fr.getCvFrame(), alt=None, yaw=None, roll=None,
                        pitch=None, gps=None, depth=depth,
                        gyro_d_roll=d_roll, gyro_d_pitch=d_pitch,
                        gyro_n=self._gyro_n,
                        t=time.monotonic() - self.t0)
        gps = None
        if s.get("lat") not in (None, 0.0):
            gps = (s["lon"], s["lat"])
        return dict(img=fr.getCvFrame(), alt=s["alt_rel"], yaw=s["yaw"],
                    roll=s["roll"], pitch=s["pitch"], gps=gps, depth=depth,
                    gyro_d_roll=d_roll, gyro_d_pitch=d_pitch,
                    gyro_n=self._gyro_n,
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

    def __init__(self, port="/dev/ttyTHS1", baud=115200):
        from pymavlink import mavutil
        self.mavutil = mavutil
        self.m = mavutil.mavlink_connection(port, baud=baud, source_system=1,
                                            source_component=197)
        if self.m.wait_heartbeat(timeout=20) is None:
            raise RuntimeError("FC se heartbeat nahi aaya")
        if bool(self.m.motors_armed()):
            raise RuntimeError("*** ARMED - bench test disarmed hi hoga ***")
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
                 sats: int = 12, send_velocity: bool = False):
        from pymavlink import mavutil
        self.mavutil = mavutil
        self.m = mavutil.mavlink_connection(port, baud=baud, source_system=1,
                                            source_component=220)
        if self.m.wait_heartbeat(timeout=20) is None:
            raise RuntimeError("FC se heartbeat nahi aaya")
        if bool(self.m.motors_armed()):
            raise RuntimeError("*** ARMED - bench test disarmed hi hoga ***")
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


class VpsNode:
    def __init__(self, cache: MapCache, focal_px: float, *,
                 fix_period_s: float = 1.0, radius_m: float = 150.0,
                 max_corners: int = 300):
        self.cache = cache
        self.focal = float(focal_px)
        self.tracker = FlowTracker(max_corners=max_corners)
        self.fuser = MapFuser(flow_drift_frac=0.07, fix_sigma_m=12.0)
        self.worker = MapFixWorker(cache, focal_px=focal_px, radius_m=radius_m)
        self.fix_period = float(fix_period_s)
        self.origin = None            # (lon, lat)
        self.mlon = None
        self._next_fix = 0.0
        self.prev_att = None
        self.prev_t = None
        self.n_fix = self.n_flow = 0
        # Flow-only on this path: LiveSource does not enable the camera IMU, so
        # nothing calls push_accel. The bridge is built for that - with no
        # acceleration it simply holds between flow updates instead of
        # dead-reckoning through them, and says how stale it is.
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
        d, ntrk = self.tracker.step(f["img"])
        if d is not None and self.prev_att is not None:
            d = d - attitude_shift_px(self.prev_att, att)
            dn, de = shift_to_ne(d, altitude_m=f["alt"], focal_px=self.focal,
                                 yaw_deg=f["yaw"])
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
                    gyro_n=f.get("gyro_n"))

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
    ap.add_argument("--fix-period", type=float, default=1.0)
    ap.add_argument("--radius", type=float, default=150.0)
    ap.add_argument("--corners", type=int, default=300)
    ap.add_argument("--origin", help="lon,lat - GPS-denied start")
    ap.add_argument("--fc-port", default="/dev/ttyTHS1",
                    help="FC endpoint: a device, or udpout:127.0.0.1:14650 "
                         "through the router")
    ap.add_argument("--fc-baud", type=int, default=921600,
                    help="TELEM2 flies at 921600. Ignored for a udp endpoint.")
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
    ap.add_argument("--send-vel", action="store_true",
                    help="also send horizontal velocity in GPS_INPUT. OFF by "
                         "default: this is the channel that reported 47.7 m/s "
                         "at a 106 m hover on 28 Sep and caused four EKF "
                         "failsafes, and it is still the largest open fault.")
    ap.add_argument("--gps-id", type=int, default=1, help="0 = GPS1, 1 = GPS2")
    a = ap.parse_args()

    if a.send_fc:
        import subprocess
        st = subprocess.run(["systemctl", "is-active", "drone-vio"],
                            capture_output=True, text=True).stdout.strip()
        if st == "active":
            sys.exit("drone-vio service chal rahi hai - ye node usse ladega. Pehle band karo.")

    gps_sender = None
    if a.send_gps:
        gps_sender = GpsInputSender(gps_id=a.gps_id, send_velocity=a.send_vel)
        print(f"GPS INPUT SENDER ON - disarmed confirmed. GPS{a.gps_id+1} ban kar jayega.\n"
              f"   (horiz_accuracy = fuser ka apna sigma)")

    sender = None
    if a.send_fc:
        sender = FcSender()
        print(f"FC SENDER ON - disarmed confirmed. VISION_POSITION_ESTIMATE jayega.\n"
              f"   (EK3 sources ko haath nahi lagaya - FC padhega par istemal nahi karega)")

    # Read the typed take-off point BEFORE the camera source opens the serial
    # port, so this works whether --fc-port is a device or the router's UDP.
    scr_origin = None
    if a.live and not a.no_scr_origin and not a.origin:
        scr_origin = read_scr_user(a.fc_port)
        if scr_origin is None:
            print("SCR_USER1/2: not readable or not set - will fall back to GPS")
        else:
            print(f"SCR_USER1/2: {scr_origin[1]:.6f}, {scr_origin[0]:.6f}")

    src = (ReplaySource(a.replay, a.every, a.min_alt) if a.replay
           else LiveSource(a.fc_port, a.fc_baud))
    cache = MapCache(a.map)
    print(cache.summary())
    print(f"focal {src.focal:.1f} px, fix har {a.fix_period:.1f} s, window +-{a.radius:.0f} m, "
          f"{a.corners} corners\n", flush=True)

    node = VpsNode(cache, src.focal, fix_period_s=a.fix_period,
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
                if a.origin:
                    lo, la = (float(x) for x in a.origin.split(","))
                    node.set_origin(lo, la)
                    why = "--origin"
                elif scr_origin is not None:
                    node.set_origin(*scr_origin)
                    why = "SCR_USER1/2 (typed on the RC)"
                elif f["gps"]:
                    node.set_origin(*f["gps"])
                    why = "the live receiver"
                else:
                    continue
                olon, olat = node.origin
                covered = cache.covers(olon, olat)
                print(f"origin from {why}: {olat:.6f}, {olon:.6f}")
                print(f"  sheet covers this point: "
                      f"{'YES' if covered else 'NO - there will be NO map fixes'}")
                if not covered:
                    print(f"  loaded sheet: {a.map}")
            est = node.step(f)
            if est is None:
                continue
            n += 1
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
                             est["gyro_n"]])
            if gps_sender is not None:
                if gps_sender.armed():
                    print("\n*** AIRCRAFT ARMED - transmission band ***")
                    break
                gps_sender.send(est["lon"], est["lat"],
                                (f.get("alt_amsl") or est["alt"]), est["sigma"],
                                est["yaw"], vel=est["vel"])
            if sender is not None:
                if sender.armed():
                    print("\n*** AIRCRAFT ARMED - transmission band ***")
                    break
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
