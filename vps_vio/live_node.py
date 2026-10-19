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

M_PER_DEG_LAT = 111_320.0
DEFAULT_MAP = "/home/abc/SSD/DRONE/ap/site_2km.tif"


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

    def __init__(self, fc_port="/dev/ttyTHS1", fc_baud=115200, *,
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
        self.t0 = time.monotonic()

    def read(self):
        fr = self.q.get()
        # tryGet, never get: the mono frame is what the flow loop is waiting on,
        # and depth is an aid to it. Missing depth means this frame has none,
        # not that the loop waits for it.
        depth = None
        if self.q_depth is not None:
            d = self.q_depth.tryGet()
            if d is not None:
                depth = d.getFrame()        # uint16, millimetres, aligned to CAM_B
        s = self.tel.snapshot()
        if s.get("alt_rel") is None or s.get("yaw") is None:
            return dict(img=fr.getCvFrame(), alt=None, yaw=None, roll=None,
                        pitch=None, gps=None, depth=depth,
                        t=time.monotonic() - self.t0)
        gps = None
        if s.get("lat") not in (None, 0.0):
            gps = (s["lon"], s["lat"])
        return dict(img=fr.getCvFrame(), alt=s["alt_rel"], yaw=s["yaw"],
                    roll=s["roll"], pitch=s["pitch"], gps=gps, depth=depth,
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
    as they do today, missions stay in latitude and longitude, and with
    GPS_AUTO_SWITCH already set to "use the best" the aircraft moves onto this
    source by itself the moment the real GPS loses its fix. Fifty-four flights
    of experience are on the GPS path; this keeps them.

    What makes the trust defensible is that horiz_accuracy carries the fuser's
    own sigma. When the estimate is drifting the flight controller is told so,
    and GPS blending weighs it against the real receiver instead of guessing.
    """

    # GPS_INPUT_IGNORE_FLAGS
    IGN_VEL_HORIZ, IGN_VEL_VERT, IGN_SPEED_ACC = 8, 16, 32

    def __init__(self, port="/dev/ttyTHS1", baud=115200, gps_id: int = 1,
                 sats: int = 12):
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
        self.sent = 0

    def armed(self) -> bool:
        msg = self.m.recv_match(type="HEARTBEAT", blocking=False)
        return False if msg is None else bool(
            msg.base_mode & self.mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    def send(self, lon, lat, alt_amsl_m, sigma_m, yaw_deg=None):
        t = time.time()
        week = int((t - 315964800) // (7 * 86400))
        week_ms = int(((t - 315964800) % (7 * 86400)) * 1000)
        # HDOP is what a GCS shows the pilot, so it should read like the real
        # thing: metres of error over the usual ~1.5 m per HDOP unit.
        hdop = float(np.clip(sigma_m / 1.5, 0.5, 99.0))
        self.m.mav.gps_input_send(
            int(t * 1e6), self.gps_id,
            self.IGN_VEL_HORIZ | self.IGN_VEL_VERT | self.IGN_SPEED_ACC,
            week_ms, week, 3,                      # fix_type 3 = 3D
            int(lat * 1e7), int(lon * 1e7), float(alt_amsl_m),
            hdop, hdop, 0.0, 0.0, 0.0,
            0.0, float(sigma_m), float(sigma_m * 2.0),
            self.sats)
        self.sent += 1


# --------------------------------------------------------------------------- #
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
        self.n_fix = self.n_flow = 0

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
        d, ntrk = self.tracker.step(f["img"])
        if d is not None and self.prev_att is not None:
            d = d - attitude_shift_px(self.prev_att, att)
            dn, de = shift_to_ne(d, altitude_m=f["alt"], focal_px=self.focal,
                                 yaw_deg=f["yaw"])
            self.fuser.predict(dn, de)
            self.n_flow += 1
        self.prev_att = att

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
                    fix_used=fix_used, alt=f["alt"], yaw=f["yaw"], t=f["t"])

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
    ap.add_argument("--out", help="estimate log CSV")
    ap.add_argument("--send-fc", action="store_true",
                    help="transmit VISION_POSITION_ESTIMATE (disarmed only)")
    ap.add_argument("--send-gps", action="store_true",
                    help="inject as GPS2 via GPS_INPUT, Stride-style (disarmed only)")
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
        gps_sender = GpsInputSender(gps_id=a.gps_id)
        print(f"GPS INPUT SENDER ON - disarmed confirmed. GPS{a.gps_id+1} ban kar jayega.\n"
              f"   (horiz_accuracy = fuser ka apna sigma)")

    sender = None
    if a.send_fc:
        sender = FcSender()
        print(f"FC SENDER ON - disarmed confirmed. VISION_POSITION_ESTIMATE jayega.\n"
              f"   (EK3 sources ko haath nahi lagaya - FC padhega par istemal nahi karega)")

    src = ReplaySource(a.replay, a.every, a.min_alt) if a.replay else LiveSource()
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
        wr.writerow(["t", "north_m", "east_m", "lon", "lat", "sigma_m",
                     "tracks", "fix_used", "alt_m", "yaw_deg",
                     "gps_north_m", "gps_east_m", "err_m"])

    errs, t_start, n = [], time.perf_counter(), 0
    try:
        while True:
            f = src.read()
            if f is None:
                break
            if node.origin is None:
                if a.origin:
                    lo, la = (float(x) for x in a.origin.split(","))
                    node.set_origin(lo, la)
                elif f["gps"]:
                    node.set_origin(*f["gps"])
                    print(f"origin GPS se: {f['gps'][1]:.6f}, {f['gps'][0]:.6f}")
                else:
                    continue
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
            if wr:
                wr.writerow([f"{est['t']:.2f}", f"{est['north']:.2f}", f"{est['east']:.2f}",
                             f"{est['lon']:.7f}", f"{est['lat']:.7f}",
                             f"{est['sigma']:.1f}", est["tracks"], int(est["fix_used"]),
                             f"{est['alt']:.1f}", f"{est['yaw']:.1f}",
                             "" if gN is None else f"{gN:.2f}",
                             "" if gE is None else f"{gE:.2f}",
                             "" if err is None else f"{err:.2f}"])
            if gps_sender is not None:
                if gps_sender.armed():
                    print("\n*** AIRCRAFT ARMED - transmission band ***")
                    break
                gps_sender.send(est["lon"], est["lat"],
                                (f.get("alt_amsl") or est["alt"]), est["sigma"],
                                est["yaw"])
            if sender is not None:
                if sender.armed():
                    print("\n*** AIRCRAFT ARMED - transmission band ***")
                    break
                sender.send(est["north"], est["east"], -est["alt"],
                            f["roll"], f["pitch"], f["yaw"])
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
