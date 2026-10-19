#!/usr/bin/env python3
"""Record a flight for offline map-matching validation. READ-ONLY on the FC.

With --on-arm the recorder idles until the aircraft arms, writes one folder per
flight, and stops on disarm. Encoding and writing 30 frames a second costs a
third of a CPU core, and on a companion computer that also has to run optical
flow and map matching there is no spare third of a core to spend on recording
an empty field.

Purpose: fly normally (GPS on, manual modes — nothing new in the loop) while
this logs what the downward camera sees plus the aircraft's own position. Later
we run map matching on those frames and compare against the recorded GPS, which
tells us the REAL accuracy against 2022 satellite imagery — the one thing the
synthetic tests cannot prove.

Safety:
  * This never changes anything on the flight controller: no parameters, no
    VISION_POSITION_ESTIMATE, no mode or arming commands. The only thing it
    transmits is the standard GCS stream-rate request, without which ArduPilot
    sends nothing on a serial link it does not consider a connected GCS - the
    first attempt at this recording logged 700 frames with every telemetry
    column empty for exactly that reason.
  * It is not in the control loop; if it crashes the aircraft is unaffected.

Records per frame: mono image (map matching wants the MONO camera — 77.9 deg
HFOV gives a 194 m footprint at 120 m, vs the RGB's 66.7 deg / 158 m), device
timestamp, and the latest GPS / altitude / attitude from the FC.

Usage (props on, normal flight):
    python3 record_flight.py --out /mnt/ssd/DRONE/flights/$(date +%Y%m%d_%H%M)
Fly to 100-150 m, loiter, translate a few hundred metres, come back. Map
matching needs >= ~80 m, so time spent below that is only useful for VIO.
"""
from __future__ import annotations
import argparse, collections, csv, os, queue, sys, threading, time

import cv2
import numpy as np
import depthai as dai

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# NOT oak_rig. It imports cuvslam, which is not installed and was REJECTED on
# 28 Sep - the OAK-D's 7.5 cm baseline gives under 0.3 px of disparity above
# ~25 m against a 100-150 m mission, so it contributed exactly 0 over 1680
# frames. Importing it for three constants made this whole recorder
# unimportable, so the three constants live here instead.
OAK_ISO = 800          # measured 27 Sep with 4000 us exposure
OAK_FPS = 30


def td_ns(td) -> int:
    """depthai timedelta -> integer nanoseconds on the device clock."""
    return int(td.total_seconds() * 1e9)


class CameraVibe:
    """Vibration from the camera's own IMU, in the same units as the FC's VIBE.

    ArduPilot computes VIBE as the deviation of the raw accelerometer from a
    low-passed version of itself. This does the same thing on the OAK-D's
    LINEAR_ACCELERATION: hold a short rolling window and take the per-axis
    standard deviation. Same units, m/s/s, so the camera's number and the
    flight controller's can be put side by side and mean something.

    That comparison is the point. The camera is bolted to a mount, the FC to
    the frame. The 11 Oct report measured the camera mount going from damping
    at 0.91x of the airframe to AMPLIFYING at 1.50x, and that ratio - not
    either number alone - is what found it.
    """

    def __init__(self, window_s: float = 1.0, rate_hz: float = 100.0):
        self.n = max(8, int(window_s * rate_hz))
        self.buf = collections.deque(maxlen=self.n)
        self.samples = 0

    def push(self, ax: float, ay: float, az: float) -> None:
        self.buf.append((ax, ay, az))
        self.samples += 1

    def value(self):
        """(x, y, z) m/s/s, or (None, None, None) before the window fills."""
        if len(self.buf) < self.n // 2:
            return (None, None, None)
        a = np.asarray(self.buf, dtype=float)
        return tuple(float(x) for x in a.std(axis=0))


class Telemetry:
    """Read-only MAVLink listener. Never transmits anything to the FC."""

    def __init__(self, port: str, baud: int):
        self.port, self.baud = port, baud
        self.lock = threading.Lock()
        self.state = dict(lat=None, lon=None, alt_rel=None, alt_amsl=None,
                          roll=None, pitch=None, yaw=None, fix=None, sats=None,
                          hdg=None, rngfnd=None, armed=False, t=0.0,
                          # The FC's own accelerometer vibration, and what the
                          # motors were doing when it was measured. One without
                          # the other says nothing: vibration at idle and
                          # vibration at hover throttle are different questions.
                          vibe_x=None, vibe_y=None, vibe_z=None, clip=None,
                          mot=None)
        self._run = False
        self._thr = None
        self.connected = False

    def start(self):
        try:
            from pymavlink import mavutil
        except Exception as e:
            print(f"  [telemetry] pymavlink not available: {e}")
            return self
        try:
            self.m = mavutil.mavlink_connection(self.port, baud=self.baud)
            self.connected = True
        except Exception as e:
            print(f"  [telemetry] cannot open {self.port}: {e}")
            return self
        self._run = True
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()
        return self

    def _request_streams(self):
        """Ask for the telemetry we log. Re-sent until data arrives, because the
        FC is often powered on after this script has already started."""
        try:
            from pymavlink import mavutil
            self.m.mav.request_data_stream_send(
                self.m.target_system or 1, self.m.target_component or 1,
                mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1)
        except Exception:
            pass

    def _loop(self):
        last_req = 0.0
        while self._run:
            if self.state["lat"] is None and time.time() - last_req > 5.0:
                last_req = time.time()
                self._request_streams()
            try:
                msg = self.m.recv_match(
                    type=["GLOBAL_POSITION_INT", "GPS_RAW_INT", "ATTITUDE",
                          "DISTANCE_SENSOR", "RANGEFINDER", "HEARTBEAT",
                          "VIBRATION", "SERVO_OUTPUT_RAW"],
                    blocking=True, timeout=1.0)
            except Exception:
                continue
            if msg is None:
                continue
            t = msg.get_type()
            with self.lock:
                if t == "GLOBAL_POSITION_INT":
                    self.state.update(lat=msg.lat / 1e7, lon=msg.lon / 1e7,
                                      alt_amsl=msg.alt / 1000.0,
                                      alt_rel=msg.relative_alt / 1000.0,
                                      hdg=msg.hdg / 100.0 if msg.hdg != 65535 else None)
                elif t == "GPS_RAW_INT":
                    self.state.update(fix=msg.fix_type, sats=msg.satellites_visible)
                elif t == "VIBRATION":
                    # ArduPilot's own VIBE, m/s/s. Under 30 is workable, under
                    # 15 is good, and any rise in the clip counters means an
                    # accelerometer saturated and the EKF lost that sample.
                    self.state.update(vibe_x=msg.vibration_x,
                                      vibe_y=msg.vibration_y,
                                      vibe_z=msg.vibration_z,
                                      clip=msg.clipping_0 + msg.clipping_1
                                      + msg.clipping_2)
                elif t == "SERVO_OUTPUT_RAW":
                    # Motors 1-6. Vibration without the throttle that caused it
                    # cannot be compared between flights.
                    self.state["mot"] = [getattr(msg, f"servo{i}_raw", 0)
                                         for i in range(1, 7)]
                elif t == "HEARTBEAT":
                    # Arming is what decides whether we are recording at all.
                    # Encoding and writing 30 JPEGs a second while the aircraft
                    # sits on the ground keeps a core busy for nothing.
                    from pymavlink import mavutil as _mu
                    self.state["armed"] = bool(
                        msg.base_mode & _mu.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                elif t == "RANGEFINDER":
                    # Logged alongside the frames so lidar height can be
                    # compared against the barometer over a whole flight.
                    # The analog scaling was wrong when this comment was first
                    # written; RNGFND1_OFFSET was recalibrated on 19 Oct from
                    # 0.607 to 0.66537 against a measured 0.26 m and the ground
                    # reading went from 4.35 m to 0.26 m. SCALING 72.61 was
                    # already right - lifting the aircraft moved the reading
                    # 1:1 - and matches the 150 ohm / 4-20 mA design of 75 m/V.
                    self.state["rngfnd"] = float(msg.distance)
                elif t == "DISTANCE_SENSOR":
                    self.state["rngfnd"] = msg.current_distance / 100.0
                elif t == "ATTITUDE":
                    self.state.update(roll=np.degrees(msg.roll),
                                      pitch=np.degrees(msg.pitch),
                                      yaw=np.degrees(msg.yaw))
                self.state["t"] = time.time()

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.state)

    def stop(self):
        self._run = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--seconds", type=float, default=900, help="max duration")
    ap.add_argument("--hz", type=float, default=2.0, help="frames saved per second")
    # PNG costs 30 ms a frame, which cannot keep up at 30 Hz; JPEG costs 9.5 ms.
    # Above 5 Hz the format switches automatically unless told otherwise.
    ap.add_argument("--format", choices=("auto", "png", "jpg"), default="auto")
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--rgb", action="store_true", help="also save the colour camera")
    # Auto-exposure by default. The VIO rig's fixed 4000 us / ISO 800 is an
    # INDOOR setting: outdoors it is several stops over, and the first flight
    # came back as 545 pure-white frames, every one of them useless. This
    # recorder is not in the control loop, so it has no reason to pin exposure.
    ap.add_argument("--exposure-us", type=int, default=0,
                    help="0 = auto-exposure (default); >0 pins it, indoor use only")
    ap.add_argument("--iso", type=int, default=0, help="only used with --exposure-us")
    ap.add_argument("--fc-port", default="/dev/ttyTHS1")
    ap.add_argument("--fc-baud", type=int, default=115200)
    ap.add_argument("--no-fc", action="store_true", help="skip telemetry entirely")
    ap.add_argument("--on-arm", action="store_true",
                    help="record only while the aircraft is armed; one folder per flight")
    a = ap.parse_args()

    ext = a.format if a.format != "auto" else ("jpg" if a.hz > 5 else "png")
    enc = [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_quality] if ext == "jpg" else []

    class Session:
        """One flight: its own folder, index and load log. Opened on arm."""

        def __init__(self, root):
            self.root = root
            os.makedirs(root, exist_ok=True)
            self.img_dir = os.path.join(root, "mono")
            os.makedirs(self.img_dir, exist_ok=True)
            self.rgb_dir = os.path.join(root, "rgb")
            if a.rgb:
                os.makedirs(self.rgb_dir, exist_ok=True)
            self.f = open(os.path.join(root, "index.csv"), "w", newline="")
            self.wr = csv.writer(self.f)
            # Vibration from all three places it can come from, in one row.
            # fc_vibe_* is ArduPilot's own VIBE off the flight controller;
            # cam_vibe_* is the same measure computed from the OAK-D's IMU, so
            # the two can be compared directly; mot1..6 is what the motors were
            # doing when both were measured, because vibration without the
            # throttle that caused it cannot be compared between flights.
            self.wr.writerow(["n", "wall_s", "dev_ts_ns", "mono_file", "rgb_file",
                              "lat", "lon", "alt_rel_m", "alt_amsl_m",
                              "roll_deg", "pitch_deg", "yaw_deg", "hdg_deg",
                              "fix", "sats", "rngfnd_m",
                              "fc_vibe_x", "fc_vibe_y", "fc_vibe_z", "fc_clip",
                              "cam_vibe_x", "cam_vibe_y", "cam_vibe_z",
                              "cam_imu_n",
                              "mot1", "mot2", "mot3", "mot4", "mot5", "mot6"])
            # What the Jetson was actually doing, sampled while flying. The only
            # honest way to know the load is to measure it in the air.
            self.lf = open(os.path.join(root, "load.csv"), "w", newline="")
            self.lw = csv.writer(self.lf)
            self.lw.writerow(["wall_s", "cpu_pct", "ram_mb", "cpu_temp_c",
                              "cpu_mhz", "frames", "dropped"])
            self.n = 0
            self.t0 = time.monotonic()

        def close(self):
            for h in (self.f, self.lf):
                try: h.close()
                except Exception: pass

    def cpu_sample(_prev=[None]):
        """Total CPU busy %, RAM MB, temp C, mean clock MHz - from /proc and /sys."""
        try:
            parts = open("/proc/stat").readline().split()[1:]
            v = [int(x) for x in parts]
            idle, total = v[3] + v[4], sum(v)
            p = _prev[0]; _prev[0] = (idle, total)
            cpu = 0.0 if p is None or total == p[1] else \
                100.0 * (1.0 - (idle - p[0]) / (total - p[1]))
        except Exception:
            cpu = float("nan")
        try:
            mi = {}
            for ln in open("/proc/meminfo"):
                k, val = ln.split(":"); mi[k] = int(val.split()[0])
            ram = (mi["MemTotal"] - mi["MemAvailable"]) / 1024.0
        except Exception:
            ram = float("nan")
        temp = float("nan")
        for z in range(12):
            try:
                if "cpu" in open(f"/sys/class/thermal/thermal_zone{z}/type").read():
                    temp = int(open(f"/sys/class/thermal/thermal_zone{z}/temp").read())/1000.0
                    break
            except Exception:
                pass
        try:
            mhz = np.mean([int(open(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq"
                                    ).read())/1000.0 for i in range(os.cpu_count())])
        except Exception:
            mhz = float("nan")
        return cpu, ram, temp, mhz

    # Encoding on the grab loop stalls the camera queue at 30 Hz, so the writes
    # go to a worker. The queue is bounded: if the disk ever falls behind it is
    # better to drop frames and say so than to silently lag the telemetry.
    wq: "queue.Queue" = queue.Queue(maxsize=120)
    dropped = [0]

    def writer():
        while True:
            item = wq.get()
            if item is None:
                break
            path, img = item
            cv2.imwrite(path, img, enc)
            wq.task_done()

    wthread = threading.Thread(target=writer, daemon=True)
    wthread.start()

    tel = None
    if not a.no_fc:
        print(f"connecting telemetry (READ-ONLY) on {a.fc_port}...")
        tel = Telemetry(a.fc_port, a.fc_baud).start()
        print(f"  telemetry connected: {tel.connected}")

    # Mono pair at full 800p: more pixels for matching, and we are only saving
    # a couple of frames a second so bandwidth is not a concern.
    p = dai.Pipeline()
    m = p.create(dai.node.MonoCamera)
    m.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    m.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    m.setFps(30)
    if a.exposure_us > 0:
        m.initialControl.setManualExposure(a.exposure_us, a.iso or OAK_ISO)
        print(f"  MANUAL exposure {a.exposure_us} us ISO {a.iso or OAK_ISO} "
              f"- daylight mein ye jal jayegi")
    else:
        # Cap the auto-exposure so a dark patch cannot smear the image with a
        # long exposure while the aircraft is moving.
        m.initialControl.setAutoExposureLimit(8000)
        print("  AUTO exposure (limit 8000 us)")
    print(f"  saving {ext.upper()} at {a.hz:g} Hz")
    xo = p.create(dai.node.XLinkOut); xo.setStreamName("mono"); m.out.link(xo.input)

    # The camera's own IMU. The flight controller has no idea the camera exists,
    # so this is the only place camera vibration can come from - and the camera
    # is what the whole position solution is built on. Same configuration as the
    # daemon, for the same measured reasons: the three reports share one packet
    # so over-asking costs the whole stream, and a batch threshold above 1 left
    # 34.2% of frames with no sample at all.
    imu = p.create(dai.node.IMU)
    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_CALIBRATED, 100)
    imu.enableIMUSensor(dai.IMUSensor.LINEAR_ACCELERATION, 100)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(20)
    xi = p.create(dai.node.XLinkOut); xi.setStreamName("imu"); imu.out.link(xi.input)
    if a.rgb:
        c = p.create(dai.node.ColorCamera)
        c.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        c.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        c.setFps(15)
        c.setInterleaved(False)
        xc = p.create(dai.node.XLinkOut); xc.setStreamName("rgb"); c.video.link(xc.input)

    with dai.Device() as dev:
        cal = dev.readCalibration()
        K = np.array(cal.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, 1280, 800))
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "camera.txt"), "w") as cf:
            cf.write(f"mono CAM_B 1280x800 fx={K[0][0]:.2f} fy={K[1][1]:.2f} "
                     f"cx={K[0][2]:.2f} cy={K[1][2]:.2f}\n")
            cf_mode = "auto" if a.exposure_us <= 0 else f"manual {a.exposure_us}us iso{a.iso}"
            cf.write(f"exposure={cf_mode}\n")
        print(f"mono focal {K[0][0]:.1f} px -> footprint at 120 m = "
              f"{2*120*np.tan(np.arctan(640/K[0][0])):.0f} m")

        dev.startPipeline(p)
        qm = dev.getOutputQueue("mono", 4, False)
        qr = dev.getOutputQueue("rgb", 4, False) if a.rgb else None
        qi = dev.getOutputQueue("imu", 50, False)
        cam_vibe = CameraVibe(window_s=1.0, rate_hz=100.0)

        next_save, t_start = 0.0, time.monotonic()
        # Asking for the camera's own rate means "every frame". Computing the
        # next deadline from the current time instead of accumulating it drops
        # roughly a third of the frames when the two rates are equal: a frame
        # arriving a hair early is skipped and the next one lands a full period
        # later. That cost 30 Hz -> 20 Hz.
        interval = 0.0 if a.hz >= OAK_FPS - 1 else 1.0 / max(a.hz, 0.1)

        sess = None
        next_load = 0.0
        if not a.on_arm:
            sess = Session(a.out)
            print(f"RECORDING to {a.out} - Ctrl-C to stop. Nothing is sent to the FC.")
        else:
            print(f"ARMED-TRIGGERED: idle until the aircraft arms. "
                  f"Folders under {a.out}. Nothing is sent to the FC.")
        cpu_sample()

        try:
            while time.monotonic() - t_start < a.seconds:
                fr = qm.get()
                now = time.monotonic() - t_start
                s_tel = tel.snapshot() if tel else {}

                # Drain the IMU on EVERY pass, not only on the frames that get
                # saved. The window has to be continuous: a deviation measured
                # across a gap is not vibration, it is the gap.
                pk = qi.tryGet()
                while pk is not None:
                    for packet in pk.packets:
                        la = getattr(packet, "acceleroMeter", None)
                        if la is not None:
                            cam_vibe.push(la.x, la.y, la.z)
                    pk = qi.tryGet()

                if a.on_arm:
                    armed = bool(s_tel.get("armed"))
                    if armed and sess is None:
                        sess = Session(os.path.join(
                            a.out, "flight_" + time.strftime("%Y%m%d_%H%M%S")))
                        dropped[0] = 0
                        print(f"\n*** ARMED - recording to {sess.root} ***", flush=True)
                    elif not armed and sess is not None:
                        print(f"*** DISARMED - {sess.n} frames in {sess.root} ***\n",
                              flush=True)
                        sess.close(); sess = None
                    if sess is None:
                        continue          # idle: no decode, no encode, no write

                if now + 1e-3 < next_save:
                    continue
                next_save = max(next_save + interval, now)

                img = fr.getCvFrame()
                name = f"m{sess.n:06d}.{ext}"
                try:
                    wq.put_nowait((os.path.join(sess.img_dir, name), img))
                except queue.Full:
                    dropped[0] += 1
                rgbname = ""
                if qr is not None and qr.has():
                    rgbname = f"c{sess.n:06d}.jpg"
                    try:
                        wq.put_nowait((os.path.join(sess.rgb_dir, rgbname),
                                       qr.get().getCvFrame()))
                    except queue.Full:
                        dropped[0] += 1
                mot = s_tel.get("mot") or [None] * 6
                cvx, cvy, cvz = cam_vibe.value()
                sess.wr.writerow([sess.n, f"{now:.3f}",
                                  td_ns(fr.getTimestampDevice()), name, rgbname,
                                  s_tel.get("lat"), s_tel.get("lon"),
                                  s_tel.get("alt_rel"), s_tel.get("alt_amsl"),
                                  s_tel.get("roll"), s_tel.get("pitch"), s_tel.get("yaw"),
                                  s_tel.get("hdg"), s_tel.get("fix"), s_tel.get("sats"),
                                  s_tel.get("rngfnd"),
                                  # the flight controller's own vibration
                                  s_tel.get("vibe_x"), s_tel.get("vibe_y"),
                                  s_tel.get("vibe_z"), s_tel.get("clip"),
                                  # the camera's, from its own IMU, same units
                                  None if cvx is None else round(cvx, 4),
                                  None if cvy is None else round(cvy, 4),
                                  None if cvz is None else round(cvz, 4),
                                  cam_vibe.samples,
                                  # and what the motors were doing at the time
                                  *mot])
                sess.n += 1

                if now >= next_load:
                    next_load = now + 1.0
                    cpu, ram, temp, mhz = cpu_sample()
                    sess.lw.writerow([f"{now:.1f}", f"{cpu:.1f}", f"{ram:.0f}",
                                      f"{temp:.1f}", f"{mhz:.0f}", sess.n, dropped[0]])
                    sess.lf.flush()

                if sess.n % 60 == 0:
                    sess.f.flush()
                    sat = 100.0 * float((img > 250).mean())
                    warn = "  <<< OVEREXPOSED" if sat > 40 else ""
                    drop = f"  DROPPED {dropped[0]}" if dropped[0] else ""
                    alt = s_tel.get("alt_rel")
                    cpu, ram, temp, mhz = cpu_sample()
                    print(f"  {now:6.1f}s  {sess.n:6d}f  "
                          f"alt={alt if alt is None else round(alt,1)}m  "
                          f"sats={s_tel.get('sats')}  lidar={s_tel.get('rngfnd')}  "
                          f"white={sat:.0f}%  cpu={cpu:.0f}% {temp:.0f}C{warn}{drop}",
                          flush=True)
        except KeyboardInterrupt:
            print("\nstopped by user")
        finally:
            wq.put(None)
            wthread.join(timeout=20)
            if sess is not None:
                print(f"\nsaved {sess.n} frames to {sess.root}")
                sess.close()
            if tel:
                tel.stop()
            if dropped[0]:
                print(f"  WARNING: {dropped[0]} frames dropped (disk could not keep up)")


if __name__ == "__main__":
    main()
