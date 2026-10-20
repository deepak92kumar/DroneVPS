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


class CamImu:
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
        # The latest raw values as well as the derived vibration. A vibration
        # figure with no raw behind it cannot be re-derived later if the way it
        # is computed ever changes, and this project has already lost one set
        # of measurements it could not reconstruct.
        self.ax = self.ay = self.az = None
        self.gx = self.gy = self.gz = None
        self.gyro_n = 0
        self._t0 = time.monotonic()

    def push(self, ax: float, ay: float, az: float) -> None:
        self.buf.append((ax, ay, az))
        self.ax, self.ay, self.az = ax, ay, az
        self.samples += 1

    def push_gyro(self, gx: float, gy: float, gz: float) -> None:
        self.gx, self.gy, self.gz = gx, gy, gz
        self.gyro_n += 1

    def rate_hz(self) -> float:
        dt = time.monotonic() - self._t0
        return self.samples / dt if dt > 0.5 else 0.0

    def value(self):
        """(x, y, z) m/s/s, or (None, None, None) before the window fills."""
        if len(self.buf) < self.n // 2:
            return (None, None, None)
        a = np.asarray(self.buf, dtype=float)
        return tuple(float(x) for x in a.std(axis=0))


class Telemetry:
    """MAVLink listener. Sends nothing that commands the aircraft.

    It is not literally silent, and 20 Oct is why. It used to transmit only a
    stream request at startup, and the router drops a UDP client it has not
    heard from in 30 seconds. The router's own log, at exactly 30 s after
    start: four of five clients gone, each having sent 1-9 frames and received
    about 3800. This was one of them.

    What that costs is not a missing log line. snapshot() keeps the last value
    of everything it has seen, so when the messages stop the dictionary does
    not empty - it FREEZES. The VPS node goes on reading roll, pitch, yaw and
    altitude out of it and they are all stale, so derotation is subtracting an
    attitude change that stopped being current minutes ago, and the altitude
    that scales every pixel into metres is whatever it was half an hour in.
    Nothing reports an error, because from here a dead link and a still
    aircraft look identical.

    So it heartbeats, to stay registered, and it tracks the age of what it
    holds so a caller can tell fresh data from frozen data.
    """

    def __init__(self, port: str, baud: int):
        self.port, self.baud = port, baud
        self.lock = threading.Lock()
        # The TF03 jumps, in both directions, and log 214 shows it on a
        # stationary aircraft: RFND spread 0.05-0.77 m with a median of 0.41.
        # cuav_mavlink_bridge had a median of 5 on its own copy; this path had
        # nothing at all. See lidar_filter.LidarFilter for why a median alone
        # is not enough.
        from lidar_filter import LidarFilter
        self.lidar = LidarFilter()
        self.state = dict(lat=None, lon=None, alt_rel=None, alt_amsl=None,
                          roll=None, pitch=None, yaw=None, fix=None, sats=None,
                          hdg=None, rngfnd=None, rngfnd_raw=None, armed=False, t=0.0,
                          gps1_alt=None,
                          # The receiver's OWN horizontal accuracy, metres.
                          # This is the origin's error, and until 20 Oct
                          # nothing carried it: every VPS position is measured
                          # from the origin, so the accuracy we declare to the
                          # flight controller is only ever as good as the point
                          # we started from. See GPS_RAW_INT below.
                          gps1_hacc=None,
                          # The raw receiver's own lat/lon, independent of the
                          # EKF. See GPS_RAW_INT below for why this is separate
                          # from lat/lon.
                          gps1_lat=None, gps1_lon=None,
                          # The FC's own accelerometer vibration, and what the
                          # motors were doing when it was measured. One without
                          # the other says nothing: vibration at idle and
                          # vibration at hover throttle are different questions.
                          vibe_x=None, vibe_y=None, vibe_z=None, clip=None,
                          mot=None,
                          # velocity the FC believes it has, which is the whole
                          # of the 54% row and the thing the bridge gets wrong
                          vx=None, vy=None, vz=None, climb=None,
                          # what the EKF thinks of its own solution. Variances
                          # above 1.0 are what an EKF failsafe is made of, and
                          # a flight that failsafed without these logged cannot
                          # be explained afterwards.
                          ekf_vel_var=None, ekf_pos_h_var=None,
                          ekf_pos_v_var=None, ekf_comp_var=None, ekf_flags=None,
                          # GPS2 is OUR injection - this is how we see whether
                          # the FC accepted it at all
                          gps2_fix=None, gps2_sats=None,
                          batt_v=None, batt_a=None, batt_pct=None,
                          throttle=None, groundspeed=None, mode=None)
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

    def _heartbeat(self):
        """Keep this client registered with the router. Commands nothing.

        MAV_TYPE_GCS / MAV_AUTOPILOT_INVALID: a heartbeat that says "a ground
        station is listening", which is what this is. It does not arm, set a
        mode or move a parameter.
        """
        try:
            from pymavlink import mavutil
            self.m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                      mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                                      0, 0, 0)
        except Exception:
            pass

    def age_s(self) -> float:
        """How long since anything arrived. Large means the state is frozen."""
        with self.lock:
            t = self.state.get("t") or 0.0
        return 1e9 if t <= 0 else max(0.0, time.time() - t)

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
        last_hb = 0.0
        while self._run:
            now = time.time()
            # Every 5 s, for the whole run. The router's client timeout is 30 s
            # and this loop can otherwise go minutes without transmitting.
            if now - last_hb > 5.0:
                last_hb = now
                self._heartbeat()
            # The stream request is also re-sent for the whole run, not only
            # until the first position. Being dropped and re-registered means
            # the FC's stream rates have to be asked for again, and asking
            # only while lat is None would never do it a second time.
            if now - last_req > 10.0:
                last_req = now
                self._request_streams()
            try:
                msg = self.m.recv_match(
                    type=["GLOBAL_POSITION_INT", "GPS_RAW_INT", "ATTITUDE",
                          "DISTANCE_SENSOR", "RANGEFINDER", "HEARTBEAT",
                          "VIBRATION", "SERVO_OUTPUT_RAW", "EKF_STATUS_REPORT",
                          "GPS2_RAW", "SYS_STATUS", "VFR_HUD", "BATTERY_STATUS"],
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
                                      hdg=msg.hdg / 100.0 if msg.hdg != 65535 else None,
                                      vx=msg.vx / 100.0, vy=msg.vy / 100.0,
                                      vz=msg.vz / 100.0)
                elif t == "EKF_STATUS_REPORT":
                    self.state.update(ekf_vel_var=msg.velocity_variance,
                                      ekf_pos_h_var=msg.pos_horiz_variance,
                                      ekf_pos_v_var=msg.pos_vert_variance,
                                      ekf_comp_var=msg.compass_variance,
                                      ekf_flags=msg.flags)
                elif t == "GPS2_RAW":
                    self.state.update(gps2_fix=msg.fix_type,
                                      gps2_sats=msg.satellites_visible)
                elif t == "VFR_HUD":
                    self.state.update(throttle=msg.throttle, climb=msg.climb,
                                      groundspeed=msg.groundspeed)
                elif t == "SYS_STATUS":
                    self.state.update(batt_v=msg.voltage_battery / 1000.0,
                                      batt_a=(msg.current_battery / 100.0
                                              if msg.current_battery != -1 else None),
                                      batt_pct=(msg.battery_remaining
                                                if msg.battery_remaining != -1 else None))
                elif t == "GPS_RAW_INT":
                    # alt as well as the fix. It is the only AMSL on this
                    # aircraft that does not come back out of the EKF, which
                    # matters because the EKF's altitude is partly OUR doing -
                    # see AmslRef in live_node.
                    # h_acc is a MAVLink 2 extension field in millimetres, and
                    # it reads UINT32_MAX when the receiver does not know - which
                    # is what the bench returns with no lock (measured 20 Oct:
                    # 4294967295 mm on every message, satellites 0). With a lock
                    # the recent logs carry 1.08 to 3.01 m. Anything outside a
                    # sane range is discarded rather than passed on as a number.
                    hacc = getattr(msg, "h_acc", None)
                    hacc_m = None
                    if hacc is not None and 0 < hacc < 100000 and msg.fix_type >= 3:
                        hacc_m = hacc / 1000.0
                    # THE RAW RECEIVER'S OWN POSITION, which is not the same
                    # thing as `lat`/`lon` above. Those come from
                    # GLOBAL_POSITION_INT, which is the EKF's FUSED output - and
                    # on a VPS flight the EKF is being driven by what we inject,
                    # so measuring ourselves against it is measuring ourselves
                    # against ourselves. It went unnoticed until 21 Oct because
                    # every flight with real travel so far had src = GPS1, where
                    # the source gate blocks our injection entirely and the EKF
                    # really was GPS-driven.
                    #
                    # GPS_RAW_INT is the receiver before any of that. It is the
                    # only independent truth this aircraft carries, so it gets
                    # its own fields and its own CSV columns.
                    lat1 = lon1 = None
                    if msg.fix_type >= 3 and msg.lat != 0:
                        lat1, lon1 = msg.lat / 1e7, msg.lon / 1e7
                    self.state.update(fix=msg.fix_type, sats=msg.satellites_visible,
                                      gps1_hacc=hacc_m,
                                      gps1_lat=lat1, gps1_lon=lon1,
                                      gps1_alt=(msg.alt / 1000.0
                                                if msg.fix_type >= 3 else None))
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
                    #
                    # ONLY THE AUTOPILOT'S OWN HEARTBEAT. Everything on this
                    # bus heartbeats every 5 s so the router does not drop it -
                    # the RC switch, the notifier, the watchdog, any tool
                    # someone runs - and every one of those carries base_mode 0,
                    # which reads as DISARMED. Taking the last heartbeat from
                    # anybody made `armed` flip false every five seconds.
                    #
                    # Measured cost on the 20 Oct 13:49 flight, the first one
                    # where anything depended on this flag: the frame gate shut
                    # once per 5 s, 846 images were saved where 980 were due
                    # (14% lost), the radio said "FRAMES OFF - DISARMED" twenty
                    # times while airborne, and one "LIDAR ZERO 3.4m - NO
                    # AUTO-DISARM" went out at 1 m altitude from a check that
                    # is only supposed to speak on the ground.
                    #
                    # A GCS-style heartbeat sets autopilot = MAV_AUTOPILOT_
                    # INVALID and sits on a component other than 1; the
                    # flight controller sets a real autopilot id on component
                    # 1. That pair is the test.
                    from pymavlink import mavutil as _mu
                    if (msg.get_srcComponent() != 1
                            or msg.autopilot == _mu.mavlink.MAV_AUTOPILOT_INVALID):
                        continue
                    self.state["armed"] = bool(
                        msg.base_mode & _mu.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                    self.state["mode"] = msg.custom_mode
                elif t == "RANGEFINDER":
                    # Logged alongside the frames so lidar height can be
                    # compared against the barometer over a whole flight.
                    # The analog scaling was wrong when this comment was first
                    # written; RNGFND1_OFFSET was recalibrated on 19 Oct from
                    # 0.607 to 0.66537 against a measured 0.26 m and the ground
                    # reading went from 4.35 m to 0.26 m. SCALING 72.61 was
                    # already right - lifting the aircraft moved the reading
                    # 1:1 - and matches the 150 ohm / 4-20 mA design of 75 m/V.
                    now = time.time()
                    self.state["rngfnd_raw"] = float(msg.distance)
                    self.state["rngfnd"] = self.lidar.push(now, msg.distance)
                elif t == "DISTANCE_SENSOR":
                    now = time.time()
                    raw = msg.current_distance / 100.0
                    self.state["rngfnd_raw"] = raw
                    self.state["rngfnd"] = self.lidar.push(now, raw)
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
    ap.add_argument("--fc-baud", type=int, default=921600,
                    help="TELEM2 flies at 921600 - SERIAL2_BAUD reads 921 "
                         "on the FC. Ignored for a udp: endpoint.")
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
                              # vibration, from both places it can be measured
                              "fc_vibe_x", "fc_vibe_y", "fc_vibe_z", "fc_clip",
                              "cam_vibe_x", "cam_vibe_y", "cam_vibe_z",
                              "cam_imu_n", "cam_imu_hz",
                              # the raw IMU the vibration was derived from, so
                              # it can be recomputed if the method ever changes
                              "cam_gyro_x", "cam_gyro_y", "cam_gyro_z",
                              "cam_accel_x", "cam_accel_y", "cam_accel_z",
                              # what the motors were doing at the time
                              "mot1", "mot2", "mot3", "mot4", "mot5", "mot6",
                              "throttle_pct",
                              # velocity - the 54% row, and the thing the
                              # bridge has been getting wrong
                              "vx", "vy", "vz", "climb", "groundspeed",
                              # what the EKF thinks of its own solution.
                              # Variances above 1.0 are what a failsafe is made
                              # of; a flight that failsafed without these
                              # logged cannot be explained afterwards.
                              "ekf_vel_var", "ekf_pos_h_var", "ekf_pos_v_var",
                              "ekf_comp_var", "ekf_flags",
                              # GPS2 is OUR injection - this is how we see
                              # whether the FC accepted it at all
                              "gps2_fix", "gps2_sats",
                              "batt_v", "batt_a", "batt_pct",
                              "mode", "armed", "dropped"])
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
        cam_imu = CamImu(window_s=1.0, rate_hz=100.0)

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
                            cam_imu.push(la.x, la.y, la.z)
                        gy = getattr(packet, "gyroscope", None)
                        if gy is not None:
                            cam_imu.push_gyro(gy.x, gy.y, gy.z)
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
                cvx, cvy, cvz = cam_imu.value()
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
                                  cam_imu.samples, round(cam_imu.rate_hz(), 1),
                                  # the raw IMU behind that vibration figure,
                                  # so it can be recomputed if the method ever
                                  # changes
                                  cam_imu.gx, cam_imu.gy, cam_imu.gz,
                                  cam_imu.ax, cam_imu.ay, cam_imu.az,
                                  # what the motors were doing at the time
                                  *mot, s_tel.get("throttle"),
                                  # velocity - the 54% row
                                  s_tel.get("vx"), s_tel.get("vy"),
                                  s_tel.get("vz"), s_tel.get("climb"),
                                  s_tel.get("groundspeed"),
                                  # what the EKF thinks of its own solution
                                  s_tel.get("ekf_vel_var"),
                                  s_tel.get("ekf_pos_h_var"),
                                  s_tel.get("ekf_pos_v_var"),
                                  s_tel.get("ekf_comp_var"),
                                  s_tel.get("ekf_flags"),
                                  # GPS2 is our own injection
                                  s_tel.get("gps2_fix"), s_tel.get("gps2_sats"),
                                  s_tel.get("batt_v"), s_tel.get("batt_a"),
                                  s_tel.get("batt_pct"),
                                  s_tel.get("mode"),
                                  int(bool(s_tel.get("armed"))), dropped[0]])
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
