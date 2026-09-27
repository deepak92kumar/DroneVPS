#!/usr/bin/env python3
"""Record a flight for offline map-matching validation. READ-ONLY on the FC.

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
import argparse, csv, os, sys, threading, time

import cv2
import numpy as np
import depthai as dai

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oak_rig  # noqa: E402


class Telemetry:
    """Read-only MAVLink listener. Never transmits anything to the FC."""

    def __init__(self, port: str, baud: int):
        self.port, self.baud = port, baud
        self.lock = threading.Lock()
        self.state = dict(lat=None, lon=None, alt_rel=None, alt_amsl=None,
                          roll=None, pitch=None, yaw=None, fix=None, sats=None,
                          hdg=None, t=0.0)
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
                    type=["GLOBAL_POSITION_INT", "GPS_RAW_INT", "ATTITUDE"],
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
    ap.add_argument("--rgb", action="store_true", help="also save the colour camera")
    ap.add_argument("--exposure-us", type=int, default=oak_rig.EXPOSURE_US)
    ap.add_argument("--iso", type=int, default=oak_rig.ISO)
    ap.add_argument("--fc-port", default="/dev/ttyTHS1")
    ap.add_argument("--fc-baud", type=int, default=115200)
    ap.add_argument("--no-fc", action="store_true", help="skip telemetry entirely")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    img_dir = os.path.join(a.out, "mono")
    os.makedirs(img_dir, exist_ok=True)
    if a.rgb:
        rgb_dir = os.path.join(a.out, "rgb")
        os.makedirs(rgb_dir, exist_ok=True)

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
    m.initialControl.setManualExposure(a.exposure_us, a.iso)
    xo = p.create(dai.node.XLinkOut); xo.setStreamName("mono"); m.out.link(xo.input)
    if a.rgb:
        c = p.create(dai.node.ColorCamera)
        c.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        c.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        c.setFps(15)
        c.setInterleaved(False)
        xc = p.create(dai.node.XLinkOut); xc.setStreamName("rgb"); c.video.link(xc.input)

    csv_path = os.path.join(a.out, "index.csv")
    f = open(csv_path, "w", newline="")
    wr = csv.writer(f)
    wr.writerow(["n", "wall_s", "dev_ts_ns", "mono_file", "rgb_file",
                 "lat", "lon", "alt_rel_m", "alt_amsl_m",
                 "roll_deg", "pitch_deg", "yaw_deg", "hdg_deg", "fix", "sats"])

    with dai.Device() as dev:
        cal = dev.readCalibration()
        K = np.array(cal.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, 1280, 800))
        with open(os.path.join(a.out, "camera.txt"), "w") as cf:
            cf.write(f"mono CAM_B 1280x800 fx={K[0][0]:.2f} fy={K[1][1]:.2f} "
                     f"cx={K[0][2]:.2f} cy={K[1][2]:.2f}\n")
            cf.write(f"exposure_us={a.exposure_us} iso={a.iso}\n")
        print(f"mono focal {K[0][0]:.1f} px -> footprint at 120 m = "
              f"{2*120*np.tan(np.arctan(640/K[0][0])):.0f} m")

        dev.startPipeline(p)
        qm = dev.getOutputQueue("mono", 4, False)
        qr = dev.getOutputQueue("rgb", 4, False) if a.rgb else None

        n, t0, next_save = 0, time.monotonic(), 0.0
        interval = 1.0 / max(a.hz, 0.1)
        print(f"RECORDING to {a.out} — Ctrl-C to stop. Nothing is sent to the FC.")
        try:
            while time.monotonic() - t0 < a.seconds:
                fr = qm.get()
                now = time.monotonic() - t0
                if now < next_save:
                    continue
                next_save = now + interval
                img = fr.getCvFrame()
                name = f"m{n:06d}.png"
                cv2.imwrite(os.path.join(img_dir, name), img)
                rgbname = ""
                if qr is not None and qr.has():
                    rgbname = f"c{n:06d}.jpg"
                    cv2.imwrite(os.path.join(rgb_dir, rgbname),
                                qr.get().getCvFrame(),
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                s = tel.snapshot() if tel else {}
                wr.writerow([n, f"{now:.3f}", oak_rig.td_ns(fr.getTimestampDevice()),
                             name, rgbname,
                             s.get("lat"), s.get("lon"), s.get("alt_rel"), s.get("alt_amsl"),
                             s.get("roll"), s.get("pitch"), s.get("yaw"), s.get("hdg"),
                             s.get("fix"), s.get("sats")])
                n += 1
                if n % 20 == 0:
                    f.flush()
                    alt = s.get("alt_rel")
                    print(f"  {now:6.1f}s  {n:5d} frames  alt={alt if alt is None else round(alt,1)} m  "
                          f"lat={s.get('lat')}  sats={s.get('sats')}")
        except KeyboardInterrupt:
            print("\nstopped by user")
        finally:
            f.close()
            if tel:
                tel.stop()
            print(f"\nsaved {n} frames to {a.out}")
            print(f"index: {csv_path}")


if __name__ == "__main__":
    main()
