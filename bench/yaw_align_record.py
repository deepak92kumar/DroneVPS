#!/usr/bin/env python3
"""Record the camera's absolute heading and the FC's yaw side by side.

The camera mount was rebuilt after 27 Oct, so camera_yaw_offset_deg = -1.95
is the OLD mount's number. Flow-direction-against-GPS could not measure the
new one: the estimate moved from +4.3 to +1.7 on one flight and +4.4 to
+14.7 on another as the sample filter changed, so it was picking up
motion-correlated flow noise rather than a constant rotation.

This measures it directly instead. Both sensors are read at once while the
aircraft sits still, the pilot turns it about 90 degrees, and the same two
are read again. The CHANGE in each is what matters:

  same magnitude, same sign   -> the frames agree, and the constant
                                 difference is the mount offset
  same magnitude, opposite    -> the yaw axis is inverted, which is what the
                                 recorded flight logs already suggest
  different magnitude         -> a scale or axis-mixing problem, and nothing
                                 downstream should trust this heading

ROTATION_VECTOR is used because it is absolute. GRAVITY and the magnetometer
come along because the turn tests them too - the field should keep its
magnitude through a rotation and only change direction.

Read-only: it opens the camera and listens to the router. It writes one CSV.
"""
import csv
import math
import sys
import threading
import time

import depthai as dai
from pymavlink import mavutil

OUT = '/data/DRONE/flight_logs/analysis/yaw_align.csv'
_fc = {"yaw": None, "roll": None, "pitch": None}
_stop = threading.Event()


def euler(i, j, k, r):
    sr = 2.0 * (r * i + j * k)
    cr = 1.0 - 2.0 * (i * i + j * j)
    sp = max(-1.0, min(1.0, 2.0 * (r * j - k * i)))
    sy = 2.0 * (r * k + i * j)
    cy = 1.0 - 2.0 * (j * j + k * k)
    return (math.degrees(math.atan2(sr, cr)), math.degrees(math.asin(sp)),
            math.degrees(math.atan2(sy, cy)))


def fc_thread(endpoint):
    """The FC's own attitude, through the router. Announce first or the
    router never learns this client - the same thing that kept
    drone_watchdog's armed check returning None for its whole life."""
    try:
        m = mavutil.mavlink_connection(endpoint, source_system=255,
                                       source_component=193)
        m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                             mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        m.mav.request_data_stream_send(
            1, 1, mavutil.mavlink.MAV_DATA_STREAM_EXTRA1, 20, 1)
        while not _stop.is_set():
            msg = m.recv_match(type='ATTITUDE', blocking=True, timeout=1.0)
            if msg is not None:
                _fc["yaw"] = math.degrees(msg.yaw)
                _fc["roll"] = math.degrees(msg.roll)
                _fc["pitch"] = math.degrees(msg.pitch)
        m.close()
    except Exception as e:
        print(f"FC thread: {type(e).__name__}: {e}", flush=True)


def main(seconds):
    t = threading.Thread(target=fc_thread,
                         args=("udpout:127.0.0.1:14650",), daemon=True)
    t.start()
    p = dai.Pipeline()
    imu = p.create(dai.node.IMU)
    for s in (dai.IMUSensor.ROTATION_VECTOR, dai.IMUSensor.GRAVITY,
              dai.IMUSensor.MAGNETOMETER_CALIBRATED):
        imu.enableIMUSensor(s, 100)
    imu.setBatchReportThreshold(5)
    imu.setMaxBatchReports(20)
    o = p.create(dai.node.XLinkOut)
    o.setStreamName("imu")
    imu.out.link(o.input)

    fh = open(OUT, 'w', newline='')
    w = csv.writer(fh)
    w.writerow(['t', 'cam_roll', 'cam_pitch', 'cam_yaw', 'cam_acc',
                'grav_x', 'grav_y', 'grav_z', 'mag_x', 'mag_y', 'mag_z',
                'fc_roll', 'fc_pitch', 'fc_yaw'])
    g = [None] * 3
    mg = [None] * 3
    with dai.Device(p) as dev:
        print(f"recording to {OUT} for {seconds:.0f}s - GHUMAO JAB CHAHO",
              flush=True)
        q = dev.getOutputQueue("imu", 50, False)
        t0 = time.monotonic()
        last = 0.0
        while time.monotonic() - t0 < seconds:
            pk = q.tryGet()
            if pk is None:
                time.sleep(0.002)
                continue
            for packet in pk.packets:
                a = getattr(packet, "acceleroMeter", None)
                if a is not None:
                    g = [a.x, a.y, a.z]
                m_ = getattr(packet, "magneticField", None)
                if m_ is not None:
                    mg = [m_.x, m_.y, m_.z]
                rv = getattr(packet, "rotationVector", None)
                if rv is None:
                    continue
                now = time.monotonic() - t0
                if now - last < 0.05:        # 20 Hz is plenty
                    continue
                last = now
                r, pi, y = euler(rv.i, rv.j, rv.k, rv.real)
                w.writerow([f"{now:.3f}", f"{r:.3f}", f"{pi:.3f}", f"{y:.3f}",
                            getattr(rv, "accuracy", ""),
                            *[("" if x is None else f"{x:.4f}") for x in g],
                            *[("" if x is None else f"{x:.2f}") for x in mg],
                            "" if _fc["roll"] is None else f"{_fc['roll']:.3f}",
                            "" if _fc["pitch"] is None else f"{_fc['pitch']:.3f}",
                            "" if _fc["yaw"] is None else f"{_fc['yaw']:.3f}"])
    _stop.set()
    fh.close()
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(float(sys.argv[1]) if len(sys.argv) > 1 else 180.0))
