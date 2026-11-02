#!/usr/bin/env python3
"""What ROTATION_VECTOR, GRAVITY and MAGNETOMETER_CALIBRATED actually give.

The pipeline asks the OAK-D's BNO086 for GYROSCOPE_CALIBRATED,
LINEAR_ACCELERATION and GAME_ROTATION_VECTOR. GAME_ROTATION_VECTOR is the
6-axis fusion - gyro and accelerometer only - so its heading has no absolute
reference and drifts. ROTATION_VECTOR is the 9-axis fusion, magnetometer
included, so its heading is absolute.

If that heading is steady, the camera can say which way it is pointing by
itself, and the difference against the FC's own yaw IS the camera mount's
yaw offset - measured directly, with no flight, no GPS and no map.

GRAVITY gives the tilt vector without integrating anything.
MAGNETOMETER_CALIBRATED is the raw heading reference underneath both.

Read-only. Opens the camera, reads for a while, prints, exits.
"""
import math
import sys
import time

import numpy as np
import depthai as dai


def quat_to_euler(i, j, k, r):
    """roll, pitch, yaw in degrees from the sensor's quaternion."""
    sinr = 2.0 * (r * i + j * k)
    cosr = 1.0 - 2.0 * (i * i + j * j)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (r * j - k * i)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    siny = 2.0 * (r * k + i * j)
    cosy = 1.0 - 2.0 * (j * j + k * k)
    yaw = math.atan2(siny, cosy)
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


GAME = False


def main(seconds=45.0, rate=100):
    p = dai.Pipeline()
    imu = p.create(dai.node.IMU)
    # MUTUALLY EXCLUSIVE on the BNO086: the device refuses
    # ROTATION_VECTOR and GAME_ROTATION_VECTOR together -
    # "IMU invalid settings!: ... are mutually exclusive". So absolute
    # heading costs us the 6-axis one, which is a real trade and not a
    # setting. Which to ask for is chosen by the caller.
    want = [dai.IMUSensor.GRAVITY, dai.IMUSensor.MAGNETOMETER_CALIBRATED]
    want.append(dai.IMUSensor.GAME_ROTATION_VECTOR if GAME
                else dai.IMUSensor.ROTATION_VECTOR)
    for s in want:
        imu.enableIMUSensor(s, rate)
    imu.setBatchReportThreshold(5)
    imu.setMaxBatchReports(20)
    out = p.create(dai.node.XLinkOut)
    out.setStreamName("imu")
    imu.out.link(out.input)

    rot, game, grav, mag = [], [], [], []
    with dai.Device(p) as dev:
        print(f"camera khula: {dev.getMxId()}", flush=True)
        q = dev.getOutputQueue("imu", 50, False)
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            pk = q.tryGet()
            if pk is None:
                time.sleep(0.002)
                continue
            for packet in pk.packets:
                rv = getattr(packet, "rotationVector", None)
                if rv is not None and rv.getTimestampDevice().total_seconds() > 0:
                    rot.append(quat_to_euler(rv.i, rv.j, rv.k, rv.real)
                               + (getattr(rv, "accuracy", float("nan")),))
                gv = getattr(packet, "gameRotationVector", None)
                if gv is not None and gv.getTimestampDevice().total_seconds() > 0:
                    game.append(quat_to_euler(gv.i, gv.j, gv.k, gv.real))
                # GRAVITY arrives in the ACCELEROMETER slot, like
                # LINEAR_ACCELERATION does - every accelerometer-family
                # report lands in acceleroMeter and which one it is was
                # decided by enableIMUSensor.
                g = getattr(packet, "acceleroMeter", None)
                if g is not None and g.getTimestampDevice().total_seconds() > 0:
                    grav.append((g.x, g.y, g.z))
                mg = getattr(packet, "magneticField", None)
                if mg is not None and mg.getTimestampDevice().total_seconds() > 0:
                    mag.append((mg.x, mg.y, mg.z))

    def circ(a):
        a = np.radians(np.array(a))
        return math.degrees(math.atan2(np.mean(np.sin(a)), np.mean(np.cos(a))))

    print(f"\nsamples: ROTATION_VECTOR {len(rot)}, GAME {len(game)}, "
          f"GRAVITY {len(grav)}, MAG {len(mag)}")

    if rot:
        R = np.array(rot, float)
        print("\nROTATION_VECTOR (9-axis, absolute heading)")
        print(f"   roll  {np.median(R[:,0]):+8.2f} deg  (spread {R[:,0].std():.3f})")
        print(f"   pitch {np.median(R[:,1]):+8.2f} deg  (spread {R[:,1].std():.3f})")
        print(f"   YAW   {circ(R[:,2]):+8.2f} deg  (spread {R[:,2].std():.3f})")
        d = R[-1, 2] - R[0, 2]
        print(f"   yaw drift over the run: {d:+.2f} deg")
        if not np.all(np.isnan(R[:, 3])):
            print(f"   accuracy flag: median {np.nanmedian(R[:,3]):.0f} "
                  f"(0 unreliable, 3 high)")
    if game:
        G = np.array(game, float)
        print("\nGAME_ROTATION_VECTOR (6-axis, what we use now)")
        print(f"   roll  {np.median(G[:,0]):+8.2f}  pitch {np.median(G[:,1]):+8.2f}"
              f"  YAW {circ(G[:,2]):+8.2f}")
        print(f"   yaw drift over the run: {G[-1,2]-G[0,2]:+.2f} deg")
    if grav:
        V = np.array(grav, float)
        n = np.linalg.norm(V, axis=1)
        tilt = np.degrees(np.arccos(np.clip(np.abs(V[:, 2]) / np.maximum(n, 1e-9),
                                            -1, 1)))
        print("\nGRAVITY")
        print(f"   vector  x {np.median(V[:,0]):+.3f}  y {np.median(V[:,1]):+.3f}"
              f"  z {np.median(V[:,2]):+.3f}  |g| {np.median(n):.3f} m/s2")
        print(f"   tilt from vertical: {np.median(tilt):.2f} deg "
              f"(spread {tilt.std():.3f})")
    if mag:
        M = np.array(mag, float)
        n = np.linalg.norm(M, axis=1)
        print("\nMAGNETOMETER_CALIBRATED")
        print(f"   x {np.median(M[:,0]):+.1f}  y {np.median(M[:,1]):+.1f}"
              f"  z {np.median(M[:,2]):+.1f} uT   |B| {np.median(n):.1f} uT"
              f" (spread {n.std():.2f})")
        print("   earth's field is 25-65 uT; far outside that means iron or "
              "current nearby")
    return 0


if __name__ == "__main__":
    GAME = "--game" in sys.argv
    args=[a for a in sys.argv[1:] if not a.startswith("--")]
    sys.exit(main(float(args[0]) if args else 45.0))
