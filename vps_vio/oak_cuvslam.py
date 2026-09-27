#!/usr/bin/env python3
"""OAK-D Pro + cuVSLAM — stereo-inertial VIO bring-up and measurement.

Part 1 of the VPS upgrade: get real 6-DOF pose out of the OAK-D using cuVSLAM
on the Jetson GPU, and measure whether it is good enough to fly on.

Adapted from NVIDIA's official examples/oak-d/run_stereo.py, with two changes:
  * DepthAI **v2** API (this box has 2.33; the official example targets v3).
  * **Inertial** mode (stereo + IMU), not stereo-only.

Run (OAK-D must be free):
    vps_vio/.venv/bin/python vps_vio/oak_cuvslam.py --seconds 60

Outputs pose rate, tracking state, camera-IMU timing and frame drops — the
Phase-0 exit criteria. Writes a CSV of poses for the axis-mapping measurement.
"""
from __future__ import annotations
import argparse, csv, os, time
from collections import deque

import numpy as np
import depthai as dai
import cuvslam as vslam
from scipy.spatial.transform import Rotation

W, H = 640, 400
CM_TO_M = 100.0

# --- IMU noise model -------------------------------------------------------
# TUNED BY MEASUREMENT (sweep_imu.py, 27 Sep). Datasheet-class values (the x1
# figures below divided by 10) made cuVSLAM over-trust this BNO086 and diverge:
# a recorded 0.5 m move read 5.07 m and never returned. Declaring the IMU 10x
# noisier matches stereo-only tracking almost exactly (0.708 vs 0.682 maxdev,
# 0.033 vs 0.020 return error). Consistent with the BNO086's jittery
# timestamping (dt std 2.15 ms at 189 Hz), which the noise model must absorb.
# x100 also works; x10 gave the best return error. Re-check with an Allan
# variance run if the IMU or its rate changes.
_IMU_NOISE_SCALE      = 10.0
IMU_ACC_NOISE_DENSITY = 2.8e-3 * _IMU_NOISE_SCALE   # m/s^2/sqrt(Hz)
IMU_ACC_RANDOM_WALK   = 8.6e-4 * _IMU_NOISE_SCALE   # m/s^3/sqrt(Hz)
IMU_GYR_NOISE_DENSITY = 1.6e-4 * _IMU_NOISE_SCALE   # rad/s/sqrt(Hz)
IMU_GYR_RANDOM_WALK   = 2.2e-5 * _IMU_NOISE_SCALE   # rad/s^2/sqrt(Hz)


def transform_to_pose(mat) -> vslam.Pose:
    """OAK 4x4 extrinsics (translation in cm) -> cuVSLAM Pose (metres, quat xyzw)."""
    m = np.asarray(mat, dtype=float)
    quat = Rotation.from_matrix(m[:3, :3]).as_quat()      # x, y, z, w
    return vslam.Pose(rotation=quat, translation=m[:3, 3] / CM_TO_M)


def build_rig(calib, imu_hz: int, border: int):
    """Build the cuVSLAM Rig (2 cameras + 1 IMU) from the OAK's own calibration."""
    cams = []
    sockets = [dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C]
    for i, sock in enumerate(sockets):
        K = np.array(calib.getCameraIntrinsics(sock, W, H))
        cam = vslam.Camera()
        cam.size = (W, H)
        cam.focal = (K[0][0], K[1][1])
        cam.principal = (K[0][2], K[1][2])
        # Polynomial (rational) model, first 8 coefficients — same as NVIDIA's OAK example
        cam.distortion = vslam.Distortion(
            vslam.Distortion.Model.Polynomial,
            list(calib.getDistortionCoefficients(sock))[:8],
        )
        # left camera is the rig origin; right is placed by the stereo extrinsics
        if i == 0:
            cam.rig_from_camera = vslam.Pose(rotation=[0, 0, 0, 1], translation=[0, 0, 0])
        else:
            cam.rig_from_camera = transform_to_pose(
                calib.getCameraExtrinsics(dai.CameraBoardSocket.CAM_C,
                                          dai.CameraBoardSocket.CAM_B))
        # mask distorted borders so cuVSLAM does not pick unreliable edge features
        cam.border_top = cam.border_bottom = cam.border_left = cam.border_right = border
        cams.append(cam)

    imu = vslam.ImuCalibration()
    imu.rig_from_imu = transform_to_pose(
        calib.getImuToCameraExtrinsics(dai.CameraBoardSocket.CAM_B))
    imu.frequency = float(imu_hz)
    imu.accelerometer_noise_density = IMU_ACC_NOISE_DENSITY
    imu.accelerometer_random_walk = IMU_ACC_RANDOM_WALK
    imu.gyroscope_noise_density = IMU_GYR_NOISE_DENSITY
    imu.gyroscope_random_walk = IMU_GYR_RANDOM_WALK

    rig = vslam.Rig()
    rig.cameras = cams
    rig.imus = [imu]
    return rig


def build_pipeline(fps: int, imu_hz: int, exposure_us: int, iso: int):
    p = dai.Pipeline()
    outs = {}
    for name, sock in (("left", dai.CameraBoardSocket.CAM_B),
                       ("right", dai.CameraBoardSocket.CAM_C)):
        m = p.create(dai.node.MonoCamera)
        m.setBoardSocket(sock)
        m.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        m.setFps(fps)
        # Fixed exposure: auto-exposure flicker and long exposures (motion blur)
        # are the main killers of feature tracking on a moving aircraft.
        m.initialControl.setManualExposure(exposure_us, iso)
        x = p.create(dai.node.XLinkOut); x.setStreamName(name)
        m.out.link(x.input); outs[name] = x

    imu = p.create(dai.node.IMU)
    imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], imu_hz)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(20)
    xi = p.create(dai.node.XLinkOut); xi.setStreamName("imu")
    imu.out.link(xi.input)
    return p


def td_ns(td) -> int:
    """depthai timedelta -> integer nanoseconds (device clock)."""
    return int(td.total_seconds() * 1e9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--imu-hz", type=int, default=200)
    ap.add_argument("--exposure-us", type=int, default=4000, help="<=8000 to avoid motion blur")
    ap.add_argument("--iso", type=int, default=400)
    ap.add_argument("--border", type=int, default=24)
    ap.add_argument("--mode", choices=["inertial","stereo"], default="inertial",
                    help="stereo = visual only (no IMU), to isolate IMU problems")
    ap.add_argument("--still-s", type=float, default=0.0,
                    help="hold still this long at start so inertial init can estimate gravity/bias")
    ap.add_argument("--cue", type=int, default=0, help="second at which to prompt the 1 m move")
    ap.add_argument("--csv", default=os.path.join(os.path.dirname(__file__), "logs", "vio_pose.csv"))
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.csv), exist_ok=True)
    print(f"cuVSLAM {vslam.get_version()[0]}  |  warming up GPU...")
    vslam.warm_up_gpu()

    with dai.Device() as dev:
        print(f"device: {dev.getDeviceName()}  usb={dev.getUsbSpeed()}  imu={dev.getConnectedIMU()}")
        calib = dev.readCalibration()
        rig = build_rig(calib, a.imu_hz, a.border)

        mode = (vslam.Tracker.OdometryMode.Inertial if a.mode == "inertial"
                else vslam.Tracker.OdometryMode.Multicamera)
        cfg = vslam.Tracker.OdometryConfig(
            odometry_mode=mode,
            use_gpu=True,                    # run on the Orin GPU, not CPU
            rectified_stereo_camera=False,   # we feed raw (unrectified) mono + distortion
            async_sba=False,
            use_motion_model=True,
            use_denoising=False,
        )
        tracker = vslam.Tracker(rig, cfg)
        print(f"tracker ready (mode={a.mode}, GPU)")

        dev.startPipeline(build_pipeline(a.fps, a.imu_hz, a.exposure_us, a.iso))
        qL = dev.getOutputQueue("left", 8, False)
        qR = dev.getOutputQueue("right", 8, False)
        qI = dev.getOutputQueue("imu", 200, False)

        pend_imu = deque()
        last_imu_ts = None
        bufL, bufR = {}, {}
        n_pose = n_track = n_drop = 0
        states = {}
        lat_sum = 0.0
        t0 = time.monotonic()
        f = open(a.csv, "w", newline="")
        wr = csv.writer(f)
        wr.writerow(["t_s", "img_ts_ns", "state", "x", "y", "z", "qx", "qy", "qz", "qw"])

        try:
            while time.monotonic() - t0 < a.seconds:
                # ---- drain IMU
                while qI.has():
                    for pk in qI.get().packets:
                        acc, gyr = pk.acceleroMeter, pk.gyroscope
                        pend_imu.append((td_ns(acc.getTimestampDevice()),
                                         [acc.x, acc.y, acc.z],
                                         [gyr.x, gyr.y, gyr.z]))
                # ---- collect stereo, pair by sequence number (OAK mono pair is HW-synced)
                while qL.has(): fr = qL.get(); bufL[fr.getSequenceNum()] = fr
                while qR.has(): fr = qR.get(); bufR[fr.getSequenceNum()] = fr
                common = sorted(set(bufL) & set(bufR))
                if not common:
                    time.sleep(0.001); continue

                for seq in common:
                    fl, fr_ = bufL.pop(seq), bufR.pop(seq)
                    ts = td_ns(fl.getTimestampDevice())
                    dt_lr = abs(ts - td_ns(fr_.getTimestampDevice())) / 1e6
                    if dt_lr > 5.0:
                        n_drop += 1; continue          # stereo pair not synced

                    # IMU must be registered up to the image timestamp, in order
                    while a.mode == "inertial" and pend_imu and pend_imu[0][0] <= ts:
                        its, la, av = pend_imu.popleft()
                        if last_imu_ts is not None and its < last_imu_ts:
                            continue
                        m = vslam.ImuMeasurement()
                        m.timestamp_ns = its
                        m.linear_accelerations = la
                        m.angular_velocities = av
                        tracker.register_imu_measurement(0, m)
                        last_imu_ts = its

                    t_call = time.monotonic()
                    est, _ = tracker.track(ts, [fl.getCvFrame(), fr_.getCvFrame()])
                    n_track += 1
                    lat_sum += time.monotonic() - t_call      # track() compute time
                    p = None
                    if est is not None:
                        wfr = getattr(est, "world_from_rig", None)
                        p = getattr(wfr, "pose", wfr)          # PoseWithCovariance or Pose
                    st = "valid" if p is not None and np.all(np.isfinite(p.translation)) else "invalid"
                    states[st] = states.get(st, 0) + 1
                    if st == "valid":
                        n_pose += 1
                        wr.writerow([f"{time.monotonic()-t0:.4f}", ts, st,
                                     f"{p.translation[0]:.5f}", f"{p.translation[1]:.5f}",
                                     f"{p.translation[2]:.5f}",
                                     f"{p.rotation[0]:.6f}", f"{p.rotation[1]:.6f}",
                                     f"{p.rotation[2]:.6f}", f"{p.rotation[3]:.6f}"])
                        el_now = time.monotonic() - t0
                        if a.cue and int(el_now) != int(el_now - 0.04):
                            sec = int(el_now)
                            if sec == a.cue:            print("\n  >>> ABHI 1 METER LE JAO <<<\n")
                            elif sec == a.cue + 8:      print("\n  >>> AB WAPAS LE AAO <<<\n")
                            elif sec == a.cue + 16:     print("\n  >>> RUK JAO, STHIR <<<\n")
                        if n_pose % 30 == 0:
                            el = time.monotonic() - t0
                            print(f"  {el:5.1f}s  poses={n_pose:5d} ({n_pose/el:4.1f} Hz)  "
                                  f"pos=({p.translation[0]:+.2f},{p.translation[1]:+.2f},"
                                  f"{p.translation[2]:+.2f})  imu_q={len(pend_imu)}  state={st}")
                # keep buffers from growing if one side stalls
                for b in (bufL, bufR):
                    while len(b) > 20: b.pop(sorted(b)[0]); n_drop += 1
        finally:
            f.close()
            el = max(time.monotonic() - t0, 1e-6)
            print("\n---- Phase 0 result ----")
            print(f"track() calls : {n_track}  ({n_track/el:.1f} Hz)")
            print(f"poses         : {n_pose}  ({n_pose/el:.1f} Hz)   target >= 20 Hz")
            if n_track:
                print(f"track() time  : {lat_sum/n_track*1e3:.1f} ms mean (GPU compute per frame)")
            print(f"dropped/unsync: {n_drop}")
            print(f"states        : {states}")
            print(f"csv           : {a.csv}")


if __name__ == "__main__":
    main()
