#!/usr/bin/env python3
"""Find the correct cuVSLAM IMU config by replaying one recorded MOTION sequence.

Stereo-only VIO already works on this rig (0.501 m measured against a real
0.5 m). Inertial mode diverges under motion, so something in the IMU setup is
wrong. Same method that fixed the camera extrinsics: record ONE sequence with
real motion, then replay it through candidate IMU configurations and compare.

The winner is judged on motion sanity, not just "smallest number" — a frozen
tracker reports zero and must be rejected.

Move the drone during capture: still, then ~0.5 m out, then back.
"""
from __future__ import annotations
import argparse, time
import numpy as np
import depthai as dai
import cuvslam as vslam
from scipy.spatial.transform import Rotation

W, H = 640, 400
CM_TO_M = 100.0


def capture(seconds, fps, imu_hz, exposure_us, iso):
    p = dai.Pipeline()
    for name, sock in (("left", dai.CameraBoardSocket.CAM_B),
                       ("right", dai.CameraBoardSocket.CAM_C)):
        m = p.create(dai.node.MonoCamera)
        m.setBoardSocket(sock)
        m.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        m.setFps(fps)
        m.initialControl.setManualExposure(exposure_us, iso)
        x = p.create(dai.node.XLinkOut); x.setStreamName(name); m.out.link(x.input)
    imu = p.create(dai.node.IMU)
    imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], imu_hz)
    imu.setBatchReportThreshold(1); imu.setMaxBatchReports(20)
    xi = p.create(dai.node.XLinkOut); xi.setStreamName("imu"); imu.out.link(xi.input)

    frames, imus = [], []
    with dai.Device() as dev:
        calib = dev.readCalibration()
        cal = {
            "K": {s.name: np.array(calib.getCameraIntrinsics(s, W, H))
                  for s in (dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C)},
            "D": {s.name: list(calib.getDistortionCoefficients(s))
                  for s in (dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C)},
            "T_CB": np.array(calib.getCameraExtrinsics(dai.CameraBoardSocket.CAM_C,
                                                       dai.CameraBoardSocket.CAM_B)),
            "T_imu": np.array(calib.getImuToCameraExtrinsics(dai.CameraBoardSocket.CAM_B)),
        }
        dev.startPipeline(p)
        qL, qR = dev.getOutputQueue("left", 8, False), dev.getOutputQueue("right", 8, False)
        qI = dev.getOutputQueue("imu", 300, False)
        bufL, bufR = {}, {}
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            while qI.has():
                for pk in qI.get().packets:
                    a, g = pk.acceleroMeter, pk.gyroscope
                    imus.append((int(a.getTimestampDevice().total_seconds() * 1e9),
                                 [a.x, a.y, a.z], [g.x, g.y, g.z]))
            while qL.has():
                f = qL.get(); bufL[f.getSequenceNum()] = f
            while qR.has():
                f = qR.get(); bufR[f.getSequenceNum()] = f
            for seq in sorted(set(bufL) & set(bufR)):
                fl, fr = bufL.pop(seq), bufR.pop(seq)
                frames.append((int(fl.getTimestampDevice().total_seconds() * 1e9),
                               fl.getCvFrame(), fr.getCvFrame()))
            time.sleep(0.002)
    imus.sort(key=lambda s: s[0])
    return frames, imus, cal


def to_pose(mat):
    m = np.asarray(mat, float)
    return vslam.Pose(rotation=Rotation.from_matrix(m[:3, :3]).as_quat(),
                      translation=m[:3, 3] / CM_TO_M)


def build(cal, imu_pose_mat, noise_scale, imu_hz, border=24):
    cams = []
    for i, sname in enumerate(("CAM_B", "CAM_C")):
        K = cal["K"][sname]
        c = vslam.Camera()
        c.size = (W, H); c.focal = (K[0][0], K[1][1]); c.principal = (K[0][2], K[1][2])
        c.distortion = vslam.Distortion(vslam.Distortion.Model.Polynomial, cal["D"][sname][:8])
        c.rig_from_camera = (vslam.Pose(rotation=[0, 0, 0, 1], translation=[0, 0, 0])
                             if i == 0 else to_pose(cal["T_CB"]))
        c.border_top = c.border_bottom = c.border_left = c.border_right = border
        cams.append(c)
    im = vslam.ImuCalibration()
    im.rig_from_imu = to_pose(imu_pose_mat)
    im.frequency = float(imu_hz)
    im.accelerometer_noise_density = 2.8e-3 * noise_scale
    im.accelerometer_random_walk = 8.6e-4 * noise_scale
    im.gyroscope_noise_density = 1.6e-4 * noise_scale
    im.gyroscope_random_walk = 2.2e-5 * noise_scale
    r = vslam.Rig(); r.cameras = cams; r.imus = [im]
    return r


def run(rig, frames, imus, inertial=True):
    cfg = vslam.Tracker.OdometryConfig(
        odometry_mode=(vslam.Tracker.OdometryMode.Inertial if inertial
                       else vslam.Tracker.OdometryMode.Multicamera),
        use_gpu=True, rectified_stereo_camera=False,
        async_sba=False, use_motion_model=True, use_denoising=False)
    tr = vslam.Tracker(rig, cfg)
    k, last = 0, None
    pts, bad = [], 0
    for ts, l, r in frames:
        if inertial:
            while k < len(imus) and imus[k][0] <= ts:
                its, la, av = imus[k]; k += 1
                if last is not None and its < last:
                    continue
                m = vslam.ImuMeasurement()
                m.timestamp_ns = its; m.linear_accelerations = la; m.angular_velocities = av
                tr.register_imu_measurement(0, m); last = its
        est, _ = tr.track(ts, [l, r])
        p = getattr(est, "world_from_rig", None) if est is not None else None
        p = getattr(p, "pose", p)
        if p is None or not np.all(np.isfinite(p.translation)):
            bad += 1; continue
        pts.append(np.array(p.translation))
    if len(pts) < 10:
        return None
    P = np.stack(pts)
    steps = np.linalg.norm(np.diff(P, axis=0), axis=1)
    return dict(maxdev=float(np.linalg.norm(P - P[0], axis=1).max()),
                final=float(np.linalg.norm(P[-1] - P[0])),
                path=float(steps.sum()), maxstep=float(steps.max()),
                frozen=bool(steps.max() < 1e-6), bad=bad, n=len(P))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=22)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--imu-hz", type=int, default=200)
    ap.add_argument("--exposure-us", type=int, default=4000)
    ap.add_argument("--iso", type=int, default=400)
    a = ap.parse_args()

    vslam.warm_up_gpu()
    print(f"RECORDING {a.seconds:.0f}s — still ~6s, then move ~0.5 m out and back...")
    frames, imus, cal = capture(a.seconds, a.fps, a.imu_hz, a.exposure_us, a.iso)
    print(f"captured {len(frames)} stereo pairs, {len(imus)} IMU samples\n")

    T = cal["T_imu"]
    T_inv = np.linalg.inv(np.asarray(T, float))

    trials = [("stereo-only (reference)", None, 1.0, False)]
    for label, mat in (("imu T(as-is)", T), ("imu T(inverted)", T_inv)):
        for ns in (1.0, 10.0, 100.0):
            trials.append((f"{label} noise x{ns:g}", mat, ns, True))

    print(f"{'config':30s} {'maxdev':>9s} {'final':>9s} {'path':>9s} {'maxstep':>9s} {'bad':>4s}")
    print("-" * 76)
    out = []
    for label, mat, ns, inertial in trials:
        try:
            rig = (build(cal, T, 1.0, a.imu_hz) if mat is None
                   else build(cal, mat, ns, a.imu_hz))
            if not inertial:
                rig.imus = []
            r = run(rig, frames, imus, inertial)
        except Exception as e:
            print(f"{label:30s}  ERROR: {str(e)[:32]}"); continue
        if r is None:
            print(f"{label:30s}  no valid poses"); continue
        flag = "  FROZEN" if r["frozen"] else ""
        print(f"{label:30s} {r['maxdev']:9.3f} {r['final']:9.3f} {r['path']:9.3f} "
              f"{r['maxstep']:9.4f} {r['bad']:4d}{flag}")
        if not r["frozen"]:
            out.append((r["maxdev"], label, r))

    print("\nExpect: maxdev ~0.5 m (the real move), final ~0 m (returned), maxstep small.")
    print("Reject anything with metres-scale maxstep or a huge maxdev — that is divergence.")


if __name__ == "__main__":
    main()
