#!/usr/bin/env python3
"""Measure which cuVSLAM axis corresponds to a known physical direction.

cuVSLAM's world frame is arbitrary (set at init), so we cannot measure North
directly. What we CAN measure — and what config.py actually needs — is the
fixed rotation between the VIO frame and the DRONE BODY frame (nose-forward,
right, down). The compass supplies body->NED at runtime.

Procedure, with the drone LEVEL in flight attitude and a textured scene below:
    python3 axis_map.py --dir forward   # then move the drone along its nose
    python3 axis_map.py --dir right     # then move it to its right
    python3 axis_map.py --dir up        # then lift it straight up

Each run: stay still for the settle period, then make ONE clean move of roughly
the stated distance and stop. The script reports the displacement in VIO axes
so the dominant axis and its sign can be read off.
"""
from __future__ import annotations
import argparse, time
import numpy as np
import depthai as dai
import cuvslam as vslam
from scipy.spatial.transform import Rotation

W, H = 640, 400
CM_TO_M = 100.0
IMU_SCALE = 10.0   # measured; see oak_cuvslam.py


def to_pose(mat):
    m = np.asarray(mat, float)
    return vslam.Pose(rotation=Rotation.from_matrix(m[:3, :3]).as_quat(),
                      translation=m[:3, 3] / CM_TO_M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, choices=["forward", "right", "up", "back", "left", "down"])
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--settle", type=float, default=6.0)
    ap.add_argument("--exposure-us", type=int, default=4000)
    ap.add_argument("--iso", type=int, default=800)
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()

    vslam.warm_up_gpu()
    p = dai.Pipeline()
    for name, sock in (("left", dai.CameraBoardSocket.CAM_B),
                       ("right", dai.CameraBoardSocket.CAM_C)):
        m = p.create(dai.node.MonoCamera)
        m.setBoardSocket(sock)
        m.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        m.setFps(a.fps)
        m.initialControl.setManualExposure(a.exposure_us, a.iso)
        x = p.create(dai.node.XLinkOut); x.setStreamName(name); m.out.link(x.input)

    with dai.Device() as dev:
        cal = dev.readCalibration()
        cams = []
        for i, sock in enumerate((dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C)):
            K = np.array(cal.getCameraIntrinsics(sock, W, H))
            c = vslam.Camera()
            c.size = (W, H); c.focal = (K[0][0], K[1][1]); c.principal = (K[0][2], K[1][2])
            c.distortion = vslam.Distortion(vslam.Distortion.Model.Polynomial,
                                            list(cal.getDistortionCoefficients(sock))[:8])
            c.rig_from_camera = (vslam.Pose(rotation=[0, 0, 0, 1], translation=[0, 0, 0]) if i == 0
                                 else to_pose(cal.getCameraExtrinsics(dai.CameraBoardSocket.CAM_C,
                                                                      dai.CameraBoardSocket.CAM_B)))
            c.border_top = c.border_bottom = c.border_left = c.border_right = 24
            cams.append(c)
        rig = vslam.Rig(); rig.cameras = cams; rig.imus = []
        tracker = vslam.Tracker(rig, vslam.Tracker.OdometryConfig(
            odometry_mode=vslam.Tracker.OdometryMode.Multicamera,
            use_gpu=True, rectified_stereo_camera=False,
            async_sba=False, use_motion_model=True, use_denoising=False))

        dev.startPipeline(p)
        qL, qR = dev.getOutputQueue("left", 8, False), dev.getOutputQueue("right", 8, False)
        bufL, bufR = {}, {}
        T, P = [], []
        t0 = time.monotonic()
        while time.monotonic() - t0 < a.seconds:
            while qL.has():
                f = qL.get(); bufL[f.getSequenceNum()] = f
            while qR.has():
                f = qR.get(); bufR[f.getSequenceNum()] = f
            for seq in sorted(set(bufL) & set(bufR)):
                fl, fr = bufL.pop(seq), bufR.pop(seq)
                ts = int(fl.getTimestampDevice().total_seconds() * 1e9)
                est, _ = tracker.track(ts, [fl.getCvFrame(), fr.getCvFrame()])
                q = getattr(est, "world_from_rig", None) if est is not None else None
                q = getattr(q, "pose", q)
                if q is not None and np.all(np.isfinite(q.translation)):
                    T.append(time.monotonic() - t0); P.append(np.array(q.translation))
            time.sleep(0.002)

    T = np.array(T); P = np.stack(P)
    # Auto-detect the move instead of trusting wall-clock timing: find where the
    # smoothed speed rises, and use the quiet plateau before and after it.
    v = np.linalg.norm(np.diff(P, axis=0), axis=1)
    k = max(3, int(0.3 * len(v) / max(T[-1], 1)))          # ~0.3 s smoothing
    vs = np.convolve(v, np.ones(k) / k, mode="same")
    thr = max(vs.max() * 0.25, 1e-4)
    moving = vs > thr
    idx = np.flatnonzero(moving)
    if len(idx) < 3:
        print("no clear motion detected — move further"); return
    i0, i1 = idx[0], idx[-1] + 1
    pre = P[:max(i0, 5)]
    post = P[min(i1 + 1, len(P) - 5):]
    start, end = pre.mean(axis=0), post.mean(axis=0)
    d = end - start
    drift = np.linalg.norm(pre - start, axis=1).max()

    print(f"\n=== physical direction moved: {a.dir.upper()} ===")
    print(f"samples {len(P)}  |  move detected {T[i0]:.1f}s -> {T[min(i1,len(T)-1)]:.1f}s")
    print(f"pre-move noise floor: {drift*100:.1f} cm   (quiet samples: {len(pre)})")
    print("displacement in VIO axes (metres):")
    for i, ax in enumerate("xyz"):
        bar = "#" * int(min(abs(d[i]) * 60, 40))
        print(f"   vio_{ax} = {d[i]:+.3f}   {bar}")
    k2 = int(np.argmax(np.abs(d)))
    mag = np.linalg.norm(d)
    ratio = abs(d[k2]) / (mag + 1e-9)
    print(f"total magnitude: {mag:.3f} m")
    print(f"DOMINANT AXIS  : vio_{'xyz'[k2]}  sign {'+' if d[k2] > 0 else '-'}  "
          f"({ratio*100:.0f}% of the motion)")
    if mag < 5 * drift:
        print("WARNING: move was too small relative to noise — repeat with a bigger move")
    if ratio < 0.8:
        print("WARNING: motion was not clean along one axis — repeat more carefully")


if __name__ == "__main__":
    main()
