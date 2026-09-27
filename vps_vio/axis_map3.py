#!/usr/bin/env python3
"""Measure the full VIO->body axis mapping in ONE tracker session.

Doing each direction as a separate run re-initialises cuVSLAM every time, so the
three answers live in three different world frames and cannot be combined. Here
all three moves happen inside one session: the world frame is the rig frame at
t=0, which (with the drone level and not rotated) IS the body frame. So the
three measured displacement vectors directly give the VIO->body rotation.

Schedule (the script auto-detects the three motion bursts, so timing is loose):
    still ~12 s  ->  FORWARD ~40 cm  ->  still  ->  RIGHT ~40 cm  ->  still
                 ->  UP ~40 cm       ->  still

Do not rotate the drone at any point — slide/lift it only.
"""
from __future__ import annotations
import argparse, time
import numpy as np
import depthai as dai
import cuvslam as vslam
from scipy.spatial.transform import Rotation

W, H = 640, 400


def to_pose(mat):
    m = np.asarray(mat, float)
    return vslam.Pose(rotation=Rotation.from_matrix(m[:3, :3]).as_quat(),
                      translation=m[:3, 3] / 100.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=80)
    ap.add_argument("--exposure-us", type=int, default=4000)
    ap.add_argument("--iso", type=int, default=800)
    ap.add_argument("--csv", default="/mnt/ssd/DRONE/vps_vio/logs/axis3.csv")
    a = ap.parse_args()

    vslam.warm_up_gpu()
    p = dai.Pipeline()
    for name, sock in (("left", dai.CameraBoardSocket.CAM_B),
                       ("right", dai.CameraBoardSocket.CAM_C)):
        m = p.create(dai.node.MonoCamera)
        m.setBoardSocket(sock)
        m.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        m.setFps(30)
        m.initialControl.setManualExposure(a.exposure_us, a.iso)
        x = p.create(dai.node.XLinkOut); x.setStreamName(name); m.out.link(x.input)

    with dai.Device() as dev:
        cal = dev.readCalibration(); cams = []
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
            odometry_mode=vslam.Tracker.OdometryMode.Multicamera, use_gpu=True,
            rectified_stereo_camera=False, async_sba=False,
            use_motion_model=True, use_denoising=False))

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
                est, _ = tracker.track(int(fl.getTimestampDevice().total_seconds() * 1e9),
                                       [fl.getCvFrame(), fr.getCvFrame()])
                q = getattr(est, "world_from_rig", None) if est is not None else None
                q = getattr(q, "pose", q)
                if q is not None and np.all(np.isfinite(q.translation)):
                    T.append(time.monotonic() - t0); P.append(np.array(q.translation))
            time.sleep(0.002)

    T = np.array(T); P = np.stack(P)
    np.savetxt(a.csv, np.column_stack([T, P]), delimiter=",", header="t,x,y,z", comments="")

    v = np.linalg.norm(np.diff(P, axis=0), axis=1)
    k = max(5, int(0.4 * len(v) / max(T[-1], 1)))
    vs = np.convolve(v, np.ones(k) / k, mode="same")
    thr = max(vs.max() * 0.15, 2e-4)
    moving = vs > thr
    # group contiguous moving stretches
    segs, i = [], 0
    while i < len(moving):
        if moving[i]:
            j = i
            while j < len(moving) and moving[j]:
                j += 1
            if T[min(j, len(T) - 1)] - T[i] > 0.5:
                segs.append((i, j))
            i = j
        else:
            i += 1
    # merge segments separated by short gaps
    merged = []
    for s in segs:
        if merged and T[s[0]] - T[merged[-1][1]] < 1.5:
            merged[-1] = (merged[-1][0], s[1])
        else:
            merged.append(list(s))
    merged = [tuple(m) for m in merged]

    print(f"\nsamples {len(P)}  duration {T[-1]:.1f}s")
    print(f"detected {len(merged)} motion segment(s)\n")
    names = ["FORWARD", "RIGHT", "UP"]
    results = []
    for n, (i0, i1) in enumerate(merged):
        pre = P[max(0, i0 - 60):i0] if i0 >= 10 else P[:10]
        post = P[i1:min(len(P), i1 + 60)] if i1 + 10 <= len(P) else P[-10:]
        d = post.mean(axis=0) - pre.mean(axis=0)
        mag = np.linalg.norm(d)
        kk = int(np.argmax(np.abs(d)))
        lbl = names[n] if n < len(names) else f"move{n+1}"
        print(f"--- segment {n+1} ({T[i0]:.1f}-{T[min(i1,len(T)-1)]:.1f}s)  = {lbl}")
        print(f"    vio = [{d[0]:+.3f}, {d[1]:+.3f}, {d[2]:+.3f}]   |d| = {mag:.3f} m")
        print(f"    dominant: vio_{'xyz'[kk]} {'+' if d[kk]>0 else '-'}  "
              f"({abs(d[kk])/(mag+1e-9)*100:.0f}%)")
        results.append((lbl, d))

    if len(results) == 3:
        print("\n=== VIO -> BODY mapping ===")
        for lbl, d in results:
            u = d / (np.linalg.norm(d) + 1e-9)
            print(f"  body {lbl:8s} = [{u[0]:+.2f}, {u[1]:+.2f}, {u[2]:+.2f}] in VIO axes")
    else:
        print(f"\nExpected 3 segments, found {len(merged)} — redo with clearer pauses.")


if __name__ == "__main__":
    main()
