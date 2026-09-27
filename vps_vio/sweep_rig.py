#!/usr/bin/env python3
"""Find the correct cuVSLAM rig config for THIS OAK-D by measuring stationary drift.

A stationary camera must report ~zero motion. We capture one set of stationary
stereo frames, then replay the SAME frames through several candidate rig
configurations and report the drift of each. Lowest drift wins.

NVIDIA's OAK example is written for the OAK-D **W** Pro and its README warns that
the distortion model and camera order differ on other OAK-D models — this sweep
settles it by measurement instead of guesswork.

Keep the camera completely still while it captures.
"""
from __future__ import annotations
import argparse, time
import numpy as np
import depthai as dai
import cuvslam as vslam
from scipy.spatial.transform import Rotation

W, H = 640, 400
CM_TO_M = 100.0


def capture(n_frames: int, fps: int, exposure_us: int, iso: int):
    p = dai.Pipeline()
    for name, sock in (("left", dai.CameraBoardSocket.CAM_B),
                       ("right", dai.CameraBoardSocket.CAM_C)):
        m = p.create(dai.node.MonoCamera)
        m.setBoardSocket(sock)
        m.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        m.setFps(fps)
        m.initialControl.setManualExposure(exposure_us, iso)
        x = p.create(dai.node.XLinkOut); x.setStreamName(name); m.out.link(x.input)

    with dai.Device() as dev:
        calib = dev.readCalibration()
        dev.startPipeline(p)
        qL, qR = dev.getOutputQueue("left", 8, False), dev.getOutputQueue("right", 8, False)
        bufL, bufR, frames = {}, {}, []
        t0 = time.monotonic()
        while len(frames) < n_frames and time.monotonic() - t0 < 60:
            while qL.has():
                f = qL.get(); bufL[f.getSequenceNum()] = f
            while qR.has():
                f = qR.get(); bufR[f.getSequenceNum()] = f
            for seq in sorted(set(bufL) & set(bufR)):
                fl, fr = bufL.pop(seq), bufR.pop(seq)
                ts = int(fl.getTimestampDevice().total_seconds() * 1e9)
                frames.append((ts, fl.getCvFrame(), fr.getCvFrame()))
            time.sleep(0.002)
        # pull calibration data out while the device is open
        cal = {
            "K": {s.name: np.array(calib.getCameraIntrinsics(s, W, H))
                  for s in (dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C)},
            "D": {s.name: list(calib.getDistortionCoefficients(s))
                  for s in (dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C)},
            "T_CB": np.array(calib.getCameraExtrinsics(dai.CameraBoardSocket.CAM_C,
                                                       dai.CameraBoardSocket.CAM_B)),
            "T_BC": np.array(calib.getCameraExtrinsics(dai.CameraBoardSocket.CAM_B,
                                                       dai.CameraBoardSocket.CAM_C)),
        }
    return frames, cal


def to_pose(mat) -> vslam.Pose:
    m = np.asarray(mat, float)
    return vslam.Pose(rotation=Rotation.from_matrix(m[:3, :3]).as_quat(),
                      translation=m[:3, 3] / CM_TO_M)


def make_rig(cal, *, order, dist_model, n_coeff, extr, border):
    """order: ('CAM_B','CAM_C') or swapped. extr: 4x4 placing the 2nd cam in rig frame."""
    cams = []
    for i, sname in enumerate(order):
        K = cal["K"][sname]
        c = vslam.Camera()
        c.size = (W, H)
        c.focal = (K[0][0], K[1][1])
        c.principal = (K[0][2], K[1][2])
        if dist_model == "pinhole":
            c.distortion = vslam.Distortion(vslam.Distortion.Model.Pinhole, [])
        elif dist_model == "brown":
            c.distortion = vslam.Distortion(vslam.Distortion.Model.Brown, cal["D"][sname][:n_coeff])
        else:
            c.distortion = vslam.Distortion(vslam.Distortion.Model.Polynomial, cal["D"][sname][:n_coeff])
        c.rig_from_camera = (vslam.Pose(rotation=[0, 0, 0, 1], translation=[0, 0, 0])
                             if i == 0 else to_pose(extr))
        c.border_top = c.border_bottom = c.border_left = c.border_right = border
        cams.append(c)
    r = vslam.Rig(); r.cameras = cams; r.imus = []
    return r


def drift_of(rig, frames, swap_images: bool):
    cfg = vslam.Tracker.OdometryConfig(
        odometry_mode=vslam.Tracker.OdometryMode.Multicamera,
        use_gpu=True, rectified_stereo_camera=False,
        async_sba=False, use_motion_model=True, use_denoising=False)
    tr = vslam.Tracker(rig, cfg)
    pts, bad = [], 0
    for ts, l, r in frames:
        imgs = [r, l] if swap_images else [l, r]
        est, _ = tr.track(ts, imgs)
        p = getattr(est, "world_from_rig", None) if est is not None else None
        p = getattr(p, "pose", p)
        if p is None or not np.all(np.isfinite(p.translation)):
            bad += 1; continue
        pts.append(np.array(p.translation))
    if len(pts) < 5:
        return None
    P = np.stack(pts)
    steps = np.linalg.norm(np.diff(P, axis=0), axis=1)
    return dict(final=float(np.linalg.norm(P[-1] - P[0])),
                maxdev=float(np.linalg.norm(P - P[0], axis=1).max()),
                path=float(steps.sum()), maxstep=float(steps.max()), bad=bad, n=len(P))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--exposure-us", type=int, default=4000)
    ap.add_argument("--iso", type=int, default=400)
    ap.add_argument("--border", type=int, default=24)
    a = ap.parse_args()

    vslam.warm_up_gpu()
    print(f"capturing {a.frames} STATIONARY stereo frames — do not touch the camera...")
    frames, cal = capture(a.frames, a.fps, a.exposure_us, a.iso)
    print(f"captured {len(frames)} pairs\n")

    T_CB, T_BC = cal["T_CB"], cal["T_BC"]
    order_n = ("CAM_B", "CAM_C")
    order_s = ("CAM_C", "CAM_B")

    trials = [
        ("poly8  | B,C | T(C->B)",      dict(order=order_n, dist_model="poly", n_coeff=8, extr=T_CB), False),
        ("poly8  | B,C | T(B->C)",      dict(order=order_n, dist_model="poly", n_coeff=8, extr=T_BC), False),
        ("pinhole| B,C | T(C->B)",      dict(order=order_n, dist_model="pinhole", n_coeff=0, extr=T_CB), False),
        ("pinhole| B,C | T(B->C)",      dict(order=order_n, dist_model="pinhole", n_coeff=0, extr=T_BC), False),
        ("brown5 | B,C | T(C->B)",      dict(order=order_n, dist_model="brown", n_coeff=5, extr=T_CB), False),
        ("brown5 | B,C | T(B->C)",      dict(order=order_n, dist_model="brown", n_coeff=5, extr=T_BC), False),
        ("poly8  | C,B | T(B->C) swap", dict(order=order_s, dist_model="poly", n_coeff=8, extr=T_BC), True),
        ("pinhole| C,B | T(B->C) swap", dict(order=order_s, dist_model="pinhole", n_coeff=0, extr=T_BC), True),
    ]

    print(f"{'config':32s} {'final':>9s} {'maxdev':>9s} {'path':>9s} {'maxstep':>9s} {'bad':>4s}")
    print("-" * 78)
    results = []
    for label, kw, swap in trials:
        try:
            rig = make_rig(cal, border=a.border, **kw)
            r = drift_of(rig, frames, swap)
        except Exception as e:
            print(f"{label:32s}  ERROR: {str(e)[:34]}"); continue
        if r is None:
            print(f"{label:32s}  no valid poses"); continue
        results.append((r["final"], label, r))
        print(f"{label:32s} {r['final']:9.3f} {r['maxdev']:9.3f} {r['path']:9.3f} "
              f"{r['maxstep']:9.3f} {r['bad']:4d}")

    if results:
        results.sort()
        best = results[0]
        print("\nBEST (lowest stationary drift):", best[1])
        print("  A correct config should show final/maxdev of a few CENTIMETRES, not metres.")


if __name__ == "__main__":
    main()
