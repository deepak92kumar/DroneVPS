#!/usr/bin/env python3
"""What the OAK-D's own FeatureTracker costs, against the flow we run on the host.

THE QUESTION. The flow loop is the thing that limits this aircraft. Measured
1 Nov on the ground: the camera delivers 24.7 fps, the loop runs 9.06 Hz and
15.65 frames a second are dropped - and the matcher is not why, its child
process sits at 0.5% CPU. Two FlowTrackers at 53.39 ms each is 106.8 ms a
frame, a ceiling of 9.37 Hz, and the loop measured 9.06.

The OAK-D has a FeatureTracker node: Harris or Shi-Tomasi corners and
Lucas-Kanade optical flow, run on the Myriad, with a feature maintainer that
keeps ids alive between frames - which is what FlowTracker does in Python.
StereoDepth already runs there and costs the host nothing.

WHAT THIS MEASURES, and what it does not. It measures host CPU per frame and
the frame rate each path sustains, with the same camera, same resolution, and
the same target feature count. It does NOT measure navigation error: the
device tracker returns ids and positions, not the affine this project fits,
so whether the shift it implies is as good as OpenCV's is a separate
question that needs flight frames and the receiver.

THE CAMERA HAS ONE OWNER. live_node holds it while it runs, so this cannot
be run beside the flying node - stop the service, run this, start it again,
and wait for 03e7:2485 to come back between the two.

    sudo systemctl stop drone-vps
    python3 bench/featuretracker_vs_host.py
    sudo systemctl start drone-vps
"""
import sys
import time

import cv2
import numpy as np

SECONDS = 20.0
TARGET = 300
RES = "THE_800_P"


def host_path(seconds=SECONDS):
    """goodFeaturesToTrack + two-way LK on the host, as FlowTracker does it."""
    import depthai as dai
    p = dai.Pipeline()
    m = p.create(dai.node.MonoCamera)
    m.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    m.setResolution(getattr(dai.MonoCameraProperties.SensorResolution, RES))
    m.setFps(30)
    xo = p.create(dai.node.XLinkOut)
    xo.setStreamName("img")
    m.out.link(xo.input)

    frames = 0
    prev = None
    pts = None
    c0 = None
    t0 = None
    with dai.Device(p) as dev:
        q = dev.getOutputQueue("img", 4, False)
        while True:
            f = q.get().getCvFrame()
            if t0 is None:                      # first frame is warm-up
                t0 = time.perf_counter(); c0 = time.process_time()
                prev = f
                pts = cv2.goodFeaturesToTrack(f, TARGET, 0.01, 8)
                continue
            if pts is None or len(pts) < 30:
                pts = cv2.goodFeaturesToTrack(f, TARGET, 0.01, 8)
                prev = f
                continue
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, f, pts, None,
                                                 winSize=(31, 31), maxLevel=4)
            p0r, st2, _ = cv2.calcOpticalFlowPyrLK(f, prev, p1, None,
                                                   winSize=(31, 31), maxLevel=4)
            good = ((st.ravel() == 1) & (st2.ravel() == 1)
                    & (np.linalg.norm(pts - p0r, axis=2).ravel() < 1.0))
            pts = p1[good].reshape(-1, 1, 2)
            prev = f
            frames += 1
            if time.perf_counter() - t0 > seconds:
                break
        wall = time.perf_counter() - t0
        cpu = time.process_time() - c0
    return frames, wall, cpu, (0 if pts is None else len(pts))


def device_path(seconds=SECONDS):
    """Corners and optical flow on the Myriad; the host only reads the result."""
    import depthai as dai
    p = dai.Pipeline()
    m = p.create(dai.node.MonoCamera)
    m.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    m.setResolution(getattr(dai.MonoCameraProperties.SensorResolution, RES))
    m.setFps(30)
    ft = p.create(dai.node.FeatureTracker)
    cfg = ft.initialConfig
    cfg.setCornerDetector(dai.FeatureTrackerConfig.CornerDetector.Type.HARRIS)
    cfg.setNumTargetFeatures(TARGET)
    cfg.setMotionEstimator(True)
    cfg.setOpticalFlow()
    m.out.link(ft.inputImage)
    xo = p.create(dai.node.XLinkOut)
    xo.setStreamName("feat")
    ft.outputFeatures.link(xo.input)

    frames = 0
    n_last = 0
    ages = []
    t0 = c0 = None
    with dai.Device(p) as dev:
        q = dev.getOutputQueue("feat", 4, False)
        while True:
            fts = q.get().trackedFeatures
            if t0 is None:
                t0 = time.perf_counter(); c0 = time.process_time()
                continue
            n_last = len(fts)
            if fts:
                ages.append(float(np.median([f.age for f in fts])))
            frames += 1
            if time.perf_counter() - t0 > seconds:
                break
        wall = time.perf_counter() - t0
        cpu = time.process_time() - c0
    return frames, wall, cpu, n_last, (np.median(ages) if ages else 0.0)


def main():
    print(f"same camera, {RES}, {TARGET} target features, {SECONDS:.0f} s each\n")
    print("%-28s %8s %9s %11s %9s" % ("path", "fps", "CPU/frame", "CPU cores", "features"))
    fr, wall, cpu, n = host_path()
    h_fps, h_cpu = fr / wall, cpu / fr * 1000
    print("%-28s %8.2f %7.1f ms %10.2f %9d"
          % ("host: LK + goodFeatures", h_fps, h_cpu, cpu / wall, n))
    time.sleep(2.0)
    fr, wall, cpu, n, age = device_path()
    d_fps, d_cpu = fr / wall, cpu / fr * 1000
    print("%-28s %8.2f %7.1f ms %10.2f %9d   (median age %.0f frames)"
          % ("device: FeatureTracker", d_fps, d_cpu, cpu / wall, n, age))
    print("\n  CPU per frame: %.1f -> %.1f ms  =  %.0f%% less"
          % (h_cpu, d_cpu, 100 * (1 - d_cpu / h_cpu)))
    print("  fps: %.2f -> %.2f" % (h_fps, d_fps))
    print("\n  two trackers run on the aircraft, so the host figure doubles:")
    print("    host   %.1f ms/frame -> ceiling %.2f Hz" % (2 * h_cpu, 1000 / (2 * h_cpu)))
    print("    device %.1f ms/frame -> ceiling %.2f Hz" % (2 * d_cpu, 1000 / (2 * d_cpu)))


if __name__ == "__main__":
    sys.exit(main())
