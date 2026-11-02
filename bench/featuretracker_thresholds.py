#!/usr/bin/env python3
"""Can the device tracker be made to HOLD its features?

The cost is settled - 1.7 ms a frame against the host's 62.4, 97% less. The
blocker is persistence: with the defaults only 30% of one frame's ids appear
in the next and 90% of features are dead within 2 frames, where FlowTracker
keeps 707 of 714 alive across 59.

The maintainer is not the reason: FeatureMaintainer.enable is already True.
What it carries are two thresholds that decide when a feature is given up -
lostFeatureErrorThreshold 50000 and trackedFeatureThreshold 200000 - and a
minimumDistanceBetweenFeatures of 50 px, which on 1280x800 is itself a limit
on how many can coexist. This sweeps them.

The camera has one owner. Stop drone-vps, run this, start it again.
"""
import sys
import time
from collections import Counter

import numpy as np

SECONDS = 12.0


def run(lost, tracked, mindist, target=300, shaves=2, slices=2):
    import depthai as dai
    p = dai.Pipeline()
    m = p.create(dai.node.MonoCamera)
    m.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    m.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    m.setFps(30)
    ft = p.create(dai.node.FeatureTracker)
    ft.setHardwareResources(shaves, slices)
    cfg = ft.initialConfig
    cfg.setCornerDetector(dai.FeatureTrackerConfig.CornerDetector.Type.HARRIS)
    cfg.setNumTargetFeatures(target)
    cfg.setMotionEstimator(True)
    cfg.setOpticalFlow()
    raw = cfg.get()
    raw.featureMaintainer.enable = True
    raw.featureMaintainer.lostFeatureErrorThreshold = float(lost)
    raw.featureMaintainer.trackedFeatureThreshold = float(tracked)
    raw.featureMaintainer.minimumDistanceBetweenFeatures = float(mindist)
    cfg.set(raw)
    m.out.link(ft.inputImage)
    xo = p.create(dai.node.XLinkOut)
    xo.setStreamName("feat")
    ft.outputFeatures.link(xo.input)

    counts, carried, lifetimes = [], [], Counter()
    prev = None
    t0 = None
    with dai.Device(p) as dev:
        q = dev.getOutputQueue("feat", 8, False)
        while True:
            fts = q.get().trackedFeatures
            if t0 is None:
                t0 = time.perf_counter()
                continue
            ids = {f.id for f in fts}
            counts.append(len(fts))
            for f in fts:
                lifetimes[f.id] = max(lifetimes[f.id], f.age)
            if prev:
                carried.append(len(ids & prev) / len(prev))
            prev = ids
            if time.perf_counter() - t0 > SECONDS:
                break
    lv = list(lifetimes.values())
    return (np.median(counts) if counts else 0,
            np.median(carried) if carried else 0,
            np.percentile(lv, 50) if lv else 0,
            np.percentile(lv, 90) if lv else 0,
            max(lv) if lv else 0)


def main():
    print("%-38s %8s %11s %8s %8s %6s"
          % ("lost / tracked / min-dist", "feats", "carried", "life50", "life90", "max"))
    cases = [(50000, 200000, 50), (50000, 200000, 15), (500000, 200000, 15),
             (5e6, 2e6, 15), (5e6, 2e6, 8), (5e7, 2e7, 8)]
    for lost, tracked, md in cases:
        tag = "%-9g %-9g %3d" % (lost, tracked, md)
        try:
            n, c, l50, l90, mx = run(lost, tracked, md)
            print("%-38s %8.0f %10.0f%% %8.0f %8.0f %6.0f"
                  % (tag, n, 100 * c, l50, l90, mx))
        except Exception as exc:
            print("%-38s  %s" % (tag, str(exc)[:42]))
        time.sleep(2.0)
    print("\n  FlowTracker's bar: 707 of 714 alive after 59 frames = 99% carried.")


if __name__ == "__main__":
    sys.exit(main())

# MEASURED 2 NOV, on the ground, camera static, daylight:
#
#     lost / tracked / min-dist   feats  carried  life50  life90   max
#     50000   200000   50            40     95%       4      30    276
#     50000   200000   15            51     95%       4      32    302
#     500000  200000   15            47     94%       4      30    308
#     5e6     2e6      15            27     76%       2       8    270
#     5e6     2e6       8            31     77%       2       8    287
#     5e7     2e7       8            24      5%       1       1     30
#
# THE HYPOTHESIS WAS BACKWARDS. Raising the thresholds does not keep features
# alive, it kills them - monotonically, until at 5e7 only 5% survive a single
# frame. The defaults are the best setting on the board and the only knob
# worth moving is minimumDistanceBetweenFeatures, 50 -> 15, which buys 40 ->
# 51 features for nothing.
#
# AND THE 30% WAS THE SCENE. The run the night before gave 30% carried and a
# life90 of 2 on these same defaults, which read like a tracker that cannot
# hold anything. In daylight the identical configuration gives 95% and a life
# of 300 frames - ten seconds. A dark low-texture scene is not a verdict on
# the tracker, and one run at night nearly became one.
#
# WHAT IS STILL OPEN: 51 features against the 300 FlowTracker fits its affine
# to. The project swept 150 to 1200 corners over 900 flight frames and found
# the drift flat - 6.8, 6.9, 6.8, 6.1 m - but it never went below 150, and
# this is a third of that. Whether 50 points fit as good an affine needs
# flight frames, not a static camera on the ground.
