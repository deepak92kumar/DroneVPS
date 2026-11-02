#!/usr/bin/env python3
"""Do the device's features PERSIST, and how many can it actually hold?

The cost benchmark said 1.7 ms a frame against the host's 62.4 - 97% less -
but it also reported a median age of 1 frame, and a tracker whose features
are new every frame is not a tracker. FlowTracker's whole saving comes from
keeping corners between frames: "707 of 714 corners were still alive after
59 frames", and re-detecting every frame was 30 ms of a 60 ms step.

So this asks the only question that matters next: given an id, does it come
back on the following frame, and for how many frames.

It also answers the warning the first run printed - "Target number of
features 300 exceeds the maximum 146. Clamping." - by sweeping
setHardwareResources, which is how the node is given more memory slices.

The camera has one owner. Stop drone-vps, run this, start it again.
"""
import sys
import time
from collections import Counter

import numpy as np

SECONDS = 14.0


def run(target, shaves, slices):
    import depthai as dai
    p = dai.Pipeline()
    m = p.create(dai.node.MonoCamera)
    m.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    m.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    m.setFps(30)
    ft = p.create(dai.node.FeatureTracker)
    if shaves is not None:
        ft.setHardwareResources(shaves, slices)
    cfg = ft.initialConfig
    cfg.setCornerDetector(dai.FeatureTrackerConfig.CornerDetector.Type.HARRIS)
    cfg.setNumTargetFeatures(target)
    cfg.setMotionEstimator(True)
    cfg.setOpticalFlow()
    m.out.link(ft.inputImage)
    xo = p.create(dai.node.XLinkOut)
    xo.setStreamName("feat")
    ft.outputFeatures.link(xo.input)

    counts, ages, carried, lifetimes = [], [], [], Counter()
    prev_ids = None
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
            if fts:
                ages.append(np.median([f.age for f in fts]))
                for f in fts:
                    lifetimes[f.id] = max(lifetimes[f.id], f.age)
            if prev_ids is not None and prev_ids:
                carried.append(len(ids & prev_ids) / len(prev_ids))
            prev_ids = ids
            if time.perf_counter() - t0 > SECONDS:
                break
    return (np.median(counts) if counts else 0,
            np.median(ages) if ages else 0,
            np.median(carried) if carried else 0,
            np.percentile(list(lifetimes.values()), 90) if lifetimes else 0,
            max(lifetimes.values()) if lifetimes else 0)


def main():
    print("%-34s %9s %8s %12s %10s %8s"
          % ("setup", "features", "age p50", "carried over", "life p90", "max"))
    for target, shaves, slices in ((300, None, None), (300, 2, 2),
                                   (300, 4, 4), (600, 4, 4)):
        tag = f"target {target}, " + ("default res" if shaves is None
                                      else f"{shaves} shaves {slices} slices")
        try:
            n, age, carry, p90, mx = run(target, shaves, slices)
            print("%-34s %9.0f %8.0f %11.0f%% %10.0f %8.0f"
                  % (tag, n, age, 100 * carry, p90, mx))
        except Exception as exc:
            print("%-34s  %s" % (tag, str(exc)[:46]))
        time.sleep(2.0)
    print("\n  carried over = fraction of one frame's ids still present in the next.")
    print("  FlowTracker's own bar: 707 of 714 corners alive after 59 frames.")


if __name__ == "__main__":
    sys.exit(main())
