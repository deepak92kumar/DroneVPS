#!/usr/bin/env python3
"""How good is the affine with 50 corners, against the 300 that fly?

The camera's FeatureTracker costs 1.7 ms a frame where the host costs 62.4,
and it holds 95% of its features frame to frame. The one thing against it is
how MANY: 51 on this board against the 300 FlowTracker fits its affine to.

The project swept corner count once, over 900 flight frames, and found the
drift flat from 150 to 1200 - 6.8, 6.9, 6.8, 6.1 m - so more points buy
nothing. It never went below 150. 51 is a third of that, and nothing says
the curve stays flat down there.

WHAT THIS MEASURES. The same saved frames from flight 382, tracked at each
corner count, and the shift each produces turned into a velocity and scored
against GPS_RAW_INT over the same interval. Same frames, same chain, one
variable.

WHAT IT DOES NOT. The saved frames are 4 Hz, not the 25 the aircraft flew,
so every pair here is a 0.24 s gap where the flight had 0.04. That makes the
tracking harder than it really is for every setting equally - the comparison
between counts is fair, the absolute numbers are not the flight's.
"""
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vps_vio"))
from flow_odom import FlowTracker  # noqa: E402

FOLD = "/data/DRONE/flight_logs/vps/vps_00382"
COUNTS = (40, 60, 100, 150, 300, 600)


def main():
    cam = dict(l.split("=") for l in open(f"{FOLD}/camera.txt").read().split()[:4])
    fx = float(cam["fx"])
    d = pd.read_csv(f"{FOLD}/vps_00382.csv", low_memory=False)
    for c in ("t", "alt_m", "gps1_north_m", "gps1_east_m"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["src"] = d["src"].astype(str)
    V = d[(d.src == "VPS") & d.frame_file.notna()].reset_index(drop=True)
    V = V[np.isfinite(V.gps1_north_m)].reset_index(drop=True)
    print(f"{len(V)} saved frames on the VPS leg, {V.alt_m.median():.0f} m")
    print("scored against the 300-corner answer on the SAME frame pair, so the")
    print("4 Hz gap and the scene are identical for every count and only the")
    print("number of corners differs. An absolute truth cannot be used here:")
    print("the first attempt scored against the receiver and got 524% at 40")
    print("corners against 630% at 600, which is not the tracker getting worse")
    print("with more points - it is 40 corners failing on 310 of the 862 pairs")
    print("and being judged only on the easy ones it survived.\n")

    imgs = [f"{FOLD}/{f}" for f in V.frame_file]
    out = {}
    for n in COUNTS:
        tr = FlowTracker(max_corners=n, use_gpu=False)
        sh, ntr = [], []
        for path in imgs:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                sh.append(None); ntr.append(0); continue
            s_, k = tr.step(img)
            ntr.append(int(k))
            sh.append(None if s_ is None else np.asarray(s_, float).ravel()[:2])
        out[n] = (sh, int(np.median(ntr)))

    ref, _ = out[300]
    print("%8s %10s %9s %13s %10s %10s"
          % ("corners", "tracks", "answered", "vs 300: p50", "p90", "both"))
    for n in COUNTS:
        sh, tk = out[n]
        pairs = [(a, b) for a, b in zip(sh, ref)
                 if a is not None and b is not None]
        ans = sum(1 for x in sh if x is not None)
        if len(pairs) < 20:
            print("%8d %10d %9d   too few shared pairs" % (n, tk, ans)); continue
        dif = np.array([np.hypot(*(a - b)) for a, b in pairs])
        print("%8d %10d %9d %13.3f %10.3f %10d"
              % (n, tk, ans, np.median(dif), np.percentile(dif, 90), len(pairs)))
    print("\n  vs 300 is in PIXELS: how far this count's shift lands from the")
    print("  shift 300 corners gives on the same pair. 0 would be identical.")


if __name__ == "__main__":
    sys.exit(main())

# MEASURED 2 NOV on flight 382's 962 saved VPS frames at 88 m:
#
#     corners  tracks  answered  vs 300 p50  p90
#        40      31      600       1.804    6.464
#        60      37      766       1.016    6.210
#       100      51      863       0.633    5.741
#       150      64      900       0.474    4.687
#       300     100      935       0.000    0.000
#       600     162      946       0.491    3.886
#
# THE DEVICE'S 51 FEATURES ARE OUR 100-CORNER ROW - that setting gives
# tracks p50 51 too, so the comparison lands exactly. Against the 300 that
# fly it costs 7% of the steps (97% answered -> 90%) and moves the shift by
# 0.633 px p50, which at 88 m and fx 795 is 0.07 m a frame.
#
# 600 corners differs from 300 by 0.491 px, and that is the floor of this
# comparison rather than a real disagreement - two settings that should
# agree, measured the same way. So of 100's 0.633 px, only the part above
# ~0.49 is the cost of having fewer points.
#
# Below 60 it stops being a trade: 40 corners answers 600 of 962 pairs and
# lands 1.804 px out.
#
# THE FIRST VERSION OF THIS SCRIPT SCORED AGAINST THE RECEIVER and reported
# 524% relative error at 40 corners against 630% at 600 - which reads as
# "more corners is worse" and is nothing of the kind. 40 corners failed on
# 310 of the 862 pairs and was then judged only on the easy ones it
# survived. Scoring each count against the 300-corner answer ON THE SAME
# PAIR removes both that and the 4 Hz frame gap.
