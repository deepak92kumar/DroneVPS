#!/usr/bin/env python3
"""If the estimate is 200 or 500 m out, can the matcher find the frame again?

    python3 wide_search.py vps_00476 [--offsets 0,50,100,200,500]

The pilot's question: at 100 m on the VPS, a stretch of ground changes enough
that the matcher stops matching, the flow carries on alone, and it
under-measures - 0.40 to 1.08 of true distance across this project's flights,
0.694 on the flight that failed - so 500 m flown can read 350 m. The estimate
is then hundreds of metres out. Can the camera frame be found in the REST of
the sheet?

It cannot today. The matcher is handed a window centred on the estimate,
radius clip(0.60 x alt, 40, 150) m, so a prior 200 m out means the frame's
own footprint is not inside the window at all. Measured on 675 frames, XFeat
accepts 78% of frames at a prior under 10 m, 26% at 20-50 m and 1% at
50-200 m. That 1% was still 11 m accurate, which is why this is worth
measuring rather than assuming.

So: the prior is deliberately moved D metres away in eight bearings, the
window is widened to cover the truth anyway, and three things are recorded.

    RECOVERED   accepted, and within 20 m of the M9N          - the point
    FALSE FIX   accepted, and further than 20 m               - the danger
    declined    the gate held                                 - the safe case

A relocalisation mode is only worth building if RECOVERED is common and
FALSE FIX is rare. A false fix while lost is worse than staying lost, because
staying lost is ALT_HOLD and a false fix is a confident wrong position.
"""
import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay as R                                      # noqa: E402

sys.path.insert(0, "/data/DRONE/vps_vio")
from map_cache import MapCache                          # noqa: E402
from map_match_xfeat import locate as xf_locate         # noqa: E402

MDEG = 111320.0
BEARINGS = [0, 45, 90, 135, 180, 225, 270, 315]


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--offsets", default="0,50,100,200,500")
    ap.add_argument("--frames", type=int, default=8,
                    help="frames per flight (each is tried in 8 bearings)")
    ap.add_argument("--min-alt", type=float, default=58.0)
    ap.add_argument("--map", default="/data/DRONE/maps/active.tif")
    a = ap.parse_args()
    offsets = [float(v) for v in a.offsets.split(",")]

    cache = MapCache(os.path.realpath(a.map))
    sel = []
    for run in a.runs:
        fr = R.frames(run, alt_min=a.min_alt, step=3)
        for f in fr:
            f["prior_err"] = R.metres(f["est_lon"], f["est_lat"],
                                      f["true_lon"], f["true_lat"])
            f["run"] = run
        fr = [f for f in fr if f["prior_err"] < 12.0]
        if not fr:
            continue
        idx = np.linspace(0, len(fr) - 1, min(a.frames, len(fr)))
        sel += [fr[int(round(i))] for i in idx]
    print(f"{len(sel)} frames x {len(BEARINGS)} bearings x {len(offsets)} "
          f"offsets = {len(sel) * len(BEARINGS) * len(offsets)} matches\n",
          flush=True)

    rows, t0 = [], time.time()
    for n, f in enumerate(sel):
        img = cv2.imread(f["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        mlon = MDEG * math.cos(math.radians(f["true_lat"]))
        for D in offsets:
            # The window has to be able to contain the truth, or the question
            # is not being asked. Half the north-up query's own side is
            # 0.95 x altitude; 30 m of slack on top.
            rad = D + 0.95 * f["alt"] + 30.0
            for brg in (BEARINGS if D > 0 else [0]):
                th = math.radians(brg)
                plat = f["true_lat"] + (D * math.cos(th)) / MDEG
                plon = f["true_lon"] + (D * math.sin(th)) / mlon
                try:
                    if not cache.covers(plon, plat):
                        continue
                    win, wg = cache.window(plon, plat, rad, kind="grey")
                    t1 = time.time()
                    res = xf_locate(img, win, wg, altitude_m=f["alt"],
                                    focal_px=R.CAM["fx"],
                                    yaw_deg=f["yaw"] + R.YAW_OFFSET,
                                    map_gsd_m=cache.gsd,
                                    map_gsd_x_m=getattr(cache, "gsd_x", None),
                                    min_score=0.07, min_margin=0.20,
                                    principal_xy=(R.CAM["cx"], R.CAM["cy"]),
                                    query_shape="full")
                    ms = 1000 * (time.time() - t1)
                except Exception as exc:
                    rows.append(dict(run=f["run"], D=D, brg=brg, alt=f["alt"],
                                     rad=rad, win_px=0, ok=0,
                                     err=float("nan"), ms=0.0,
                                     why=f"raised {exc}"))
                    continue
                err = (R.metres(res.lon, res.lat, f["true_lon"], f["true_lat"])
                       if res.lon else float("nan"))
                rows.append(dict(run=f["run"], D=D, brg=brg, alt=f["alt"],
                                 rad=rad, win_px=win.shape[1], ok=int(res.ok),
                                 err=err, ms=ms, score=res.score,
                                 why=res.reason or ""))
        print(f"  frame {n+1}/{len(sel)}  alt {f['alt']:.0f} m  "
              f"({time.time()-t0:.0f} s)", flush=True)

    print(f"\n{'=' * 78}")
    print("  prior   window    n  |  RECOVERED   FALSE FIX   declined |  "
          "err of recovered   cost")
    print(f"{'=' * 78}")
    for D in offsets:
        s = [r for r in rows if r["D"] == D and r["win_px"]]
        if not s:
            continue
        acc = [r for r in s if r["ok"] and np.isfinite(r["err"])]
        rec = [r for r in acc if r["err"] <= 20.0]
        bad = [r for r in acc if r["err"] > 20.0]
        e = [r["err"] for r in rec]
        print("  %5.0f m  %5.0f px %4d  |  %4d %4.0f%%  %4d %4.0f%%  %4d %4.0f%% |"
              "  p50 %5.2f max %6.2f  %5.0f ms" % (
                  D, np.median([r["win_px"] for r in s]), len(s),
                  len(rec), 100 * len(rec) / len(s),
                  len(bad), 100 * len(bad) / len(s),
                  len(s) - len(acc), 100 * (len(s) - len(acc)) / len(s),
                  np.median(e) if e else float("nan"),
                  max(e) if e else float("nan"),
                  np.median([r["ms"] for r in s])))
    print(f"\n  RECOVERED = accepted and within 20 m of the M9N")
    print("  FALSE FIX = accepted and further than 20 m - the number that "
          "decides whether a relocalisation mode is safe to build")
    import csv as _csv
    p = "/data/DRONE/bench/xfeat_audit/wide_search.csv"
    with open(p, "w", newline="") as fh:
        ks = sorted({k for r in rows for k in r})
        w = _csv.DictWriter(fh, fieldnames=ks); w.writeheader(); w.writerows(rows)
    print(f"\n  -> {p}   ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
