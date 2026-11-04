#!/usr/bin/env python3
"""The real worker, the real config, real frames. Does it fly?

Not a unit test: MapFixWorker as live_node builds it, matcher
"xfeat,xfeat_lg", on saved frames from both 4 Nov flights. Checks the three
things that decide whether tomorrow's flight is worth making:

  1 fixes still come out, and more of them than mutual NN alone
  2 LighterGlue runs ONLY on the frames mutual NN refuses, so the cost is
    paid only when needed
  3 fix_source names which one answered, every time
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay as R
import fclog
sys.path.insert(0, "/data/DRONE/vps_vio")
from map_cache import MapCache
from map_fix_worker import MapFixWorker

MDEG = 111320.0


def main():
    import cv2
    cache = MapCache(os.path.realpath("/data/DRONE/maps/active.tif"))
    out = {}
    for matcher in ("xfeat", "xfeat,xfeat_lg"):
        w = MapFixWorker(cache, focal_px=R.CAM["fx"], matcher=matcher,
                         radius_m=150.0, window_alt_frac=0.60,
                         query_shape="full", min_alt_m=60.0, fix_period_s=0.0,
                         principal_xy=(R.CAM["cx"], R.CAM["cy"]))
        rows = []
        for run in ("vps_00476", "vps_00478"):
            fclog.attitude(run, verbose=False)
            fr = R.frames(run, alt_min=58, step=14)
            for f in fr:
                img = cv2.imread(f["path"], cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                roll, pitch, _ = fclog.at(run, f["t"])
                t0 = time.monotonic()
                if not w.submit(img, prior_lon=f["est_lon"],
                                prior_lat=f["est_lat"], altitude_m=f["alt"],
                                yaw_deg=f["yaw"] + R.YAW_OFFSET,
                                roll_deg=roll, pitch_deg=pitch,
                                att_age_s=0.05):
                    continue
                res = None
                for _ in range(2000):
                    res = w.poll()
                    if res is not None:
                        break
                    time.sleep(0.01)
                ms = 1000 * (time.monotonic() - t0)
                if res is None:
                    continue
                err = float("nan")
                if res.ok:
                    err = R.metres(res.lon, res.lat, f["true_lon"],
                                   f["true_lat"])
                rows.append(dict(
                    run=run, t=f["t"], alt=f["alt"], ok=int(res.ok),
                    src=res.source or "", err=err, ms=ms,
                    prior=R.metres(f["est_lon"], f["est_lat"],
                                   f["true_lon"], f["true_lat"]),
                    reason=res.reason or ""))
        w.stop()
        out[matcher] = rows
        acc = [r for r in rows if r["ok"] and np.isfinite(r["err"])]
        e = np.array([r["err"] for r in acc]) if acc else np.array([])
        print("  %-16s %3d attempts | fix %3d (%3.0f%%) | err p50 %5.2f "
              "WORST %6.2f | >20 m %d | %5.0f ms p50, %5.0f p90" % (
                  matcher, len(rows), len(acc),
                  100 * len(acc) / max(len(rows), 1),
                  np.median(e) if len(e) else float("nan"),
                  e.max() if len(e) else float("nan"),
                  int((e > 20).sum()) if len(e) else 0,
                  np.median([r["ms"] for r in rows]),
                  np.percentile([r["ms"] for r in rows], 90)), flush=True)

    rows = out["xfeat,xfeat_lg"]
    base = {(r["run"], r["t"]): r for r in out["xfeat"]}
    print()
    srcs = {}
    for r in rows:
        if r["ok"]:
            srcs[r["src"]] = srcs.get(r["src"], 0) + 1
    print("  fix_source ki ginti:", srcs)
    lg = [r for r in rows if r["ok"] and r["src"] == "xfeat_lg"]
    bad = [r for r in lg if base.get((r["run"], r["t"]), {}).get("ok")]
    print("  LighterGlue ne %d fix diye; unme se %d aise frame par jahan "
          "mutual NN bhi safal tha (0 hona chahiye)" % (len(lg), len(bad)))
    e = [r["err"] for r in lg if np.isfinite(r["err"])]
    if e:
        print("  LG ke un %d fix ka error: p50 %.2f  WORST %.2f m  | "
              "unka prior p50 %.1f m" % (
                  len(e), np.median(e), max(e),
                  np.median([r["prior"] for r in lg])))
    mnn_only = len([r for r in rows if r["ok"] and r["src"] == "xfeat"])
    print("  fix rate: xfeat akela %d -> dono %d  (+%d fixes)" % (
        len([r for r in out["xfeat"] if r["ok"]]), mnn_only + len(lg),
        mnn_only + len(lg) - len([r for r in out["xfeat"] if r["ok"]])))
    slow = [r for r in rows if r["ms"] > 2000]
    print("  2000 ms (fix_period) se upar: %d/%d attempts (%.0f%%)" % (
        len(slow), len(rows), 100 * len(slow) / max(len(rows), 1)))


if __name__ == "__main__":
    main()
