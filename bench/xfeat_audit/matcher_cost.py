"""What each matcher actually costs on this Orin, per call.

The question was "how much did LighterGlue add to the load", and it cannot be
answered from the flights: the worker stops at the first trusted matcher, so
LighterGlue only runs on frames XFeat failed, and 494 ran it 29 times while
489 ran it once. So time them here, on the same frames, one call at a time.

The window each matcher gets is the window the worker would give it -
_radius_for widens NCC's by ncc_search_m, and NCC starves on XFeat's radius.
Model load is excluded: the first call per matcher is a warm-up and is
reported separately, because it happens once before arming and the pilot
already sees it on the radio.

Nothing here changes flight code. It imports it.
"""
import os, sys, time, statistics as st
import numpy as np

ROOT = "/data/DRONE"
sys.path.insert(0, os.path.join(ROOT, "vps_vio"))
sys.path.insert(0, os.path.join(ROOT, "bench/xfeat_audit"))
import cv2
from replay import frames, radius_for, CAM, YAW_OFFSET
from map_cache import MapCache
from map_match_xfeat import locate as loc_xfeat, locate_lg as loc_lg
from map_match_ncc import locate as loc_ncc

GATE = {"xfeat": (0.07, 0.20), "xfeat_lg": (0.07, 0.20), "ncc": (0.05, 0.01)}
NCC_SEARCH_M = 45.0
MATCHERS = [("xfeat", loc_xfeat), ("xfeat_lg", loc_lg), ("ncc", loc_ncc)]

run = sys.argv[1] if len(sys.argv) > 1 else "vps_00494"
n_want = int(sys.argv[2]) if len(sys.argv) > 2 else 12

fr = frames(run, alt_min=85.0, step=7, limit=n_want)
if not fr:
    fr = frames(run, alt_min=60.0, step=7, limit=n_want)
print("run %s | %d frames | alt %.0f-%.0f m"
      % (run, len(fr), min(f["alt"] for f in fr), max(f["alt"] for f in fr)))

cache = MapCache(os.path.realpath(os.path.join(ROOT, "maps/active.tif")))
res = {m: [] for m, _ in MATCHERS}
warm = {}
okc = {m: 0 for m, _ in MATCHERS}

for i, f in enumerate(fr):
    img = cv2.imread(f["path"], cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    for name, fn in MATCHERS:
        r = radius_for(f["alt"])
        if name == "ncc":
            # what _radius_for does: the template has to fit AND slide
            half = 0.5 * f["alt"] * max(img.shape) / CAM["fx"]
            r = max(r, half + NCC_SEARCH_M)
        win, wg = cache.window(f["est_lon"], f["est_lat"], r, kind="grey")
        ms, mm = GATE[name]
        t0 = time.perf_counter()
        try:
            out = fn(img, win, wg, altitude_m=f["alt"], focal_px=CAM["fx"],
                     yaw_deg=f["yaw"] + YAW_OFFSET, map_gsd_m=cache.gsd,
                     map_gsd_x_m=getattr(cache, "gsd_x", None),
                     min_score=ms, min_margin=mm, window_is_edges=False,
                     principal_xy=(CAM["cx"], CAM["cy"]), query_shape="full")
        except Exception as exc:
            out = None
            if i == 0:
                print("  %s raised: %s" % (name, exc))
        dt = time.perf_counter() - t0
        if name not in warm:
            warm[name] = dt            # first call carries the model load
        else:
            res[name].append(dt)
        if out is not None and getattr(out, "ok", False):
            okc[name] += 1

print()
print("%-10s %8s %8s %8s %8s %8s  %s" % (
    "matcher", "n", "p50 ms", "mean ms", "p90 ms", "max ms", "ok / n"))
for name, _ in MATCHERS:
    v = sorted(res[name])
    if not v:
        print("%-10s  no timed calls" % name); continue
    print("%-10s %8d %8.1f %8.1f %8.1f %8.1f  %d/%d" % (
        name, len(v), 1000*st.median(v), 1000*st.mean(v),
        1000*v[int(len(v)*0.9)], 1000*max(v), okc[name], len(fr)))
print()
print("first call, model load included:")
for name, _ in MATCHERS:
    if name in warm:
        print("  %-10s %8.0f ms" % (name, 1000*warm[name]))
