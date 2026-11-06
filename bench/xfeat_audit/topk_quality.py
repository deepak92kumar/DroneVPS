"""top_k 1024 vs 512: does the cheaper budget still find the right place?

lg_topk.py showed the time halving. That is useless on its own - 7 frames and
one acceptance. top_k is ONE model shared by locate() and locate_lg(), so
lowering it changes XFeat's mutual-NN path too, and both have to be scored.

Scored against gps1_north_m/gps1_east_m, the raw M9N, never the EKF's fused
position - that would be our own injection coming back.
"""
import os, sys, time, statistics as st
ROOT = "/data/DRONE"
sys.path.insert(0, os.path.join(ROOT, "vps_vio"))
sys.path.insert(0, os.path.join(ROOT, "bench/xfeat_audit"))
import cv2
from replay import frames, radius_for, metres, CAM, YAW_OFFSET
from map_cache import MapCache
import map_match_xfeat as MX

fr = []
for run in ("vps_00494", "vps_00486"):
    fr += frames(run, alt_min=85.0, step=5, limit=22)
cache = MapCache(os.path.realpath(os.path.join(ROOT, "maps/active.tif")))
imgs = [(cv2.imread(f["path"], cv2.IMREAD_GRAYSCALE), f) for f in fr]
imgs = [(i, f) for i, f in imgs if i is not None]
print("%d frames, alt %.0f-%.0f m" % (len(imgs), min(f["alt"] for _, f in imgs),
                                      max(f["alt"] for _, f in imgs)))

def score(fn, tk):
    model = MX._load(); model.top_k = tk
    errs, ts, ok = [], [], 0
    for n, (img, f) in enumerate(imgs):
        win, wg = cache.window(f["est_lon"], f["est_lat"],
                               radius_for(f["alt"]), kind="grey")
        t0 = time.perf_counter()
        try:
            r = fn(img, win, wg, altitude_m=f["alt"], focal_px=CAM["fx"],
                   yaw_deg=f["yaw"] + YAW_OFFSET, map_gsd_m=cache.gsd,
                   map_gsd_x_m=getattr(cache, "gsd_x", None),
                   min_score=0.07, min_margin=0.20, window_is_edges=False,
                   principal_xy=(CAM["cx"], CAM["cy"]), query_shape="full")
        except Exception:
            r = None
        dt = time.perf_counter() - t0
        if n > 0:
            ts.append(dt)
        if r is not None and getattr(r, "ok", False):
            ok += 1
            errs.append(metres(r.lon, r.lat, f["true_lon"], f["true_lat"]))
    return ok, errs, ts

print("\n%-10s %6s %5s %8s %8s %8s %9s" % (
    "matcher", "top_k", "ok", "err p50", "err p90", "err max", "ms p50"))
for name, fn in (("xfeat", MX.locate), ("xfeat_lg", MX.locate_lg)):
    for tk in (1024, 512):
        ok, errs, ts = score(fn, tk)
        if errs:
            print("%-10s %6d %5s %8.2f %8.2f %8.2f %9.0f" % (
                name, tk, "%d/%d" % (ok, len(imgs)), st.median(errs),
                sorted(errs)[int(len(errs)*0.9)], max(errs),
                1000*st.median(ts)))
        else:
            print("%-10s %6d %5s %8s %8s %8s %9.0f" % (
                name, tk, "%d/%d" % (ok, len(imgs)), "-", "-", "-",
                1000*st.median(ts)))
