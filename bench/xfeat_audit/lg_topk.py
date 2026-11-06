"""Does LighterGlue's 1.9 s scale with top_k, or is it a fixed cost?

matcher_cost.py measured xfeat_lg at 1910 ms a call against XFeat's 278 on
this CPU-only Orin. LightGlue is attention over the keypoint set, so its cost
should go with top_k squared while XFeat's mutual-NN goes linearly. If so,
halving top_k is a lever; if not, there is nothing to tune and the only
choice is when to call it.
"""
import os, sys, time, statistics as st
ROOT = "/data/DRONE"
sys.path.insert(0, os.path.join(ROOT, "vps_vio"))
sys.path.insert(0, os.path.join(ROOT, "bench/xfeat_audit"))
import cv2
from replay import frames, radius_for, CAM, YAW_OFFSET
from map_cache import MapCache
import map_match_xfeat as MX

fr = frames("vps_00494", alt_min=85.0, step=7, limit=7)
cache = MapCache(os.path.realpath(os.path.join(ROOT, "maps/active.tif")))
imgs = [(cv2.imread(f["path"], cv2.IMREAD_GRAYSCALE), f) for f in fr]
imgs = [(i, f) for i, f in imgs if i is not None]
print("%d frames, alt %.0f-%.0f m" % (len(imgs), min(f["alt"] for _, f in imgs),
                                      max(f["alt"] for _, f in imgs)))
print("%8s %8s %8s %8s" % ("top_k", "p50 ms", "mean ms", "ok/n"))
for tk in (1024, 512, 256):
    # detectAndCompute reads model.top_k, so the loaded model is reused and
    # only the keypoint budget changes - no rebuild, no second set of weights.
    model = MX._load()
    model.top_k = tk
    ts, ok = [], 0
    for n, (img, f) in enumerate(imgs):
        win, wg = cache.window(f["est_lon"], f["est_lat"],
                               radius_for(f["alt"]), kind="grey")
        t0 = time.perf_counter()
        try:
            out = MX.locate_lg(img, win, wg, altitude_m=f["alt"],
                               focal_px=CAM["fx"], yaw_deg=f["yaw"] + YAW_OFFSET,
                               map_gsd_m=cache.gsd,
                               map_gsd_x_m=getattr(cache, "gsd_x", None),
                               min_score=0.07, min_margin=0.20,
                               window_is_edges=False,
                               principal_xy=(CAM["cx"], CAM["cy"]),
                               query_shape="full")
        except Exception as exc:
            out = None
            if n == 0:
                print("  raised:", exc)
        dt = time.perf_counter() - t0
        if n > 0:
            ts.append(dt)
        if out is not None and getattr(out, "ok", False):
            ok += 1
    if ts:
        print("%8d %8.0f %8.0f %8s" % (tk, 1000*st.median(ts),
                                       1000*st.mean(ts), "%d/%d" % (ok, len(imgs))))
