"""Step 1: does the offline replay reproduce what flew?

Picks rows where the FLIGHT produced a fix, runs the same call offline, and
compares lon/lat. If these do not agree the whole audit is built on sand.
"""
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import replay as R
sys.path.insert(0, "/data/DRONE/vps_vio")
from map_cache import MapCache
from map_match_xfeat import locate

MAP = os.path.realpath("/data/DRONE/maps/active.tif")
print(f"map: {MAP}")
t0 = time.time()
cache = MapCache(MAP)
print(f"loaded in {time.time()-t0:.1f} s   gsd {cache.gsd:.4f} / {getattr(cache,'gsd_x',None)}")

for run in ("vps_00476", "vps_00478"):
    fr = [f for f in R.frames(run, alt_min=60) if f["flew_fix"]][:5]
    print(f"\n===== {run} — {len(fr)} frames jahan flight me fix laga tha =====")
    print("   t      alt   | flight ne jo diya      | replay ne jo diya      | antar | ok score margin inl/tot  ms")
    for f in fr:
        img = cv2.imread(f["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"   {f['t']:7.1f}  frame nahi mila"); continue
        rad = R.radius_for(f["alt"])
        win, wg = cache.window(f["est_lon"], f["est_lat"], rad, kind="grey")
        t1 = time.time()
        res = locate(img, win, wg, altitude_m=f["alt"], focal_px=R.CAM["fx"],
                     yaw_deg=f["yaw"] + R.YAW_OFFSET, map_gsd_m=cache.gsd,
                     map_gsd_x_m=getattr(cache, "gsd_x", None),
                     min_score=0.07, min_margin=0.20,
                     principal_xy=(R.CAM["cx"], R.CAM["cy"]),
                     query_shape="full")
        ms = 1000 * (time.time() - t1)
        if res.lon:
            dd = R.metres(f["flew_lon"], f["flew_lat"], res.lon, res.lat)
            print(f"   {f['t']:7.1f} {f['alt']:6.1f} | {f['flew_lat']:.6f},{f['flew_lon']:.6f} "
                  f"| {res.lat:.6f},{res.lon:.6f} | {dd:5.2f} m | "
                  f"{'Y' if res.ok else 'n'} {res.score:.3f} {res.margin:.3f} "
                  f"{res.inliers:3d}/{res.total_matches:4d} {ms:5.0f}")
        else:
            print(f"   {f['t']:7.1f} {f['alt']:6.1f} | koi fix nahi: {res.reason}  ({ms:.0f} ms)")
