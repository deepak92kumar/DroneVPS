"""Step 1b: replay the frame the matcher ACTUALLY used.

A fix is recorded on the row where it was applied, but it was computed from a
frame fix_age_s earlier (p50 0.44 s) against the prior of that moment. Pick
the saved frame nearest t - fix_age_s and take the prior from the same row.
"""
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2, csv
import replay as R
sys.path.insert(0, "/data/DRONE/vps_vio")
from map_cache import MapCache
from map_match_xfeat import locate

cache = MapCache(os.path.realpath("/data/DRONE/maps/active.tif"))

def run_one(run, n=5):
    d, rows = R.load_run(run)
    t  = np.array([R.fnum(x, "t") for x in rows])
    ff = np.array([bool(x.get("frame_file")) for x in rows])
    fl = np.array([R.fnum(x, "fix_lat") for x in rows])
    age= np.array([R.fnum(x, "fix_age_s") for x in rows])
    used = np.array([x.get("fix_used") == "1" for x in rows])
    alt= np.array([R.fnum(x, "alt_m") for x in rows])
    fi = np.where(used & np.isfinite(fl) & (alt > 60))[0][:n]
    fidx = np.where(ff)[0]
    print(f"\n===== {run} =====")
    print("   fix_t   src_t   alt   | flight ka fix        | replay            | antar | ok  score  inl/tot")
    for i in fi:
        want = t[i] - (age[i] if np.isfinite(age[i]) else 0.44)
        j = fidx[int(np.argmin(np.abs(t[fidx] - want)))]
        r = rows[j]
        lat, lon = R.fnum(r, "lat"), R.fnum(r, "lon")
        a, yaw = R.fnum(r, "alt_m"), R.fnum(r, "yaw_deg")
        img = cv2.imread(os.path.join(d, os.path.basename(r["frame_file"])), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        win, wg = cache.window(lon, lat, R.radius_for(a), kind="grey")
        res = locate(img, win, wg, altitude_m=a, focal_px=R.CAM["fx"],
                     yaw_deg=yaw + R.YAW_OFFSET, map_gsd_m=cache.gsd,
                     map_gsd_x_m=getattr(cache, "gsd_x", None),
                     min_score=0.07, min_margin=0.20,
                     principal_xy=(R.CAM["cx"], R.CAM["cy"]), query_shape="full")
        if res.lon:
            dd = R.metres(R.fnum(rows[i], "fix_lon"), fl[i], res.lon, res.lat)
            print(f"   {t[i]:7.1f} {t[j]:7.1f} {a:6.1f} | {fl[i]:.6f},{R.fnum(rows[i],'fix_lon'):.6f} "
                  f"| {res.lat:.6f},{res.lon:.6f} | {dd:5.2f} | {'Y' if res.ok else 'n'} {res.score:.3f} {res.inliers:3d}/{res.total_matches:4d}")
        else:
            print(f"   {t[i]:7.1f} {t[j]:7.1f} {a:6.1f} | fix nahi: {res.reason}")

run_one("vps_00476"); run_one("vps_00478")
