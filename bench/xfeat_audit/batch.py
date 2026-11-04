"""B + D: run the flown configuration over many frames, both flights, and
record everything per frame into a CSV for plotting.

Scored against the M9N, which is the only independent position there is.
Also records the geometric overlap between the camera footprint and the map
window, because that is the hypothesis under test.
"""
import sys, os, time, math, csv as _csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import replay as R
sys.path.insert(0, "/data/DRONE/vps_vio")
from map_cache import MapCache
from map_match_xfeat import locate

OUT = "/data/DRONE/bench/xfeat_audit/batch.csv"
cache = MapCache(os.path.realpath("/data/DRONE/maps/active.tif"))
W_PX, H_PX = 1280, 800


def overlap_frac(alt, err, frac=0.60):
    fw = W_PX * alt / R.CAM["fx"]; fh = H_PX * alt / R.CAM["fx"]
    box = 2 * float(np.clip(frac * alt, 40.0, 150.0))
    ox = max(0.0, min(fw / 2, box / 2 - err) + min(fw / 2, box / 2 + err))
    oy = max(0.0, min(fh / 2, box / 2) * 2)
    return (ox * oy) / (fw * fh)


rows = []
t0 = time.time()
for run, tag in (("vps_00476", "70m-good-110m"), ("vps_00478", "WP-110m-bad")):
    fr = R.frames(run, alt_min=55, step=9)          # ~every 2.5 s
    print(f"{run}: {len(fr)} frames", flush=True)
    for k, f in enumerate(fr):
        img = cv2.imread(f["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        prior_err = R.metres(f["est_lon"], f["est_lat"], f["true_lon"], f["true_lat"])
        rad = R.radius_for(f["alt"])
        try:
            win, wg = cache.window(f["est_lon"], f["est_lat"], rad, kind="grey")
        except Exception:
            continue
        t1 = time.time()
        res = locate(img, win, wg, altitude_m=f["alt"], focal_px=R.CAM["fx"],
                     yaw_deg=f["yaw"] + R.YAW_OFFSET, map_gsd_m=cache.gsd,
                     map_gsd_x_m=getattr(cache, "gsd_x", None),
                     min_score=0.07, min_margin=0.20,
                     principal_xy=(R.CAM["cx"], R.CAM["cy"]), query_shape="full")
        ms = 1000 * (time.time() - t1)
        err = (R.metres(res.lon, res.lat, f["true_lon"], f["true_lat"])
               if res.lon else float("nan"))
        rows.append(dict(run=run, tag=tag, t=f["t"], alt=f["alt"],
                         prior_err=prior_err, radius=rad,
                         overlap=overlap_frac(f["alt"], prior_err),
                         ok=int(res.ok), score=res.score, margin=res.margin,
                         inliers=res.inliers, total=res.total_matches,
                         err=err, ms=ms, reason=res.reason or ""))
        if k % 25 == 0:
            print(f"   {k}/{len(fr)}  t={f['t']:.0f} alt={f['alt']:.0f} "
                  f"ok={res.ok} score={res.score:.3f} err={err:.1f}", flush=True)
with open(OUT, "w", newline="") as fh:
    w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"\n{len(rows)} rows -> {OUT}   ({time.time()-t0:.0f} s)")
