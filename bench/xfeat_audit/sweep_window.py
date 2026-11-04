"""Window size sweep: the only variable is how much map the matcher is shown.

Frames chosen with a SMALL prior error so window size is isolated - with a
50 m prior error the window misses the footprint no matter how big it is, and
that is a different question (already measured: corr(prior err, ratio) -0.65).

Nothing in vps_vio is touched. radius_for's cap is lifted here only so the
largest frac is not silently clipped to 150 m.
"""
import sys, os, time, math, csv as _csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import replay as R
sys.path.insert(0, "/data/DRONE/vps_vio")
from map_cache import MapCache
from map_match_xfeat import locate

OUT = "/data/DRONE/bench/xfeat_audit/sweep_window.csv"
FRACS = [0.60, 0.95, 1.15, 1.40]
N_PER_RUN = 10
cache = MapCache(os.path.realpath("/data/DRONE/maps/active.tif"))

def pick(run):
    fr = R.frames(run, alt_min=58, step=3)
    for f in fr:
        f["prior_err"] = R.metres(f["est_lon"], f["est_lat"], f["true_lon"], f["true_lat"])
    fr = [f for f in fr if f["prior_err"] < 12.0]
    fr.sort(key=lambda f: f["alt"])
    if len(fr) <= N_PER_RUN:
        return fr
    idx = np.linspace(0, len(fr) - 1, N_PER_RUN).round().astype(int)
    return [fr[i] for i in idx]          # spread across altitude

rows, t0 = [], time.time()
for run in ("vps_00476", "vps_00478"):
    sel = pick(run)
    print(f"{run}: {len(sel)} frames, alt "
          f"{min(f['alt'] for f in sel):.0f}-{max(f['alt'] for f in sel):.0f} m, "
          f"prior err p50 {np.median([f['prior_err'] for f in sel]):.1f} m", flush=True)
    for f in sel:
        img = cv2.imread(f["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        for frac in FRACS:
            rad = float(np.clip(frac * f["alt"], 40.0, 250.0))
            try:
                win, wg = cache.window(f["est_lon"], f["est_lat"], rad, kind="grey")
            except Exception as e:
                print("   window failed", frac, e, flush=True); continue
            t1 = time.time()
            res = locate(img, win, wg, altitude_m=f["alt"], focal_px=R.CAM["fx"],
                         yaw_deg=f["yaw"] + R.YAW_OFFSET, map_gsd_m=cache.gsd,
                         map_gsd_x_m=getattr(cache, "gsd_x", None),
                         min_score=0.07, min_margin=0.20,
                         principal_xy=(R.CAM["cx"], R.CAM["cy"]), query_shape="full")
            ms = 1000 * (time.time() - t1)
            err = (R.metres(res.lon, res.lat, f["true_lon"], f["true_lat"])
                   if res.lon else float("nan"))
            rows.append(dict(run=run, t=f["t"], alt=f["alt"], prior_err=f["prior_err"],
                             frac=frac, radius=rad, win_px=win.shape[1],
                             ok=int(res.ok), score=res.score, margin=res.margin,
                             inliers=res.inliers, total=res.total_matches,
                             err=err, ms=ms, reason=res.reason or ""))
        last = rows[-len(FRACS):]
        print("   alt %5.1f prior %4.1f | " % (f["alt"], f["prior_err"]) +
              " | ".join("f%.2f ok%d r%.3f n%d/%d e%s" % (
                  q["frac"], q["ok"], q["score"], q["inliers"], q["total"],
                  "%.1f" % q["err"] if q["err"] == q["err"] else "--") for q in last),
              flush=True)

with open(OUT, "w", newline="") as fh:
    w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"\n{len(rows)} rows -> {OUT}  ({time.time()-t0:.0f} s)")
