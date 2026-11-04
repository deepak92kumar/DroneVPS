"""Is the residual error a BIAS (one direction, every frame) or scatter?

A bias survives every matcher knob and every window size - which is exactly
what the two sweeps found - and cannot be tuned away. Its candidates are the
mount yaw offset, the sheet's own georeference, and the nadir correction.
"""
import sys, os, math, csv as _csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import replay as R
sys.path.insert(0, "/data/DRONE/vps_vio")
from map_cache import MapCache
from map_match_xfeat import locate

cache = MapCache(os.path.realpath("/data/DRONE/maps/active.tif"))
MN = 111320.0
rows = []
for run in ("vps_00476", "vps_00478"):
    fr = R.frames(run, alt_min=58, step=3)
    for f in fr:
        f["prior_err"] = R.metres(f["est_lon"], f["est_lat"], f["true_lon"], f["true_lat"])
    fr = [f for f in fr if f["prior_err"] < 12.0]
    fr.sort(key=lambda f: f["alt"])
    sel = [fr[i] for i in np.linspace(0, len(fr) - 1, min(10, len(fr))).round().astype(int)]
    for f in sel:
        img = cv2.imread(f["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        rad = float(np.clip(0.60 * f["alt"], 40.0, 150.0))
        win, wg = cache.window(f["est_lon"], f["est_lat"], rad, kind="grey")
        res = locate(img, win, wg, altitude_m=f["alt"], focal_px=R.CAM["fx"],
                     yaw_deg=f["yaw"] + R.YAW_OFFSET, map_gsd_m=cache.gsd,
                     map_gsd_x_m=getattr(cache, "gsd_x", None),
                     min_score=0.07, min_margin=0.20,
                     principal_xy=(R.CAM["cx"], R.CAM["cy"]), query_shape="full")
        if not res.lon:
            continue
        me = MN * math.cos(math.radians(f["true_lat"]))
        dN = (res.lat - f["true_lat"]) * MN
        dE = (res.lon - f["true_lon"]) * me
        # the prior's own error vector, for comparison
        pN = (f["est_lat"] - f["true_lat"]) * MN
        pE = (f["est_lon"] - f["true_lon"]) * me
        rows.append(dict(run=run, t=f["t"], alt=f["alt"], ok=int(res.ok),
                         ratio=res.score, dN=dN, dE=dE, pN=pN, pE=pE,
                         yaw=f["yaw"]))
        print("%-10s alt%6.1f ok%d r%.3f | matcher dN%+7.2f dE%+7.2f (%5.2f m, %4.0f deg) | prior dN%+7.2f dE%+7.2f" % (
            run, f["alt"], res.ok, res.score, dN, dE, math.hypot(dN, dE),
            math.degrees(math.atan2(dE, dN)) % 360, pN, pE), flush=True)

a = np.array([[r["dN"], r["dE"]] for r in rows])
ok = np.array([r["ok"] for r in rows], bool)
print("\n%d frames (%d accepted)" % (len(rows), ok.sum()))
for nm, m in (("all", np.ones(len(rows), bool)), ("accepted only", ok)):
    v = a[m]
    mu = v.mean(axis=0)
    print("  %-14s mean dN %+6.2f  dE %+6.2f  -> bias %5.2f m at %3.0f deg | "
          "scatter about the mean: %5.2f m rms | raw |err| p50 %5.2f" % (
          nm, mu[0], mu[1], np.hypot(*mu), math.degrees(math.atan2(mu[1], mu[0])) % 360,
          np.sqrt(((v - mu) ** 2).sum(axis=1).mean()),
          np.median(np.hypot(v[:, 0], v[:, 1]))))
p = np.array([[r["pN"], r["pE"]] for r in rows])
print("  prior         mean dN %+6.2f  dE %+6.2f" % tuple(p.mean(axis=0)))
with open("/data/DRONE/bench/xfeat_audit/bias.csv", "w", newline="") as fh:
    w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
