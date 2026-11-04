"""Does taking the attitude at the frame's own instant actually help?

The aircraft records the LAST ATTITUDE message to arrive, with no timestamp,
so the roll and pitch used for the nadir correction are of unknown age. ATT
runs at 10 Hz and the roll rate reaches 113 deg/s, so the question is not
academic. This replays the same frames with the attitude taken at t, and at
t minus several plausible staleness values, and scores all of them against
the M9N.

If the curve is flat, timestamping the attitude is not worth the change.
"""
import sys, os, math, time, csv as _csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import replay as R
import fclog
sys.path.insert(0, "/data/DRONE/vps_vio")
from map_cache import MapCache
from map_fix_worker import MapFixWorker
from map_match_xfeat import locate as xf_locate

OUT = "/data/DRONE/bench/xfeat_audit/att_stale.csv"
MDEG = 111320.0
cache = MapCache(os.path.realpath("/data/DRONE/maps/active.tif"))
LAGS = [0.0, 0.05, 0.10, 0.20, 0.40]

rows, t0 = [], time.time()
for run in ("vps_00476", "vps_00478"):
    fclog.attitude(run)
    fr = R.frames(run, alt_min=58, step=6)
    print(f"{run}: {len(fr)} frames", flush=True)
    for k, f in enumerate(fr):
        img = cv2.imread(f["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        rad = float(np.clip(0.60 * f["alt"], 40.0, 150.0))
        try:
            win, wg = cache.window(f["est_lon"], f["est_lat"], rad, kind="grey")
        except Exception:
            continue
        # The match itself does not depend on roll/pitch - only the nadir
        # correction does - so it is run ONCE and the correction varied.
        res = xf_locate(img, win, wg, altitude_m=f["alt"], focal_px=R.CAM["fx"],
                        yaw_deg=f["yaw"] + R.YAW_OFFSET, map_gsd_m=cache.gsd,
                        map_gsd_x_m=getattr(cache, "gsd_x", None),
                        min_score=0.07, min_margin=0.20,
                        principal_xy=(R.CAM["cx"], R.CAM["cy"]), query_shape="full")
        if not res.lon:
            continue
        rec = dict(run=run, t=f["t"], alt=f["alt"], ok=int(res.ok), ratio=res.score)
        for lag in LAGS:
            roll, pitch, _ = fclog.at(run, f["t"] - lag)
            dN, dE = MapFixWorker.nadir_offset_m(f["alt"], roll, pitch, f["yaw"])
            lat = res.lat - dN / MDEG
            lon = res.lon - dE / (MDEG * math.cos(math.radians(res.lat)))
            rec[f"err_{lag:g}"] = R.metres(lon, lat, f["true_lon"], f["true_lat"])
            if lag == 0.0:
                rec["tilt"] = math.degrees(math.acos(max(-1.0, min(1.0,
                    math.cos(math.radians(roll)) * math.cos(math.radians(pitch))))))
        # how fast the attitude was moving right then, which is what decides
        # whether staleness costs anything on THIS frame
        r1, p1, _ = fclog.at(run, f["t"] + 0.05)
        r2, p2, _ = fclog.at(run, f["t"] - 0.05)
        rec["tilt_rate"] = math.hypot(r1 - r2, p1 - p2) / 0.1
        rows.append(rec)
        if k % 25 == 0:
            print("   %3d/%d alt%6.1f tilt%5.1f rate%6.1f | " % (
                k, len(fr), f["alt"], rec["tilt"], rec["tilt_rate"]) +
                "  ".join("%.0fms %5.2f" % (l * 1000, rec[f"err_{l:g}"]) for l in LAGS),
                flush=True)

with open(OUT, "w", newline="") as fh:
    ks = sorted({k for r in rows for k in r})
    w = _csv.DictWriter(fh, fieldnames=ks); w.writeheader(); w.writerows(rows)

acc = [r for r in rows if r["ok"]]
print("\n%d frames (%d accepted).  error p50 / p90 / max by attitude staleness:" % (len(rows), len(acc)))
for nm, s in (("all", rows), ("accepted", acc)):
    print("  %-9s " % nm + " | ".join(
        "%.0fms %5.2f/%5.2f/%6.2f" % (l * 1000,
            np.median([r[f"err_{l:g}"] for r in s]),
            np.percentile([r[f"err_{l:g}"] for r in s], 90),
            max(r[f"err_{l:g}"] for r in s)) for l in LAGS))
fast = [r for r in acc if r["tilt_rate"] > 10.0]
if fast:
    print("  accepted frames with tilt rate > 10 deg/s (n=%d):" % len(fast))
    print("            " + " | ".join("%.0fms %5.2f/%5.2f/%6.2f" % (l * 1000,
        np.median([r[f"err_{l:g}"] for r in fast]),
        np.percentile([r[f"err_{l:g}"] for r in fast], 90),
        max(r[f"err_{l:g}"] for r in fast)) for l in LAGS))
print("(%.0f s)" % (time.time() - t0))
