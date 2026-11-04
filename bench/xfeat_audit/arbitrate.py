"""Who is right when the two matchers disagree, and what tells you so IN FLIGHT.

Both matchers on the same frames of both flights, each with the radius
map_fix_worker gives it, both nadir-corrected with absolute attitude from the
FC log, both scored against the M9N.

The point is NOT to find the better matcher. It is to find a signal that is
available on the aircraft - no truth - and that says which of the two answers
to take. Candidate signals recorded for every frame:

    each matcher's own score and margin
    the distance between the two answers              (disagree_m)
    each answer's distance from the prior it was given
    tilt, alt*tan(tilt), altitude, and the fix's own age
"""
import sys, os, math, time, csv as _csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import replay as R
import fclog
sys.path.insert(0, "/data/DRONE/vps_vio")
from map_cache import MapCache
from map_fix_worker import MapFixWorker
import map_match_xfeat as MX
import map_match_ncc as MN_

OUT = "/data/DRONE/bench/xfeat_audit/arbitrate.csv"
MDEG = 111320.0
FOCAL = R.CAM["fx"]
cache = MapCache(os.path.realpath("/data/DRONE/maps/active.tif"))
GATES = {"ncc": dict(min_score=0.05, min_margin=0.01),
         "xfeat": dict(min_score=0.07, min_margin=0.20)}


def radius(alt, matcher):
    r = 0.60 * alt
    if matcher == "ncc":
        r = max(r, 0.5 * 1280.0 * alt / FOCAL + 45.0)
    return float(np.clip(r, 40.0, 150.0))


def nadir(alt, roll, pitch, yaw):
    return MapFixWorker.nadir_offset_m(alt, roll, pitch, yaw)


rows, t0 = [], time.time()
for run in ("vps_00476", "vps_00478"):
    fclog.attitude(run)
    fr = R.frames(run, alt_min=58, step=4)
    print(f"{run}: {len(fr)} frames", flush=True)
    for k, f in enumerate(fr):
        img = cv2.imread(f["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        roll, pitch, yaw_fc = fclog.at(run, f["t"])
        prior_err = R.metres(f["est_lon"], f["est_lat"], f["true_lon"], f["true_lat"])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0,
               math.cos(math.radians(roll)) * math.cos(math.radians(pitch))))))
        dN, dE = nadir(f["alt"], roll, pitch, f["yaw"])
        rec = dict(run=run, t=f["t"], alt=f["alt"], yaw=f["yaw"],
                   roll=roll, pitch=pitch, tilt=tilt,
                   shift=math.hypot(dN, dE), prior_err=prior_err)
        got = {}
        for name, fn in (("ncc", MN_.locate), ("xfeat", MX.locate)):
            rad = radius(f["alt"], name)
            try:
                win, wg = cache.window(f["est_lon"], f["est_lat"], rad, kind="grey")
                t1 = time.time()
                res = fn(img, win, wg, altitude_m=f["alt"], focal_px=FOCAL,
                         yaw_deg=f["yaw"] + R.YAW_OFFSET, map_gsd_m=cache.gsd,
                         map_gsd_x_m=getattr(cache, "gsd_x", None),
                         principal_xy=(R.CAM["cx"], R.CAM["cy"]),
                         query_shape="full", **GATES[name])
                ms = 1000 * (time.time() - t1)
            except Exception as e:
                rec[f"{name}_ok"] = 0; rec[f"{name}_reason"] = f"raised {e}"
                continue
            lat = res.lat - dN / MDEG
            lon = res.lon - dE / (MDEG * math.cos(math.radians(res.lat or 30.9)))
            err = (R.metres(lon, lat, f["true_lon"], f["true_lat"])
                   if res.lon else float("nan"))
            frm = (R.metres(lon, lat, f["est_lon"], f["est_lat"])
                   if res.lon else float("nan"))
            rec.update({f"{name}_ok": int(res.ok), f"{name}_score": res.score,
                        f"{name}_margin": res.margin, f"{name}_err": err,
                        f"{name}_from_prior": frm, f"{name}_ms": ms,
                        f"{name}_inliers": getattr(res, "inliers", 0),
                        f"{name}_total": getattr(res, "total_matches", 0),
                        f"{name}_reason": res.reason or "",
                        f"{name}_lon": lon, f"{name}_lat": lat})
            if res.ok:
                got[name] = (lon, lat)
        rec["disagree_m"] = (R.metres(*got["ncc"], *got["xfeat"])
                             if len(got) == 2 else float("nan"))
        rows.append(rec)
        if k % 20 == 0:
            print("   %3d/%d t=%5.0f alt%6.1f tilt%5.1f | ncc ok%s e%s | xfeat ok%s e%s | dis %s" % (
                k, len(fr), f["t"], f["alt"], tilt,
                rec.get("ncc_ok"), "%.1f" % rec.get("ncc_err", float("nan")),
                rec.get("xfeat_ok"), "%.1f" % rec.get("xfeat_err", float("nan")),
                "%.1f" % rec["disagree_m"] if rec["disagree_m"] == rec["disagree_m"] else "--"),
                flush=True)

keys = sorted({k for r in rows for k in r})
with open(OUT, "w", newline="") as fh:
    w = _csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
print(f"\n{len(rows)} frames -> {OUT}  ({time.time()-t0:.0f} s)")
