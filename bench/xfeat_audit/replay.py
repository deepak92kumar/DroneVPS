"""Replay saved camera frames against the live GeoTIFF, offline.

Reproduces exactly what map_fix_worker does per frame:

    win, wg = cache.window(plon, plat, radius_for(alt), kind="grey")
    locate(img, win, wg, altitude_m=alt, focal_px=..., yaw_deg=yaw+offset,
           map_gsd_m=cache.gsd, map_gsd_x_m=cache.gsd_x,
           min_score=..., min_margin=..., principal_xy=(cx,cy),
           query_shape="full")

plon/plat is the ESTIMATE at that moment - the window is centred on where the
aircraft THINKS it is, which is the whole point of the question being asked.
Truth is gps1_north_m / gps1_east_m, the raw receiver, converted back to
lon/lat through the per-row origin the node was using.

Nothing here changes flight code. It imports it.
"""
import csv, math, os, sys, time, json
import numpy as np

ROOT = "/data/DRONE"
sys.path.insert(0, os.path.join(ROOT, "vps_vio"))

CAM = dict(fx=795.0983, cx=644.1290, cy=404.1647)
YAW_OFFSET = -4.42          # config.camera_yaw_offset_deg, as flown
MN = 111320.0


def load_run(run):
    d = os.path.join(ROOT, "flight_logs/vps", run)
    rows = list(csv.DictReader(open(os.path.join(d, f"{run}.csv"))))
    return d, rows


def fnum(x, k):
    try:
        return float(x.get(k, ""))
    except Exception:
        return float("nan")


def frames(run, alt_min=60.0, alt_max=1e9, step=1, limit=None):
    """Rows that carry a saved frame, with everything a replay needs."""
    d, rows = load_run(run)
    out = []
    for r in rows:
        ff = r.get("frame_file") or ""
        if not ff:
            continue
        a = fnum(r, "alt_m")
        if not (alt_min <= a <= alt_max):
            continue
        lat, lon = fnum(r, "lat"), fnum(r, "lon")
        n, e = fnum(r, "north_m"), fnum(r, "east_m")
        gn, ge = fnum(r, "gps1_north_m"), fnum(r, "gps1_east_m")
        if not np.isfinite([lat, lon, n, e, gn, ge]).all():
            continue
        if abs(gn) > 400 or abs(ge) > 400:
            continue
        me = MN * math.cos(math.radians(lat))
        olat, olon = lat - n / MN, lon - e / me
        out.append(dict(
            t=fnum(r, "t"), alt=a, yaw=fnum(r, "yaw_deg"),
            path=os.path.join(d, os.path.basename(ff)),
            est_lon=lon, est_lat=lat,                      # the prior
            true_lon=olon + ge / me, true_lat=olat + gn / MN,   # the M9N
            flew_fix=(r.get("fix_used") == "1"),
            flew_score=fnum(r, "fix_score"),
            flew_lon=fnum(r, "fix_lon"), flew_lat=fnum(r, "fix_lat"),
        ))
    out = out[::step]
    return out[:limit] if limit else out


def metres(lon1, lat1, lon2, lat2):
    me = MN * math.cos(math.radians(lat1))
    return math.hypot((lat2 - lat1) * MN, (lon2 - lon1) * me)


def radius_for(alt, frac=0.60, lo=40.0, hi=150.0):
    return float(np.clip(frac * alt, lo, hi))
