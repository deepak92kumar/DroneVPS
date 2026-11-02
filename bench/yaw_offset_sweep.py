#!/usr/bin/env python3
"""Does camera_yaw_offset_deg explain the matcher regression?

The matcher worked on 27 Oct on bhh.tif at this location - run 238 applied 73
fixes with the two matchers 2.0 m apart - and today on the SAME sheet at the
SAME place it applies 59-79 with them 10-30 m apart. Between those dates
camera_yaw_offset_deg went -1.95 -> 0.0 (1 Nov, at the pilot's asking). The
query is derotated to north-up by yaw before matching, so a yaw error is a
rotation error on every template: 1.95 deg is 5 px at a 150 px radius.

This sweeps the offset through the production path and scores the answer
against gps1_north_m/gps1_east_m, on real frames from both flights.
"""
import csv, math, os, sys, time
import numpy as np, cv2

sys.path.insert(0, '/data/DRONE/vps_vio')
from map_cache import MapCache
from map_fix_worker import MapFixWorker
import config as C

cfg = C.VpsConfig()
SHEET = '/data/DRONE/bhh.tif'


def load(run):
    B = f'/data/DRONE/flight_logs/vps/{run}'
    R = list(csv.DictReader(open(f'{B}/{run}.csv')))

    def f(r, k):
        try:
            return float(r[k])
        except (TypeError, ValueError):
            return float('nan')

    r0 = next(r for r in R if not math.isnan(f(r, 'lon'))
              and not math.isnan(f(r, 'north_m')))
    lat0 = f(r0, 'lat') - f(r0, 'north_m') / 110574.0
    lon0 = f(r0, 'lon') - f(r0, 'east_m') / (
        111320.0 * math.cos(math.radians(f(r0, 'lat'))))
    rows = [r for r in R if r.get('frame_file')
            and f(r, 'alt_m') >= cfg.map_min_alt_m]
    return B, rows, lat0, lon0, f


def run_sweep(run, offsets, n_frames=20):
    B, rows, lat0, lon0, f = load(run)
    if not rows:
        print(f"{run}: alt >= {cfg.map_min_alt_m} par koi saved frame nahi")
        return
    rows = rows[::max(1, len(rows) // n_frames)][:n_frames]
    cache = MapCache(SHEET)
    print(f"\n{run}: {len(rows)} frames, alt p50 "
          f"{np.median([f(r,'alt_m') for r in rows]):.0f} m")
    print("  %-10s %8s %9s %7s %7s %9s" % ("yaw off", "fix p50", "prior p50",
                                           "<15m", "<5m", "disagree"))
    for off in offsets:
        w = MapFixWorker(cache, focal_px=795.1, matcher=cfg.matcher,
                         window_alt_frac=cfg.map_window_alt_frac,
                         query_shape=cfg.map_query_shape,
                         min_alt_m=cfg.map_min_alt_m, use_process=False,
                         principal_xy=(644.129, 404.165))
        errs, priors, dis = [], [], []
        print("    %+.2f ..." % off, flush=True)
        for r in rows:
            img = cv2.imread(os.path.join(B, r['frame_file']), 0)
            if img is None:
                continue
            alt = f(r, 'alt_m')
            yaw = f(r, 'yaw_deg') + off
            plon, plat = f(r, 'lon'), f(r, 'lat')
            tlat = lat0 + f(r, 'gps1_north_m') / 110574.0
            tlon = lon0 + f(r, 'gps1_east_m') / (
                111320.0 * math.cos(math.radians(tlat)))
            ml = 111320.0
            mo = 111320.0 * math.cos(math.radians(tlat))
            priors.append(math.hypot((plat - tlat) * ml, (plon - tlon) * mo))
            res, _, _ = w._run_job((img, plon, plat, alt, yaw, 0.0, 0.0,
                                    time.monotonic(), False))
            if res is None or res.lat is None:
                continue
            errs.append(math.hypot((res.lat - tlat) * ml,
                                   (res.lon - tlon) * mo))
            if getattr(res, 'disagree_m', None) is not None:
                dis.append(res.disagree_m)
        if not errs:
            print("  %-10.2f  koi jawab nahi" % off)
            continue
        e = np.array(errs)
        print("  %-10.2f %8.1f %9.1f %7d %7d %9s" % (
            off, np.median(e), np.median(priors), int((e < 15).sum()),
            int((e < 5).sum()),
            "%.1f" % np.median(dis) if dis else "-"))


if __name__ == "__main__":
    OFF = [0.0, -1.95]
    for run in sys.argv[1:] or ["vps_00418", "vps_00238"]:
        run_sweep(run, OFF)
