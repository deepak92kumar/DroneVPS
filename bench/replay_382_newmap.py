#!/usr/bin/env python3
"""Does the drone-made orthomosaic match, where the 2022 Google sheet did not?

Flight 382 (1 Nov) sits entirely inside bambattu 1, so for the first time the
matcher can be scored on a sheet made from this aircraft's own imagery.
Production path - MapFixWorker._run_job - with the prior the node actually
held at each row. Truth is gps1_north_m/gps1_east_m.

Sheets compared at several resolutions because the native 5 cm costs 14.8 s a
match, against 0.35 s on the old 51 cm sheet. roll/pitch go in as 0: the CSV
carries deltas, not absolute attitude.
"""
import csv, math, os, sys, time
import numpy as np, cv2

sys.path.insert(0, '/data/DRONE/vps_vio')
from map_cache import MapCache
from map_fix_worker import MapFixWorker
import config as C

cfg = C.VpsConfig()
B = '/data/DRONE/flight_logs/vps/vps_00382'
R = list(csv.DictReader(open(B + '/vps_00382.csv')))


def f(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError):
        return float('nan')


r0 = next(r for r in R if not math.isnan(f(r, 'lon'))
          and not math.isnan(f(r, 'north_m')))
lat0 = f(r0, 'lat') - f(r0, 'north_m') / 110574.0
lon0 = f(r0, 'lon') - f(r0, 'east_m') / (111320.0
                                         * math.cos(math.radians(f(r0, 'lat'))))
rows = [r for r in R if r.get('frame_file') and f(r, 'alt_m') >= 50]
rows = rows[::max(1, len(rows) // 120)][:120]
print(f"{len(rows)} frames, alt p50 "
      f"{np.median([f(r,'alt_m') for r in rows]):.0f} m", flush=True)

SHEETS = [('naya 0.20', '/data/DRONE/demo/bambattu1_0.20.tif'),
          ('naya 0.30', '/data/DRONE/demo/bambattu1_0.30.tif'),
          ('naya 0.50', '/data/DRONE/demo/bambattu1_0.50.tif'),
          ('purana 51', '/data/DRONE/demo/bam1.tif')]
out = open('/data/DRONE/flight_logs/analysis/382_newmap.csv', 'w', newline='')
wr = csv.writer(out)
wr.writerow(['sheet', 'frame', 'alt_m', 'prior_err_m', 'ok', 'score',
             'margin', 'inliers', 'fix_err_m', 'secs'])
for name, path in SHEETS:
    try:
        cache = MapCache(path)
    except Exception as e:
        print(f"{name}: khul nahi saka: {e}", flush=True)
        continue
    w = MapFixWorker(cache, focal_px=795.1, matcher=cfg.matcher,
                     window_alt_frac=cfg.map_window_alt_frac,
                     query_shape=cfg.map_query_shape,
                     min_alt_m=cfg.map_min_alt_m, use_process=False,
                     principal_xy=(644.129, 404.165))
    errs, oks, secs = [], 0, []
    for r in rows:
        img = cv2.imread(os.path.join(B, r['frame_file']), 0)
        if img is None:
            continue
        plon, plat = f(r, 'lon'), f(r, 'lat')
        alt, yaw = f(r, 'alt_m'), f(r, 'yaw_deg')
        tlat = lat0 + f(r, 'gps1_north_m') / 110574.0
        tlon = lon0 + f(r, 'gps1_east_m') / (111320.0
                                             * math.cos(math.radians(tlat)))
        ml, mo = 111320.0, 111320.0 * math.cos(math.radians(tlat))
        prior = math.hypot((plat - tlat) * ml, (plon - tlon) * mo)
        t0 = time.time()
        res, _, _ = w._run_job((img, plon, plat, alt, yaw, 0.0, 0.0,
                                time.monotonic(), False))
        dt = time.time() - t0
        secs.append(dt)
        if res is None:
            wr.writerow([name, r['frame_file'], alt, prior, 0, '', '', '',
                         '', dt])
            continue
        e = math.hypot((res.lat - tlat) * ml, (res.lon - tlon) * mo)
        errs.append((e, prior, bool(res.ok)))
        oks += int(bool(res.ok))
        wr.writerow([name, r['frame_file'], alt, prior, int(bool(res.ok)),
                     res.score, res.margin, getattr(res, 'inliers', 0), e, dt])
    out.flush()
    if errs:
        a = np.array([x[0] for x in errs])
        p = np.array([x[1] for x in errs])
        print(f"{name:10s} n={len(errs):3d} ok={oks:3d}  prior p50 "
              f"{np.median(p):5.1f}  fix p50 {np.median(a):5.1f} m  "
              f"<15m {int((a<15).sum()):3d}  <5m {int((a<5).sum()):3d}  "
              f"{np.mean(secs):.2f} s/match", flush=True)
out.close()
print("saved /data/DRONE/flight_logs/analysis/382_newmap.csv", flush=True)
