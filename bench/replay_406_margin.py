#!/usr/bin/env python3
"""Replay flight 406's matcher and measure what, if anything, predicts a bad fix.

The flight offered the fuser 304 completed matches and it used ONE. The
question this answers is whether any threshold on the numbers the matcher
already reports - score, margin, inliers, the two matchers' disagreement -
separates the attempts that land near the truth from the ones that do not.

Production path, not a reimplementation: MapFixWorker._run_job with the live
config, the same gates and the same window. The prior handed in is the one
the node actually believed at that row, so the search window is where it was
in flight. Truth is gps1_north_m/gps1_east_m, never gps_north_m and never the
node's own output.

roll and pitch go in as 0: the CSV carries per-frame DELTAS, not absolute
attitude, so the nadir tilt correction cannot be reproduced. That moves every
answer by the same few metres and does not touch score or margin, which is
what is being measured here.
"""
import csv, math, os, sys, time
import numpy as np, cv2

sys.path.insert(0, '/data/DRONE/vps_vio')
from map_cache import MapCache
from map_fix_worker import MapFixWorker
import config as C

B = '/data/DRONE/flight_logs/vps/vps_00406'
OUT = '/data/DRONE/flight_logs/analysis/406_replay.csv'
cfg = C.VpsConfig()
R = list(csv.DictReader(open(B + '/vps_00406.csv')))


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

rows = [r for r in R if r.get('frame_file')
        and f(r, 'alt_m') >= cfg.map_min_alt_m and r['src'] == 'VPS']
cache = MapCache('/data/DRONE/maps/active.tif')
w = MapFixWorker(cache, focal_px=795.1, matcher=cfg.matcher,
                 window_alt_frac=cfg.map_window_alt_frac,
                 query_shape=cfg.map_query_shape,
                 min_alt_m=cfg.map_min_alt_m, use_process=False,
                 principal_xy=(644.129, 404.165))
print(f"{len(rows)} frames, matcher {w.matcher}, pair {w._pair}, "
      f"gates {w._gate}", flush=True)

fh = open(OUT, 'w', newline='')
wr = csv.writer(fh)
wr.writerow(['frame', 't', 'alt_m', 'yaw_deg', 'prior_err_m', 'ok', 'score',
             'margin', 'inliers', 'total_matches', 'disagree_m', 'fix_err_m',
             'reason'])
t_start = time.time()
for i, r in enumerate(rows):
    p = os.path.join(B, r['frame_file'])
    img = cv2.imread(p, 0)
    if img is None:
        continue
    plon, plat = f(r, 'lon'), f(r, 'lat')
    alt, yaw = f(r, 'alt_m'), f(r, 'yaw_deg')
    tlat = lat0 + f(r, 'gps1_north_m') / 110574.0
    tlon = lon0 + f(r, 'gps1_east_m') / (111320.0
                                         * math.cos(math.radians(tlat)))
    mlat = 111320.0
    mlon = 111320.0 * math.cos(math.radians(tlat))
    prior_err = math.hypot((plat - tlat) * mlat, (plon - tlon) * mlon)
    out, _, _ = w._run_job((img, plon, plat, alt, yaw, 0.0, 0.0,
                            time.monotonic(), False))
    if out is None:
        wr.writerow([r['frame_file'], f(r, 't'), alt, yaw, prior_err,
                     0, '', '', '', '', '', '', 'no result'])
    else:
        err = math.hypot((out.lat - tlat) * mlat, (out.lon - tlon) * mlon)
        wr.writerow([r['frame_file'], f(r, 't'), alt, yaw, prior_err,
                     int(bool(out.ok)), out.score, out.margin,
                     getattr(out, 'inliers', 0),
                     getattr(out, 'total_matches', 0),
                     '' if out.disagree_m is None else out.disagree_m,
                     err, getattr(out, 'reason', '')])
    if i % 50 == 0:
        fh.flush()
        print(f"  {i}/{len(rows)}  {time.time() - t_start:.0f}s", flush=True)
fh.close()
print("saved " + OUT, flush=True)
