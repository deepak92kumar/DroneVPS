#!/usr/bin/env python3
"""Can NCC and XFeat AGREEING tell a good map fix from a bad one?

_run_job's own comment says disagree_m is the only validated predictor of a
bad fix - corr +0.928 against the error - and on flight 406 it was never
computable, because XFeat returned ok=False on all 777 attempts and the
disagreement needs both matchers to answer.

XFeat fails its gate because the model it fits is a TRANSLATION with a 4 px
tolerance. The query is pre-rotated by the compass yaw and pre-scaled by the
barometric altitude, so any error in either spreads the correspondences: a
2 deg yaw error is 5 px at a 150 px radius, already past the tolerance. A
similarity RANSAC absorbs that - measured on this flight it takes inliers
from 0-2 to 8-15 - and it also REPORTS the residual rotation and scale.

So this asks the end-to-end question: take both answers on the same frame,
and where the two land near each other, is the answer right?
"""
import csv, math, os, sys
import numpy as np, cv2

sys.path.insert(0, '/data/DRONE/vps_vio')
from map_cache import MapCache
import map_match_xfeat as X
import map_match_ncc as N
import config as C

cfg = C.VpsConfig()
B = '/data/DRONE/flight_logs/vps/vps_00406'
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
rows = [r for r in R if r.get('frame_file') and f(r, 'alt_m') >= 70
        and r['src'] == 'VPS']
rows = rows[::max(1, len(rows) // 150)][:150]
cache = MapCache('/data/DRONE/maps/active.tif')
model = X._load()
out = open('/data/DRONE/flight_logs/analysis/406_agreement.csv', 'w',
           newline='')
wr = csv.writer(out)
wr.writerow(['frame', 'alt_m', 'prior_err_m', 'ncc_err_m', 'ncc_score',
             'xf_err_m', 'xf_inl', 'xf_tot', 'xf_rot', 'xf_scale',
             'agree_m'])
print(f"{len(rows)} frames", flush=True)
for i, r in enumerate(rows):
    img = cv2.imread(os.path.join(B, r['frame_file']), 0)
    alt, yaw = f(r, 'alt_m'), f(r, 'yaw_deg')
    plon, plat = f(r, 'lon'), f(r, 'lat')
    tlat = lat0 + f(r, 'gps1_north_m') / 110574.0
    tlon = lon0 + f(r, 'gps1_east_m') / (111320.0
                                         * math.cos(math.radians(tlat)))
    ml, mo = 111320.0, 111320.0 * math.cos(math.radians(tlat))
    prior = math.hypot((plat - tlat) * ml, (plon - tlon) * mo)

    nw, ngw = cache.window(plon, plat, alt * 0.60 + 45.0, kind='grey')
    nr = N.locate(img, nw, ngw, altitude_m=alt, focal_px=795.1, yaw_deg=yaw,
                  map_gsd_m=cache.gsd,
                  map_gsd_x_m=getattr(cache, 'gsd_x', None),
                  min_score=0.05, min_margin=0.01,
                  principal_xy=(644.129, 404.165),
                  query_shape=cfg.map_query_shape)
    ne = nsc = None
    if nr is not None and nr.lat is not None:
        ne = math.hypot((nr.lat - tlat) * ml, (nr.lon - tlon) * mo)
        nsc = nr.score

    xw, xgw = cache.window(plon, plat, alt * 0.60, kind='grey')
    q, qc = X.north_up_query(img, altitude_m=alt, focal_px=795.1,
                             yaw_deg=yaw, map_gsd_m=cache.gsd,
                             map_gsd_x_m=getattr(cache, 'gsd_x', None),
                             shape=cfg.map_query_shape,
                             track_xy=(644.129, 404.165))
    xe = xi = xt = xrot = xsc = None
    xlon = xlat = None
    mm = model.match_xfeat(X._as_3ch(X._equalise(q)),
                           X._as_3ch(X._equalise(xw)))
    if mm is not None and len(mm) >= 2:
        s, d = np.asarray(mm[0], float), np.asarray(mm[1], float)
        xt = len(s)
        if len(s) >= 4:
            M, msk = cv2.estimateAffinePartial2D(
                s.reshape(-1, 1, 2), d.reshape(-1, 1, 2), method=cv2.RANSAC,
                ransacReprojThreshold=4.0, maxIters=5000, confidence=0.999)
            if M is not None:
                xi = int(msk.sum())
                xrot = math.degrees(math.atan2(M[1, 0], M[0, 0]))
                xsc = math.hypot(M[0, 0], M[1, 0])
                mc = M @ np.array([qc[0], qc[1], 1.0])
                xlon, xlat = xgw.to_lonlat(float(mc[0]), float(mc[1]))
                xe = math.hypot((xlat - tlat) * ml, (xlon - tlon) * mo)
    ag = None
    if ne is not None and xe is not None:
        ag = math.hypot((nr.lat - xlat) * ml, (nr.lon - xlon) * mo)
    wr.writerow([r['frame_file'], alt, prior, ne, nsc, xe, xi, xt, xrot, xsc,
                 ag])
    if i % 25 == 0:
        out.flush()
        print(f"  {i}/{len(rows)}", flush=True)
out.close()
print("saved /data/DRONE/flight_logs/analysis/406_agreement.csv", flush=True)
