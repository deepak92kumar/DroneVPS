#!/usr/bin/env python3
"""Does raising NCC's CLAHE clip limit actually improve the FIX, or only the
gradient number?

The raw-image gradient fell between 27 Oct and 2 Nov (39.8 -> 18.5 -> 31.3)
and the matcher fell with it, which is where "the imagery got flatter" came
from. But NCC does not see the raw image - _edges() runs CLAHE first, and
after CLAHE run 418 reads 83.2 against 27 Oct's 88.2 while still matching far
worse. So the gradient story explains 406 and not 418, and a contrast change
has to be judged on the fix, not on the gradient.

Scored against gps1_north_m / gps1_east_m. NCC only - XFeat contends with the
running node for the GPU.
"""
import csv, math, os, sys, time
import numpy as np, cv2

sys.path.insert(0, '/data/DRONE/vps_vio')
import map_match_ncc as N
from map_cache import MapCache
from map_fix_worker import MapFixWorker
import config as C

cfg = C.VpsConfig()
SHEET = '/data/DRONE/bhh.tif'


def sweep(run, clips, n_frames=20):
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
    if not rows:
        print(f"{run}: no frames above {cfg.map_min_alt_m} m")
        return
    rows = rows[::max(1, len(rows) // n_frames)][:n_frames]
    cache = MapCache(SHEET)
    print(f"\n{run}: {len(rows)} frames, alt p50 "
          f"{np.median([f(r,'alt_m') for r in rows]):.0f} m", flush=True)
    print("  %-6s %9s %8s %8s %9s" % ("clip", "fix p50", "<15m", "<5m",
                                      "prior p50"))
    orig = N._edges
    for clip in clips:
        def edges(img, notch=False, _c=clip):
            if notch:
                img = N._denotch(img)
            eq = cv2.createCLAHE(clipLimit=_c, tileGridSize=(8, 8)).apply(img)
            gx = cv2.Sobel(eq, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(eq, cv2.CV_32F, 0, 1, ksize=3)
            return cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255,
                                 cv2.NORM_MINMAX).astype(np.uint8)
        N._edges = edges
        w = MapFixWorker(cache, focal_px=795.1, matcher='ncc',
                         window_alt_frac=cfg.map_window_alt_frac,
                         query_shape=cfg.map_query_shape,
                         min_alt_m=cfg.map_min_alt_m, use_process=False,
                         principal_xy=(644.129, 404.165))
        errs, priors = [], []
        for r in rows:
            img = cv2.imread(os.path.join(B, r['frame_file']), 0)
            if img is None:
                continue
            alt = f(r, 'alt_m')
            plon, plat = f(r, 'lon'), f(r, 'lat')
            tlat = lat0 + f(r, 'gps1_north_m') / 110574.0
            tlon = lon0 + f(r, 'gps1_east_m') / (
                111320.0 * math.cos(math.radians(tlat)))
            ml = 111320.0
            mo = 111320.0 * math.cos(math.radians(tlat))
            priors.append(math.hypot((plat - tlat) * ml, (plon - tlon) * mo))
            res, _, _ = w._run_job((img, plon, plat, alt,
                                    f(r, 'yaw_deg') + cfg.camera_yaw_offset_deg,
                                    0.0, 0.0, time.monotonic(), False))
            if res is None or res.lat is None:
                continue
            errs.append(math.hypot((res.lat - tlat) * ml,
                                   (res.lon - tlon) * mo))
        e = np.array([x for x in errs if x == x])
        if not len(e):
            print("  %-6.1f   koi jawab nahi" % clip)
            continue
        print("  %-6.1f %9.1f %8d %8d %9.1f" % (
            clip, np.median(e), int((e < 15).sum()), int((e < 5).sum()),
            np.median(priors)), flush=True)
    N._edges = orig


if __name__ == "__main__":
    for run in sys.argv[1:] or ["vps_00418", "vps_00238"]:
        sweep(run, [3.0, 5.0, 8.0])
