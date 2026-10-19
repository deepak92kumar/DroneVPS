#!/usr/bin/env python3
"""Flow odometry + map fixes + fusion on one recorded flight, scored on GPS.

Flow runs on every frame; map fixes are sampled, since one costs ~1.8 s and a
fix every couple of seconds is what a real mission would use anyway.
"""
from __future__ import annotations
import argparse, csv, io, os, sys, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flow_odom import frame_shift_px, shift_to_ne, attitude_shift_px
from map_fuse import MapFuser
from map_db import GeoRef
from map_match_ncc import locate
from osgeo import gdal


def read_index(d):
    raw = open(os.path.join(d, "index.csv"), "rb").read().replace(b"\x00", b"")
    return list(csv.DictReader(io.StringIO(raw.decode("utf8", "replace"))))


def focal_of(d):
    with open(os.path.join(d, "camera.txt")) as f:
        for t in f.readline().split():
            if t.startswith("fx="):
                return float(t[3:])
    raise RuntimeError("no fx=")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flight")
    ap.add_argument("--map", default="/data/DRONE/ap/site_2km.tif")
    ap.add_argument("--min-alt", type=float, default=60.0)
    ap.add_argument("--fix-every", type=int, default=60, help="frames between map fixes")
    ap.add_argument("--flow-every", type=int, default=1)
    a = ap.parse_args()

    ds = gdal.Open(a.map); gt = ds.GetGeoTransform()
    GEO = GeoRef.from_gdal(gt); GSD = abs(gt[5]) * 111_320.0
    FOCAL = focal_of(a.flight)

    rows = [r for r in read_index(a.flight)
            if r.get("alt_rel_m") and r.get("lat") not in (None, "", "0.0")
            and r.get("yaw_deg") and r.get("roll_deg")
            and float(r["alt_rel_m"]) >= a.min_alt][::a.flow_every]
    print(f"{len(rows)} frames >= {a.min_alt:.0f} m, focal {FOCAL:.1f}, map {GSD:.3f} m/px", flush=True)

    lat0, lon0 = float(rows[0]["lat"]), float(rows[0]["lon"])
    mlat = 111_320.0; mlon = 111_320.0 * np.cos(np.radians(lat0))
    att = lambda r: (float(r["roll_deg"]), float(r["pitch_deg"]), float(r["yaw_deg"]))

    fuser = MapFuser(flow_drift_frac=0.07, fix_sigma_m=12.0)
    flow = np.zeros(2)
    prev = pr = None
    ef, em, fixerr, track = [], [], [], []
    t0 = time.time()
    for j, r in enumerate(rows):
        img = cv2.imread(os.path.join(a.flight, "mono", r["mono_file"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        alt, yaw = float(r["alt_rel_m"]), float(r["yaw_deg"])
        dn = de = 0.0
        if prev is not None:
            d, n = frame_shift_px(prev, img)
            if d is not None:
                d = d - attitude_shift_px(att(pr), att(r))
                dn, de = shift_to_ne(d, altitude_m=alt, focal_px=FOCAL, yaw_deg=yaw)
        prev, pr = img, r
        flow = flow + np.array([dn, de])
        est = fuser.predict(dn, de)

        if j % a.fix_every == 0 and j > 0:
            plat, plon = lat0 + est[0] / mlat, lon0 + est[1] / mlon
            pc, prx = GEO.to_pixel(plon, plat); half = int(250 / GSD)
            x0, y0 = max(0, int(pc) - half), max(0, int(prx) - half)
            w = min(ds.RasterXSize - x0, 2 * half); h = min(ds.RasterYSize - y0, 2 * half)
            arr = ds.ReadAsArray(x0, y0, w, h)
            win = (cv2.cvtColor(np.dstack([arr[0], arr[1], arr[2]]), cv2.COLOR_RGB2GRAY)
                   if arr.ndim == 3 else arr)
            g2 = list(gt); g2[0] = gt[0] + x0 * gt[1]; g2[3] = gt[3] + y0 * gt[5]
            res = locate(img, win, GeoRef.from_gdal(g2), altitude_m=alt, focal_px=FOCAL,
                         yaw_deg=yaw, map_gsd_m=GSD, min_score=0.05, min_margin=0.03)
            if res.ok:
                fN, fE = (res.lat - lat0) * mlat, (res.lon - lon0) * mlon
                gN0, gE0 = (float(r["lat"]) - lat0) * mlat, (float(r["lon"]) - lon0) * mlon
                fixerr.append(np.hypot(fN - gN0, fE - gE0))
                fuser.update(fN, fE)
                est = fuser.pos

        gN, gE = (float(r["lat"]) - lat0) * mlat, (float(r["lon"]) - lon0) * mlon
        ef.append(np.hypot(flow[0] - gN, flow[1] - gE))
        em.append(np.hypot(est[0] - gN, est[1] - gE))
        track.append((flow[0], flow[1], est[0], est[1], gN, gE))
        if j % 1000 == 0:
            print(f"  {j:6d}/{len(rows)}  flow {ef[-1]:6.1f} m  fused {em[-1]:6.1f} m  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    t = np.array(track); np.save(os.path.join(a.flight, "track.npy"), t)
    ef, em = np.array(ef), np.array(em)
    dist = np.hypot(np.diff(t[:, 4]), np.diff(t[:, 5])).sum()
    print(f"\n=== {len(t)} frames, GPS path {dist:.0f} m, {time.time()-t0:.0f}s ===")
    print(f"flow only  : median {np.median(ef):6.1f}  max {ef.max():6.1f}  final {ef[-1]:6.1f}  "
          f"= {100*ef[-1]/dist:.1f}% of distance")
    print(f"flow + map : median {np.median(em):6.1f}  max {em.max():6.1f}  final {em[-1]:6.1f}")
    if fixerr:
        print(f"map fixes  : {fuser.n_fix} accepted, {fuser.n_rejected} rejected, "
              f"median error {np.median(fixerr):.1f} m")


if __name__ == "__main__":
    main()
