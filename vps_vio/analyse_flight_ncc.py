#!/usr/bin/env python3
"""Real map-matching accuracy for a recorded flight, using correlation.

Search window is centred on a PRIOR. Offline that prior is the recorded GPS,
offset deliberately so the matcher has to find the true position rather than
being handed it; in flight the prior is VIO. The reported error is the distance
between the correlation peak and the GPS that was recorded with the frame.
"""
from __future__ import annotations
import argparse, csv, io, os, sys, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_db import GeoRef
from map_match import geo_error_m
from map_match_ncc import locate
from osgeo import gdal


def read_index(flight):
    raw = open(os.path.join(flight, "index.csv"), "rb").read().replace(b"\x00", b"")
    return list(csv.DictReader(io.StringIO(raw.decode("utf8", "replace"))))


def focal_of(flight):
    with open(os.path.join(flight, "camera.txt")) as f:
        for tok in f.readline().split():
            if tok.startswith("fx="):
                return float(tok[3:])
    raise RuntimeError("no fx=")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flight")
    ap.add_argument("--map", default="/data/DRONE/ap/site_2km.tif")
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--min-alt", type=float, default=60.0)
    ap.add_argument("--radius", type=float, default=300.0, help="prior search radius, m")
    ap.add_argument("--prior-offset", type=float, default=120.0,
                    help="metres the prior is deliberately wrong by")
    ap.add_argument("--min-score", type=float, default=0.05)
    ap.add_argument("--min-margin", type=float, default=0.0)
    a = ap.parse_args()

    ds = gdal.Open(a.map)
    gt = ds.GetGeoTransform()
    geo = GeoRef.from_gdal(gt)
    gsd = abs(gt[5]) * 111_320.0
    focal = focal_of(a.flight)
    print(f"map {ds.RasterXSize}x{ds.RasterYSize} @ {gsd:.3f} m/px   focal {focal:.1f} px")
    print(f"prior: GPS offset by {a.prior_offset:.0f} m, searched +-{a.radius:.0f} m\n")

    rows = [r for r in read_index(a.flight)
            if r.get("alt_rel_m") and r.get("lat") not in (None, "", "0.0")
            and r.get("yaw_deg") and float(r["alt_rel_m"]) >= a.min_alt][::a.every]

    rng = np.random.default_rng(0)
    errs, scores, alts, margins, t0 = [], [], [], [], time.time()
    nfail = 0
    for r in rows:
        alt, lat, lon, yaw = (float(r["alt_rel_m"]), float(r["lat"]),
                              float(r["lon"]), float(r["yaw_deg"]))
        img = cv2.imread(os.path.join(a.flight, "mono", r["mono_file"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # Prior: true position pushed away in a random direction.
        th = rng.uniform(0, 2 * np.pi)
        dlat = a.prior_offset * np.cos(th) / 111_320.0
        dlon = a.prior_offset * np.sin(th) / (111_320.0 * np.cos(np.radians(lat)))
        pc, pr = geo.to_pixel(lon + dlon, lat + dlat)
        half = int(a.radius / gsd)
        x0, y0 = max(0, int(pc) - half), max(0, int(pr) - half)
        w = min(ds.RasterXSize - x0, 2 * half)
        h = min(ds.RasterYSize - y0, 2 * half)
        arr = ds.ReadAsArray(x0, y0, w, h)
        win = (cv2.cvtColor(np.dstack([arr[0], arr[1], arr[2]]), cv2.COLOR_RGB2GRAY)
               if arr.ndim == 3 else arr)
        g2 = list(gt); g2[0] = gt[0] + x0 * gt[1]; g2[3] = gt[3] + y0 * gt[5]

        res = locate(img, win, GeoRef.from_gdal(g2), altitude_m=alt, focal_px=focal,
                     yaw_deg=yaw, map_gsd_m=gsd,
                     min_score=a.min_score, min_margin=a.min_margin)
        if not res.ok:
            nfail += 1
            print(f"  --  alt {alt:6.1f}  {res.reason}", flush=True)
            continue
        e = geo_error_m(res.lon, res.lat, lon, lat)
        errs.append(e); scores.append(res.score); alts.append(alt); margins.append(res.margin)
        print(f"  {'OK ' if e < 30 else 'BAD'} alt {alt:6.1f}  err {e:7.1f} m  "
              f"score {res.score:.3f}  margin {res.margin:.3f}", flush=True)

    n = len(errs) + nfail
    print(f"\n=== {len(errs)}/{n} located ({100*len(errs)/max(n,1):.0f}%)  "
          f"{1000*(time.time()-t0)/max(n,1):.0f} ms/frame ===")
    if not errs:
        return
    err = np.array(errs)
    for lbl, v in [("median", np.median(err)), ("mean", err.mean()),
                   ("p25", np.percentile(err, 25)), ("p75", np.percentile(err, 75)),
                   ("p90", np.percentile(err, 90)), ("max", err.max())]:
        print(f"  {lbl:7s} {v:8.1f} m")
    print()
    for thr in (5, 10, 20, 30, 50):
        print(f"  < {thr:3d} m : {100*(err < thr).mean():5.1f}%  ({int((err<thr).sum())}/{len(err)})")
    print(f"\n  correlation score: median {np.median(scores):.3f}  min {min(scores):.3f}")
    mg = np.array(margins)
    print("\n  can the margin tell a good fix from a bad one?")
    for t in (0.02, 0.03, 0.04, 0.05, 0.06):
        k = mg >= t
        if k.sum():
            print(f"    margin >= {t:.2f} : keeps {k.sum():3d}/{len(err)} frames, "
                  f"median err {np.median(err[k]):6.1f} m, "
                  f"{100*(err[k] < 30).mean():5.1f}% within 30 m")


if __name__ == "__main__":
    main()
