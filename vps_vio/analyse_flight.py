#!/usr/bin/env python3
"""Measure the REAL map-matching accuracy of a recorded flight.

Every test before this matched crops of the satellite map against the map
itself, which proves the matcher works and says nothing about how the 2022
imagery compares to what the camera sees today - different season, different
crops, new buildings. This matches the actual flight frames and scores them
against the GPS that was recorded alongside, which is the number that decides
whether the map can replace the ArUco marker.

    python3 analyse_flight.py FLIGHT_DIR [--every N] [--min-alt M]
"""
from __future__ import annotations
import argparse, csv, io, os, sys, time
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_db import MapDatabase
from map_match import match_image, geo_error_m

MAP = "/data/DRONE/bhh.tif"


def read_index(flight: str):
    # The recorder can be killed mid-write, which leaves NUL padding in the CSV.
    raw = open(os.path.join(flight, "index.csv"), "rb").read().replace(b"\x00", b"")
    return list(csv.DictReader(io.StringIO(raw.decode("utf8", "replace"))))


def focal_from_camera_txt(flight: str) -> float:
    path = os.path.join(flight, "camera.txt")
    with open(path) as f:
        for tok in f.readline().split():
            if tok.startswith("fx="):
                return float(tok[3:])
    raise RuntimeError(f"no fx= in {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flight")
    ap.add_argument("--map", default=MAP)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--min-alt", type=float, default=60.0)
    ap.add_argument("--detector", default="AKAZE")
    a = ap.parse_args()

    focal = focal_from_camera_txt(a.flight)
    t = time.time()
    db = MapDatabase(a.detector).load_geotiff(a.map).build()
    print(db.summary(), f"({time.time()-t:.0f}s)", flush=True)
    print(f"camera focal {focal:.1f} px -> GSD {100/focal*100:.1f} cm/px at 100 m", flush=True)

    rows = [r for r in read_index(a.flight)
            if r.get("alt_rel_m") and r.get("lat") and r["lat"] not in ("", "0.0")
            and float(r["alt_rel_m"]) >= a.min_alt]
    rows = rows[::a.every]
    print(f"{len(rows)} frames at >= {a.min_alt:.0f} m (every {a.every}th)\n", flush=True)

    res = []
    for r in rows:
        alt, lat, lon = float(r["alt_rel_m"]), float(r["lat"]), float(r["lon"])
        img = cv2.imread(os.path.join(a.flight, "mono", r["mono_file"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        m = match_image(img, db, altitude_m=alt, focal_px=focal)
        if not m.ok:
            print(f"  --  alt {alt:6.1f}  no match: {m.reason}", flush=True)
            continue
        e = geo_error_m(m.lon, m.lat, lon, lat)
        res.append(dict(alt=alt, err=e, inl=m.inliers, tot=m.total_matches,
                        ratio=m.inlier_ratio, rot=m.rotation_deg, scale=m.scale,
                        yaw=r.get("yaw_deg"), roll=r.get("roll_deg"), pitch=r.get("pitch_deg")))
        print(f"  {'OK ' if e < 30 else 'BAD'} alt {alt:6.1f}  err {e:8.1f} m  "
              f"inliers {m.inliers:3d}/{m.total_matches:3d} ({m.inlier_ratio:.2f})  "
              f"rot {m.rotation_deg:7.1f}  scale {m.scale:.3f}", flush=True)

    print(f"\n=== {len(res)}/{len(rows)} matched ({100*len(res)/max(len(rows),1):.0f}%) ===")
    if not res:
        return
    err = np.array([r["err"] for r in res])
    for lbl, v in [("median", np.median(err)), ("mean", err.mean()),
                   ("p25", np.percentile(err, 25)), ("p75", np.percentile(err, 75)),
                   ("p90", np.percentile(err, 90)), ("min", err.min()), ("max", err.max())]:
        print(f"  {lbl:7s} {v:9.1f} m")
    print()
    for thr in (5, 10, 20, 30, 50, 100):
        print(f"  < {thr:3d} m : {100*(err < thr).mean():5.1f}%  ({int((err < thr).sum())}/{len(err)})")
    good = [r for r in res if r["err"] < 30]
    if good:
        rot = np.array([r["rot"] for r in good])
        print(f"\n  good matches: {len(good)}  median err "
              f"{np.median([r['err'] for r in good]):.1f} m  "
              f"median inliers {np.median([r['inl'] for r in good]):.0f}")
        print(f"  fitted rotation: median {np.median(rot):7.1f} deg  sd {rot.std():.1f}")
        print(f"  fitted scale   : median {np.median([r['scale'] for r in good]):.3f} (1.0 = altitude correct)")


if __name__ == "__main__":
    main()
