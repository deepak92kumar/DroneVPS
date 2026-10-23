#!/usr/bin/env python3
"""Measure the REAL map-matching accuracy of a recorded flight.

Every test before this matched crops of the satellite map against the map
itself, which proves the matcher works and says nothing about how the 2022
imagery compares to what the camera sees today - different season, different
crops, new buildings. This matches the actual flight frames and scores them
against the GPS that was recorded alongside, which is the number that decides
whether the map can replace the ArUco marker.

    python3 analyse_flight.py FLIGHT_DIR [--every N] [--min-alt M] [--out CSV]

TWO FOLDER SHAPES, because there are now two recorders:

  record_flight.py   DIR/index.csv  +  DIR/mono/mNNNNNN.png  +  camera.txt
  live_node.py       DIR/vps_NNNNN.csv  +  DIR/fNNNNNN.jpg   +  camera.txt

The second shape is new on 20 Oct and is the one a real flight now produces.
The OAK-D permits ONE owner, so record_flight.py cannot run while the node is
navigating; until live_node learned to write frames itself (--save-frames,
frame_writer.py) no flight came back with images at all, and this tool had
nothing to run on but bench recordings. Both shapes carry a heading and a
height per image, which is all the matcher needs as a prior.

WHAT "ERROR" MEANS IN EACH SHAPE. record_flight's index.csv records the
aircraft's own GPS, so the error is against that and nothing else. live_node's
CSV has THREE positions per row, and picking the wrong one is how 23 Oct's
conclusions were drawn and then withdrawn:

  lat / lon                   the VPS ESTIMATE - what the node believed
  gps1_north_m / gps1_east_m  GPS_RAW_INT, the receiver BEFORE the EKF.
                              The only independent truth in the file.
  gps_north_m / gps_east_m    GLOBAL_POSITION_INT, the EKF's FUSED position.
                              On a flight that injects, this is driven by what
                              WE send, so scoring against it compares the VPS
                              with itself. The two agree to 0.10 m while src
                              is GPS1 and differ by 5.59 m p50 while src is
                              VPS - which is exactly the part anyone cares
                              about.

So gps1_* is used when the row has it and gps_* is the fallback for
recordings made before those columns existed, which is what match_flight.py
already did; this module was simply never updated to match. Every run says
which it used rather than implying it. Rows with no fix carry no truth, and
are counted rather than quietly dropped.

The honest limit, unchanged: the receiver itself wandered 2.4 to 27.4 m on the
ground (10 Oct), and the surveyed point asked for on three separate days has
still not been taken. Independent is not the same as correct.
"""
from __future__ import annotations
import argparse, csv, glob, io, os, sys, time
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_db import MapDatabase
from map_match import match_image, geo_error_m

MAP = "/data/DRONE/bhh.tif"
M_PER_DEG_LAT = 111_320.0


def _rows(path: str):
    # Either recorder can be killed mid-write, which leaves NUL padding in the
    # CSV. A file that ends in NULs is not a corrupt flight, it is a flight
    # that was still being written when the power went off.
    raw = open(path, "rb").read().replace(b"\x00", b"")
    return list(csv.DictReader(io.StringIO(raw.decode("utf8", "replace"))))


def _f(row, key):
    """A float, or None for the empty string the CSVs use for 'not known'."""
    v = row.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def read_flight(folder: str):
    """(shape, frames, notes) for either recorder's folder.

    Each frame is a dict: path, alt, yaw, truth (lon, lat) or None, est
    (lon, lat) or None, label. Nothing is read off the disk here but the CSV -
    the images are opened one at a time by the caller, because a flight is
    thousands of frames and they do not fit in RAM at once.
    """
    notes = []
    index = os.path.join(folder, "index.csv")
    if os.path.exists(index):
        frames = []
        for r in _rows(index):
            alt = _f(r, "alt_rel_m")
            name = r.get("mono_file") or ""
            if alt is None or not name:
                continue
            lat, lon = _f(r, "lat"), _f(r, "lon")
            truth = None if lat in (None, 0.0) or lon in (None, 0.0) else (lon, lat)
            frames.append(dict(path=os.path.join(folder, "mono", name), alt=alt,
                               yaw=_f(r, "yaw_deg"), truth=truth, est=None,
                               label=name))
        return "record_flight", frames, notes

    csvs = sorted(glob.glob(os.path.join(folder, "vps_*.csv")))
    if not csvs:
        raise RuntimeError(
            f"{folder} is neither recorder's folder: no index.csv "
            f"(record_flight) and no vps_NNNNN.csv (live_node --save-frames)")
    if len(csvs) > 1:
        notes.append(f"{len(csvs)} vps_*.csv here; reading {os.path.basename(csvs[0])}")
    rows = _rows(csvs[0])

    # The node's origin is not written in the CSV, but it is recoverable
    # exactly: every row's lat/lon IS the origin plus its own north/east, so
    # one row with both gives the origin back. That is what turns the
    # receiver's gps_north_m/gps_east_m into a lat/lon to score against.
    origin = None
    for r in rows:
        lat, lon = _f(r, "lat"), _f(r, "lon")
        n, e = _f(r, "north_m"), _f(r, "east_m")
        if None not in (lat, lon, n, e):
            olat = lat - n / M_PER_DEG_LAT
            mlon = M_PER_DEG_LAT * float(np.cos(np.radians(olat)))
            origin = (lon - e / mlon, olat, mlon)
            break
    if origin is None:
        notes.append("no row carries both a position and a north/east, so the "
                     "origin could not be recovered: the receiver's metres "
                     "cannot be turned into a position and there is no truth "
                     "to score against")

    frames, no_truth = [], 0
    n_raw = n_fused = 0
    for r in rows:
        name = r.get("frame_file") or ""
        if not name:
            continue                      # most rows: the rate gate saved none
        alt = _f(r, "alt_m")
        # The RAW receiver first. See the module docstring: the fused pair
        # measures the VPS against itself on any flight that injects.
        gn, ge = _f(r, "gps1_north_m"), _f(r, "gps1_east_m")
        if None not in (gn, ge):
            n_raw += 1
        else:
            gn, ge = _f(r, "gps_north_m"), _f(r, "gps_east_m")
            if None not in (gn, ge):
                n_fused += 1
        truth = None
        if origin is not None and None not in (gn, ge):
            truth = (origin[0] + ge / origin[2], origin[1] + gn / M_PER_DEG_LAT)
        if truth is None:
            no_truth += 1
        lat, lon = _f(r, "lat"), _f(r, "lon")
        frames.append(dict(path=os.path.join(folder, name),
                           alt=0.0 if alt is None else alt,
                           yaw=_f(r, "yaw_deg"), truth=truth,
                           est=None if None in (lat, lon) else (lon, lat),
                           label=name, t=r.get("t"), src=r.get("src")))
    if no_truth:
        notes.append(f"{no_truth} of {len(frames)} frames have no receiver fix "
                     f"on their row - matched but not scored")
    # SAY WHAT THIS WAS SCORED AGAINST. A reader cannot tell otherwise, and on
    # 23 Oct that is precisely what went wrong.
    if n_raw:
        notes.append(f"truth: GPS_RAW_INT, the raw receiver ({n_raw} frames"
                     + (f"; {n_fused} fell back to the EKF's fused position)"
                        if n_fused else ")"))
    elif n_fused:
        notes.append(f"truth: GLOBAL_POSITION_INT, the EKF's FUSED position "
                     f"({n_fused} frames) - independent ONLY where src was "
                     f"GPS1, because on the VPS it is driven by our own "
                     f"injection")
    else:
        notes.append("truth: none - no position to compare against")
    return "live_node", frames, notes


def focal_from_camera_txt(folder: str) -> float:
    """fx, in pixels. Both recorders write the same first line.

    live_node's is already crop-adjusted: the left 130 px where the aircraft's
    leg sits are cut off before anything sees the frame, so the saved images
    are 1150 wide and the cx in this file has moved with them. fx does not.
    """
    path = os.path.join(folder, "camera.txt")
    with open(path) as f:
        for tok in f.readline().split():
            if tok.startswith("fx="):
                return float(tok[3:])
    raise RuntimeError(f"no fx= in {path}")


def sheet_covers(db, lon: float, lat: float) -> bool:
    h, w = db.image.shape[:2]
    lon0, lat0 = db.geo.to_lonlat(0, 0)
    lon1, lat1 = db.geo.to_lonlat(w, h)
    return (min(lon0, lon1) <= lon <= max(lon0, lon1)
            and min(lat0, lat1) <= lat <= max(lat0, lat1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flight", help="a record_flight folder or a live_node "
                                   "--save-frames run folder")
    ap.add_argument("--map", default=MAP)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--min-alt", type=float, default=60.0,
                    help="map matching needs roughly 80 m of height; below "
                         "that a frame is only useful to VIO")
    ap.add_argument("--detector", default="AKAZE")
    ap.add_argument("--out", help="per-image results CSV")
    a = ap.parse_args()

    shape, frames, notes = read_flight(a.flight)
    print(f"{a.flight}: a {shape} folder, {len(frames)} frames named")
    for n in notes:
        print(f"  note: {n}")
    focal = focal_from_camera_txt(a.flight)

    t = time.time()
    db = MapDatabase(a.detector).load_geotiff(a.map).build()
    print(db.summary(), f"({time.time()-t:.0f}s)", flush=True)
    print(f"camera focal {focal:.1f} px -> GSD {100/focal*100:.1f} cm/px at 100 m",
          flush=True)

    # Is this flight even on this sheet? A take-off point outside it means no
    # match for the whole flight, and that has to be said as "wrong sheet",
    # not reported as a matcher that scored zero. bhh.tif does not cover the
    # point the FC last flew from: the sheet is 30.884-30.903 N / 74.598-
    # 74.632 E and SCR_USER1/2 in log 00000195.BIN reads 30.7844, 74.6072,
    # which is 11.1 km south of it.
    anchor = next((f["truth"] or f["est"] for f in frames
                   if f["truth"] or f["est"]), None)
    if anchor is None:
        print("\n*** no position on any frame's row - nothing to score "
              "against, and no way to tell whether this flight is on this "
              "sheet. Matching anyway. ***")
    elif not sheet_covers(db, *anchor):
        lon0, lat0 = db.geo.to_lonlat(0, 0)
        h, w = db.image.shape[:2]
        lon1, lat1 = db.geo.to_lonlat(w, h)
        print(f"\n*** THE WRONG SHEET. This flight is at "
              f"{anchor[1]:.5f}, {anchor[0]:.5f}; {os.path.basename(a.map)} "
              f"spans {min(lat0,lat1):.5f}-{max(lat0,lat1):.5f} N, "
              f"{min(lon0,lon1):.5f}-{max(lon0,lon1):.5f} E - "
              f"{geo_error_m(anchor[0], anchor[1], (lon0+lon1)/2, (lat0+lat1)/2)/1000:.1f} km "
              f"from its centre. Expect no matches at all: that is the sheet, "
              f"not the matcher. Pass --map for this site. ***")

    usable = [f for f in frames if f["alt"] >= a.min_alt][::a.every]
    print(f"\n{len(usable)} frames at >= {a.min_alt:.0f} m (every {a.every}th "
          f"of {len(frames)})\n", flush=True)
    if not usable:
        print(f"nothing to match: the highest frame here is "
              f"{max((f['alt'] for f in frames), default=0.0):.1f} m. Map "
              f"matching wants ~80 m; a bench recording never reaches it.")
        return

    res, gone, nomatch = [], 0, 0
    for f in usable:
        img = cv2.imread(f["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            # The writer drops a frame rather than stalling the flow loop, and
            # can lose a queued one at shutdown - it counts both and says so.
            # A named frame that is not on disk is therefore expected, not a
            # corrupt folder.
            gone += 1
            continue
        m = match_image(img, db, altitude_m=f["alt"], focal_px=focal)
        if not m.ok:
            nomatch += 1
            print(f"  --  alt {f['alt']:6.1f}  no match: {m.reason}", flush=True)
            continue
        err = None if f["truth"] is None else geo_error_m(m.lon, m.lat, *f["truth"])
        off = None if f["est"] is None else geo_error_m(m.lon, m.lat, *f["est"])
        res.append(dict(label=f["label"], alt=f["alt"], yaw=f["yaw"],
                        lon=m.lon, lat=m.lat, err=err, est_off=off,
                        inl=m.inliers, tot=m.total_matches,
                        ratio=m.inlier_ratio, rot=m.rotation_deg,
                        scale=m.scale))
        print(f"  {'OK ' if (err is not None and err < 30) else '?  '} "
              f"alt {f['alt']:6.1f}  "
              + ("err      -  " if err is None else f"err {err:8.1f} m") +
              f"  inliers {m.inliers:3d}/{m.total_matches:3d} "
              f"({m.inlier_ratio:.2f})  rot {m.rotation_deg:7.1f}  "
              f"scale {m.scale:.3f}"
              + ("" if off is None else f"  vs VPS {off:7.1f} m"), flush=True)

    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame_file", "alt_m", "yaw_deg", "match_lon",
                        "match_lat", "err_vs_gps_m", "offset_vs_vps_m",
                        "inliers", "total_matches", "inlier_ratio",
                        "rotation_deg", "scale"])
            for r in res:
                w.writerow([r["label"], f"{r['alt']:.1f}",
                            "" if r["yaw"] is None else f"{r['yaw']:.1f}",
                            f"{r['lon']:.7f}", f"{r['lat']:.7f}",
                            "" if r["err"] is None else f"{r['err']:.2f}",
                            "" if r["est_off"] is None else f"{r['est_off']:.2f}",
                            r["inl"], r["tot"], f"{r['ratio']:.3f}",
                            f"{r['rot']:.2f}", f"{r['scale']:.4f}"])
        print(f"\nper-image results -> {a.out}")

    print(f"\n=== {len(res)}/{len(usable)} matched "
          f"({100*len(res)/max(len(usable),1):.0f}%) ===")
    if gone:
        print(f"  {gone} named frames were not on disk (dropped by the writer)")
    if nomatch:
        print(f"  {nomatch} frames the matcher refused")
    scored = [r for r in res if r["err"] is not None]
    if not res:
        print("  nothing matched, so there is no accuracy number here. Check "
              "the sheet and the height before the matcher: both are a "
              "likelier cause than the thresholds.")
        return
    if not scored:
        print("  none of the matches has a receiver fix to be scored against, "
              "so there is NO accuracy number here - only that the matcher "
              "found something.")
        return
    err = np.array([r["err"] for r in scored])
    print(f"  scored against the receiver on {len(scored)} of them:")
    for lbl, v in [("median", np.median(err)), ("mean", err.mean()),
                   ("p25", np.percentile(err, 25)), ("p75", np.percentile(err, 75)),
                   ("p90", np.percentile(err, 90)), ("min", err.min()), ("max", err.max())]:
        print(f"  {lbl:7s} {v:9.1f} m")
    print()
    for thr in (5, 10, 20, 30, 50, 100):
        print(f"  < {thr:3d} m : {100*(err < thr).mean():5.1f}%  "
              f"({int((err < thr).sum())}/{len(err)})")
    good = [r for r in scored if r["err"] < 30]
    if good:
        rot = np.array([r["rot"] for r in good])
        print(f"\n  good matches: {len(good)}  median err "
              f"{np.median([r['err'] for r in good]):.1f} m  "
              f"median inliers {np.median([r['inl'] for r in good]):.0f}")
        print(f"  fitted rotation: median {np.median(rot):7.1f} deg  sd {rot.std():.1f}")
        print(f"  fitted scale   : median {np.median([r['scale'] for r in good]):.3f} "
              f"(1.0 = altitude correct)")
    vs = [r["est_off"] for r in res if r["est_off"] is not None]
    if vs:
        print(f"\n  against what the NODE believed at the same instant: median "
              f"{np.median(vs):.1f} m over {len(vs)} frames")


if __name__ == "__main__":
    main()
