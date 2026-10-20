#!/usr/bin/env python3
"""Map-match a recorded flight on the ground, and measure what the map is worth.

THE QUESTION THIS ANSWERS. Every number the map matcher has ever been judged by
on this project came from a synthetic pair - a patch cut out of the map itself,
rescaled and rotated, handed back to the matcher. map_match_xfeat's own
docstring is explicit that such a pair is "the SAME image on both sides, so it
is an upper bound on agreement in every respect". What it cannot tell anyone is
the only thing that matters in the air: how far a real camera frame, over real
trees, against a four-year-old satellite sheet, lands from where the aircraft
actually was. That gap is why min_score is still marked UNVERIFIED and why
min_inliers of 12 sits under a measured 14 with no margin.

Closing it needs frames and truth from the same flight. Until 20 Oct that was
impossible: the OAK-D permits one owner, live_node held it to navigate, and
record_flight.py could therefore never run beside it, so a flight produced
navigation or images, never both. live_node --save-frames now writes the images
from inside the navigating process, into a folder beside the estimate CSV, with
each image's name in its row's frame_file column. That row already carries the
heading and height the image was taken at and the recorded GPS beside the
estimate. This script is the consumer that was missing: it reads such a folder
back, matches every saved frame against the sheet with the same MapCache and
MapFixWorker the aircraft flies with, and prints the error against the recorded
GPS.

WHAT IT MEASURES, AND WHAT IT DOES NOT. The output's d_gps_m is the distance
from the match to the GPS position recorded on the same row. That is the figure
the exercise exists for. Two honest limits on it:

  * It is the error of the MATCH, not of the flight's estimate. The aircraft's
    own error is already in the CSV's err_m column, and map_fuse deliberately
    does not snap to a fix (snapping measured worse than no map at all: 7.0 m
    median to 12.3 with spikes to 81).
  * The match reports where the IMAGE CENTRE lands, because north_up_query
    centres its query geometrically and never reads cx. The frames are cropped
    by 130 px on the left, so the principal point sits about 80 px from the
    image centre - 11 m on the ground at 110 m altitude. The flying path has
    exactly the same offset and treats the fix as the aircraft's position, so
    this script does too, by default, and the two agree. --pp-correct applies
    the shift instead, which is how to find out whether that 11 m is really
    there; it is off by default because changing it here while the aircraft
    does something else would make these numbers describe nothing that flies.

WHAT A BENCH FOLDER CAN AND CANNOT PROVE. Measured here 20 Oct: a folder
recorded on the bench has the camera looking at the room at alt_rel ~0.1 m, so
the frames are not aerial imagery of anywhere, the altitude is far below the
60 m floor the map path refuses to run under, and the right answer is zero
matches. Such a folder proves the plumbing - the join, camera.txt, a row whose
image is missing, the summary - and nothing about accuracy. It is reported as
"no frame was above the altitude floor", which is a different sentence from
"nothing matched", on purpose.

And bhh.tif covers 30.884-30.903 N, 74.598-74.632 E, while the point the FC
last flew from reads 30.784 N - 11 km south of the sheet. A flight whose
take-off point is off the sheet produces no matches on every frame for one
reason that has nothing to do with the matcher, so that case is detected BEFORE
any matching is attempted and said plainly, with both extents printed. Reporting
it as "0 of 240 matched" would send the next person to sweep thresholds that
were never consulted.

EXISTING RECORDED FLIGHTS. This reads the vps_NNNNN/ shape live_node writes.
A record_flight folder (index.csv plus mono/) is not read directly and does not
need to be: live_node --replay FOLDER --save-frames N --out DIR runs the node
over it and writes exactly this shape, real GPS and all, so one command turns
the older recordings into an input for this script. On this machine after the
SSD loss there is one such folder left, 46 frames at -6.8 m and never armed, so
there is nothing aerial to point either path at yet.

USAGE
    python3 vps_vio/match_flight.py flight_logs/vps/vps_00004 --map bhh.tif

EXIT CODES. 0 ran and reported, 2 the take-off point is not on this sheet (no
matching was attempted), 1 the folder could not be read at all.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2                                                    # noqa: E402
from map_cache import MapCache                                # noqa: E402
from map_fix_worker import MapFixWorker                        # noqa: E402

M_PER_DEG_LAT = 111_320.0          # same constant live_node converts with

# The matcher's own floor, from map_match_xfeat: a query smaller than this is
# refused there. Checked on this side too, because the altitude that produces
# it comes from the CSV and a frame refused for being 0.3 px wide should say
# so with its altitude, not arrive as an unexplained exception from cv2.resize.
MIN_QUERY_PX = 32

# Header and row are defined together and the test runs both, for the reason
# test_csv_columns.py exists: on 20 Oct seven columns were added to live_node's
# row writer and not to its header, and 1162 rows were written 39 wide under a
# 32-wide header before anyone counted. A file that opens fine and is wrong in
# every column after the 32nd is worse than one that refuses to open.
MATCH_COLUMNS = [
    "row",              # 1-based data row in the estimate CSV (header is not a row)
    "t",                # that row's flight time, so a result can be found in the CSV
    "frame_file",
    "alt_m", "yaw_deg",         # the prior the match was given, from that row
    "vps_lon", "vps_lat",       # where the flight thought it was
    "match_lon", "match_lat",   # where the map put the image centre
    "ok",                       # did it pass the gate the aircraft flies with
    "inliers", "total_matches", "score", "margin",
    "reason",                   # why not, when not
    "d_vps_m",                  # match to the VPS estimate on the same row
    "gps_lon", "gps_lat",       # the recorded GPS, where that row has one
    "d_gps_m",                  # match to recorded GPS - THE number
    "fix_s",                    # what the fix cost
]


def match_row(r, idx, res, *, lon, lat, gps, d_vps, d_gps, reason, fix_s):
    """One result row, in MATCH_COLUMNS order. Same length as MATCH_COLUMNS."""
    def f(x, p=2):
        return "" if x is None else f"{x:.{p}f}"
    return [idx, r.get("t", ""), r.get("frame_file", ""),
            f(_num(r.get("alt_m")), 1), f(_num(r.get("yaw_deg")), 1),
            f(lon, 7), f(lat, 7),
            f(None if res is None else res.lon, 7),
            f(None if res is None else res.lat, 7),
            "" if res is None else int(res.ok),
            "" if res is None else res.inliers,
            "" if res is None else res.total_matches,
            f(None if res is None else res.score, 3),
            f(None if res is None else res.margin, 3),
            reason,
            f(d_vps), f(None if gps is None else gps[0], 7),
            f(None if gps is None else gps[1], 7), f(d_gps),
            f(fix_s, 3)]


# --------------------------------------------------------------------------- #
def read_camera(folder: str) -> dict:
    """The intrinsics the frames were taken with, from the folder's camera.txt.

    Read, never assumed. fx is not a property of the camera model alone - it is
    whatever that OAK-D's own calibration reported at 1280x800 - and cx is not
    the image centre, because live_node crops 130 px off the left where the
    aircraft's landing leg sits and moves cx with the crop. A hard-coded fx
    here would be a silent scale error in every query, which is the one error
    this matcher cannot detect: it would simply match fewer frames.

    Two shapes are written by live_node and both are accepted: the full
    "fx= fy= cx= cy= crop_left=" line when the camera reported a calibration,
    and a bare "fx=" line when it did not (a replayed folder, where
    ReplaySource knows the focal length and nothing else). D, when present,
    carries the OAK-D's 14 coefficients - the 8-term rational model plus thin
    prism, not the 5 cv2.undistort takes by default - and is read for the
    record only. It is NOT applied: the flying map path hands the raw cropped
    frame to north_up_query, flow undistorts POINTS rather than images
    (flow_odom._undistort), and undistorting here and not there would measure
    a pipeline the aircraft does not run.
    """
    path = os.path.join(folder, "camera.txt")
    if not os.path.exists(path):
        raise RuntimeError(
            f"no camera.txt in {folder} - a folder written by live_node "
            f"--save-frames always has one, holding the fx the frames were "
            f"taken with. Without it the query scale would be a guess.")
    out: dict = {"D": None}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("D="):
                out["D"] = [float(x) for x in line[2:].split()]
                continue
            for tok in line.split():
                k, _, v = tok.partition("=")
                if not _:
                    continue
                try:
                    out[k] = float(v)
                except ValueError:
                    pass
    if not out.get("fx"):
        raise RuntimeError(f"no fx= in {path} - cannot scale a query without it")
    out.setdefault("fy", out["fx"])
    out["crop_left"] = int(out.get("crop_left", 0) or 0)
    return out


def find_csv(folder: str) -> str:
    """The run's estimate CSV inside its folder.

    Named after the folder (vps_00004/vps_00004.csv) but found by looking,
    not by rebuilding the name: a folder that was renamed or copied still has
    exactly one CSV in it, and refusing a renamed folder would be pedantry.
    """
    here = [n for n in sorted(os.listdir(folder)) if n.endswith(".csv")]
    # A previous run of this script leaves its own results CSV in the folder.
    # It is not the estimate log and must never be mistaken for it.
    here = [n for n in here if not n.startswith("match")]
    # Nor is the camera IMU log, which live_node started writing beside the
    # estimate CSV on 20 Oct. Adding a second CSV to the run folder broke this
    # function the first time it was asked to read a folder recorded after
    # that change - "2 CSVs in ... say which with --csv" on a folder that has
    # exactly one estimate log in it. Anything that is not the estimate log
    # has to be excluded here, by name, as it is added.
    here = [n for n in here if not n.endswith("_imu.csv")]
    if not here:
        raise RuntimeError(f"no estimate CSV in {folder}")
    if len(here) > 1:
        raise RuntimeError(f"{len(here)} CSVs in {folder} ({', '.join(here)}) "
                           f"- say which with --csv")
    return os.path.join(folder, here[0])


def _num(s):
    """A CSV field as a float, or None. Empty is normal, not an error.

    Most optional columns are empty on most rows - err_m and the GPS pair are
    empty whenever the real receiver had no fix - so this is the common path,
    not the exceptional one.
    """
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def gps_lonlat(row) -> tuple | None:
    """The recorded GPS for a row, as lon/lat, or None if it had no fix.

    The CSV stores GPS as gps_north_m/gps_east_m: metres from the run's origin,
    and the origin itself is NOT in the file. Rather than reconstruct it - which
    would need the first row's own position and would inherit every rounding in
    it - the GPS position is taken as the estimate's own lon/lat plus the
    north/east DIFFERENCE between the two, both of which are on the row. The
    difference is a few metres, so the only error this makes is the change in
    metres-per-degree over those few metres, which is nanometres.
    """
    gn, ge = _num(row.get("gps_north_m")), _num(row.get("gps_east_m"))
    n, e = _num(row.get("north_m")), _num(row.get("east_m"))
    lon, lat = _num(row.get("lon")), _num(row.get("lat"))
    if None in (gn, ge, n, e, lon, lat):
        return None
    return (lon + (ge - e) / (M_PER_DEG_LAT * math.cos(math.radians(lat))),
            lat + (gn - n) / M_PER_DEG_LAT)


def metres_between(lon1, lat1, lon2, lat2) -> float:
    dn = (lat2 - lat1) * M_PER_DEG_LAT
    de = (lon2 - lon1) * M_PER_DEG_LAT * math.cos(math.radians(lat1))
    return float(math.hypot(dn, de))


def pp_shift(cam: dict, w: int, h: int, alt: float, yaw_deg: float):
    """Image centre -> principal point, as (d_lon_m_east, d_lat_m_north).

    north_up_query takes the query's centre to be the geometric centre of the
    frame; it never reads cx or cy. So a match reports the ground point under
    the image centre, while the aircraft is over the ground point under the
    PRINCIPAL point. After the 130 px left crop those differ by about 80 px,
    which is 11 m at 110 m altitude - a systematic bias in the same direction
    on every frame, which is the kind that never shows up as a worse match.

    The rotation is the one north_up_query implies and that this project's own
    synthesis verified on 20 Oct: a frame built by rotating a north-up ground
    patch by -yaw is recovered to 0.22-0.34 m at yaw 0, +30, -45 and +120,
    while +yaw gives 0-2 inliers of 127-175. So camera -> north-up is +yaw:

        east  =  g * ( dx cos(yaw) - dy sin(yaw))
        north = -g * ( dx sin(yaw) + dy cos(yaw))

    Sanity, yaw 0: a pixel below the image centre is south of it. Yaw 90, nose
    east: a pixel to the image's right is south. Both hold.
    """
    dx = float(cam.get("cx", w / 2.0)) - w / 2.0
    dy = float(cam.get("cy", h / 2.0)) - h / 2.0
    gx, gy = alt / float(cam["fx"]), alt / float(cam["fy"])
    th = math.radians(yaw_deg)
    east = gx * dx * math.cos(th) - gy * dy * math.sin(th)
    north = -(gx * dx * math.sin(th) + gy * dy * math.cos(th))
    return east, north


def query_px(img_shape, alt: float, focal_px: float, gsd: float) -> float:
    """How wide the query will be after north_up_query rescales it.

    Computed here so an altitude too low to produce a query at all is reported
    with its altitude, instead of reaching cv2.resize as a zero dsize and
    coming back out of MapFixWorker's except clause as no result and no reason.
    """
    scale = (alt / focal_px) / gsd
    return min(img_shape[0], img_shape[1]) * scale * 0.95


# --------------------------------------------------------------------------- #
def fix_one(worker, img, *, lon, lat, alt, yaw, timeout=120.0):
    """One fix, synchronously. Returns (FixResult or None, seconds).

    MapFixWorker is built for the flight loop: submit() returns at once and
    poll() takes whatever has finished, dropping a request while one is already
    running because a fix against a two-second-stale prior is worse than none.
    Offline there is no loop to protect and every frame is wanted, so the
    worker is driven one job at a time - submit, wait for it to stop being
    busy, take the result. It is the same thread, the same gates and the same
    window arithmetic the aircraft uses; only the pacing differs, and
    fix_period_s is set to 0 so the rate gate cannot silently skip frames.

    busy goes false and _result is set under one lock in the worker's loop, so
    once it is not busy the result is final and poll() cannot race it. A None
    result after a completed job means the worker's own except clause fired, or
    the map does not cover the prior.
    """
    t0 = time.monotonic()
    if not worker.submit(img, prior_lon=lon, prior_lat=lat,
                         altitude_m=alt, yaw_deg=yaw):
        return None, 0.0
    deadline = t0 + timeout
    while worker.busy and time.monotonic() < deadline:
        time.sleep(0.005)
    return worker.poll(), time.monotonic() - t0


def run(folder: str, map_path: str, *, out_csv: str | None = None,
        csv_path: str | None = None, every: int = 1, limit: int = 0,
        min_alt: float = 60.0, radius_m: float = 150.0,
        min_score: float | None = None, min_margin: float | None = None,
        pp_correct: bool = False, quiet: bool = False) -> dict:
    """Match a run folder and return the summary as a dict.

    Returns rather than only printing, so a test can assert on the counts
    instead of parsing the report.
    """
    def say(*a):
        if not quiet:
            print(*a, flush=True)

    csv_path = csv_path or find_csv(folder)
    cam = read_camera(folder)
    rows = list(csv.DictReader(open(csv_path, newline="")))
    named = [(i + 1, r) for i, r in enumerate(rows) if (r.get("frame_file") or "").strip()]
    named = named[::max(1, every)]
    if limit:
        named = named[:limit]

    cache = MapCache(map_path)
    say(cache.summary())
    say(f"{len(rows)} estimate rows, {len(named)} naming a frame "
        f"(fx={cam['fx']:.2f} cx={cam.get('cx', float('nan')):.2f} "
        f"crop_left={cam['crop_left']}"
        + (f", D[{len(cam['D'])}] read and NOT applied" if cam["D"] else "")
        + ")")

    summary = dict(rows=len(rows), named=len(named), missing_file=0, unreadable=0,
                   no_prior=0, off_sheet=0, below_floor=0, too_small=0,
                   attempted=0, matched=0, rejected=0, no_result=0,
                   with_gps=0, reasons={}, errs_gps=[], errs_vps=[],
                   fix_times=[], off_sheet_all=False)

    if not named:
        say("\nNo row names a frame. Either the run was not started with "
            "--save-frames, or the writer dropped every frame it was offered "
            "(the run's own summary line says which).")
        return summary

    # ----------------------------------------------------------- wrong sheet #
    # Before any matching. A take-off point off the sheet makes every frame
    # fail for one reason that is not about the matcher, and the 11 km case is
    # the normal one on this machine, not an edge case.
    priors = [(_num(r.get("lon")), _num(r.get("lat"))) for _, r in named]
    inside = [p for p in priors if None not in p and cache.covers(p[0], p[1])]
    if not inside:
        have = [p for p in priors if None not in p]
        summary["off_sheet"] = len(named)
        summary["off_sheet_all"] = True
        lon0 = lat0 = float("nan")
        if have:
            lon0, lat0 = have[0]
        west, north = cache.gt[0], cache.gt[3]
        east = west + cache.w * cache.gt[1]
        south = north + cache.h * cache.gt[5]
        say("\n*** THE FLIGHT IS NOT ON THIS SHEET - no matching was "
            "attempted ***")
        say(f"    the run starts at   {lat0:.5f} N, {lon0:.5f} E")
        say(f"    {os.path.basename(map_path)} covers "
            f"{south:.5f}-{north:.5f} N, {west:.5f}-{east:.5f} E")
        if have:
            dn = (lat0 - (south + north) / 2) * M_PER_DEG_LAT
            de = (lon0 - (west + east) / 2) * M_PER_DEG_LAT * math.cos(math.radians(lat0))
            say(f"    that is {abs(dn)/1000:.1f} km "
                f"{'north' if dn > 0 else 'south'} and {abs(de)/1000:.1f} km "
                f"{'east' if de > 0 else 'west'} of the sheet's centre")
        say("    This is NOT 'nothing matched': the matcher was never asked. "
            "Either the run's origin is wrong (live_node --origin, or "
            "SCR_USER1/2 on the FC) or this is the wrong GeoTIFF for the site.")
        return summary

    # ------------------------------------------------------------- the match #
    worker = MapFixWorker(cache, focal_px=cam["fx"], radius_m=radius_m,
                          fix_period_s=0.0, min_alt_m=min_alt,
                          min_score=min_score, min_margin=min_margin)
    out_csv = out_csv or os.path.join(folder, "match_results.csv")
    fh = open(out_csv, "w", newline="")
    wr = csv.writer(fh)
    wr.writerow(MATCH_COLUMNS)

    say(f"\nmatching {len(named)} frames, altitude floor {min_alt:.0f} m, "
        f"window +-{radius_m:.0f} m, gates score>={worker.min_score} "
        f"margin>={worker.min_margin}"
        + (", principal-point corrected" if pp_correct else "")
        + f"\nresults to {out_csv}\n")

    def note(reason):
        summary["reasons"][reason] = summary["reasons"].get(reason, 0) + 1

    for idx, r in named:
        name = (r.get("frame_file") or "").strip()
        lon, lat = _num(r.get("lon")), _num(r.get("lat"))
        alt, yaw = _num(r.get("alt_m")), _num(r.get("yaw_deg"))
        gps = gps_lonlat(r)
        if gps is not None:
            summary["with_gps"] += 1

        def emit(res=None, reason="", d_vps=None, d_gps=None, fix_s=None):
            wr.writerow(match_row(r, idx, res, lon=lon, lat=lat, gps=gps,
                                  d_vps=d_vps, d_gps=d_gps, reason=reason,
                                  fix_s=fix_s))

        path = os.path.join(folder, name)
        if not os.path.exists(path):
            # The writer can accept a frame, put its name in the row, and then
            # lose the image: a queue still full when close() ran out of time,
            # or a write that failed on a full disk. FrameWriter counts both
            # and says so out loud. So a row naming a file that is not there is
            # a KNOWN case, not corruption, and it is skipped by name.
            summary["missing_file"] += 1
            note("named frame not on disk")
            emit(reason="named frame not on disk")
            continue
        if None in (lon, lat):
            summary["no_prior"] += 1
            note("row has no position to search around")
            emit(reason="row has no position to search around")
            continue
        if alt is None or yaw is None:
            summary["no_prior"] += 1
            note("row has no altitude or heading")
            emit(reason="row has no altitude or heading")
            continue
        if not cache.covers(lon, lat):
            summary["off_sheet"] += 1
            note("prior off the sheet")
            emit(reason="prior off the sheet")
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # On disk and not an image: a half-written JPEG from a run that was
            # killed mid-encode reads back as None rather than raising.
            summary["unreadable"] += 1
            note("frame on disk but unreadable")
            emit(reason="frame on disk but unreadable")
            continue
        if alt < min_alt:
            summary["below_floor"] += 1
            note(f"altitude below the {min_alt:.0f} m floor")
            emit(reason=f"alt {alt:.1f} m is below the {min_alt:.0f} m floor")
            continue
        qpx = query_px(img.shape, alt, cam["fx"], cache.gsd)
        if qpx < MIN_QUERY_PX:
            summary["too_small"] += 1
            note("query too small at this altitude")
            emit(reason=f"at {alt:.1f} m the frame rescales to {qpx:.0f} px, "
                        f"under XFeat's {MIN_QUERY_PX} px floor")
            continue

        summary["attempted"] += 1
        res, fix_s = fix_one(worker, img, lon=lon, lat=lat, alt=alt, yaw=yaw)
        summary["fix_times"].append(fix_s)
        if res is None:
            summary["no_result"] += 1
            note("the worker returned nothing")
            emit(reason="the worker returned nothing (it swallowed an "
                        "exception, see map_fix_worker._loop)", fix_s=fix_s)
            continue

        mlon, mlat = res.lon, res.lat
        if pp_correct:
            de, dn = pp_shift(cam, img.shape[1], img.shape[0], alt, yaw)
            mlat += dn / M_PER_DEG_LAT
            mlon += de / (M_PER_DEG_LAT * math.cos(math.radians(mlat)))
            res.lon, res.lat = mlon, mlat

        d_vps = metres_between(lon, lat, mlon, mlat)
        d_gps = None if gps is None else metres_between(gps[0], gps[1], mlon, mlat)
        if res.ok:
            summary["matched"] += 1
            summary["errs_vps"].append(d_vps)
            if d_gps is not None:
                summary["errs_gps"].append(d_gps)
        else:
            summary["rejected"] += 1
            note(res.reason or "rejected by the gate")
        emit(res=res, reason="" if res.ok else (res.reason or "rejected"),
             d_vps=d_vps, d_gps=d_gps, fix_s=fix_s)

    worker.stop()
    fh.close()
    if not quiet:
        report(summary, min_alt)
    return summary


def pct(v, q):
    return float(np.percentile(np.asarray(v, float), q)) if v else float("nan")


def report(s: dict, min_alt: float):
    print("-" * 68)
    print(f"{s['named']} rows named a frame")
    for key, label in (("missing_file", "named a frame that is not on disk"),
                       ("unreadable", "on disk but not readable as an image"),
                       ("no_prior", "had no position, altitude or heading"),
                       ("off_sheet", "had a prior off the sheet"),
                       ("below_floor", f"were below the {min_alt:.0f} m altitude floor"),
                       ("too_small", "rescale to under 32 px at their altitude"),
                       ("no_result", "came back from the worker with nothing")):
        if s[key]:
            print(f"  {s[key]:5d} {label}")
    print(f"  {s['attempted']:5d} were matched against the map")
    print(f"  {s['matched']:5d} PASSED the gate the aircraft flies with, "
          f"{s['rejected']} were rejected")
    if s["fix_times"]:
        print(f"\na fix cost {pct(s['fix_times'], 50)*1000:.0f} ms median, "
              f"{max(s['fix_times'])*1000:.0f} ms worst "
              f"({sum(s['fix_times'])/60:.1f} min of CPU in all)")

    if s["reasons"]:
        print("\nwhy the rest did not count:")
        for reason, n in sorted(s["reasons"].items(), key=lambda kv: -kv[1]):
            print(f"  {n:5d}  {reason}")

    # THE number the exercise exists for.
    print()
    if s["errs_gps"]:
        e = s["errs_gps"]
        print(f"ERROR AGAINST THE RECORDED GPS, over {len(e)} matched frames:")
        print(f"  median {pct(e, 50):6.2f} m    p90 {pct(e, 90):6.2f} m    "
              f"max {max(e):6.2f} m")
    elif s["matched"]:
        print(f"{s['matched']} frames matched and NONE of them has a recorded "
              f"GPS position, so there is no error to report. The gps_north_m/"
              f"gps_east_m columns are empty whenever the real receiver had no "
              f"fix - which is the whole point of this system, and also why a "
              f"GNSS-denied flight cannot measure itself.")
    elif s["below_floor"] == s["named"] - s["missing_file"]:
        print(f"NO FRAME WAS ABOVE THE {min_alt:.0f} m ALTITUDE FLOOR, so the "
              f"matcher was never asked and there is nothing to score. This is "
              f"what a bench recording looks like: the camera sees the room at "
              f"about 0.1 m and the map path refuses to run under {min_alt:.0f} m "
              f"(map_fix_worker.min_alt_m). Not a matching failure.")
    else:
        print("Nothing matched. Unlike the two cases above this one IS about "
              "the matcher or the imagery: look at the reasons listed, and at "
              "the inlier counts in the results CSV - map_match_xfeat's "
              "min_score of 0.25 and min_inliers of 12 are both marked "
              "UNVERIFIED against real flight imagery.")
    if s["errs_vps"]:
        e = s["errs_vps"]
        print(f"\nand against the flight's own VPS estimate (not truth - the "
              f"estimate drifts, which is what the map is for):")
        print(f"  median {pct(e, 50):6.2f} m    p90 {pct(e, 90):6.2f} m    "
              f"max {max(e):6.2f} m")
    print("-" * 68)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Map-match a recorded flight folder offline and measure "
                    "the error against the recorded GPS.")
    ap.add_argument("folder", help="a run folder written by live_node "
                                   "--save-frames (flight_logs/vps/vps_NNNNN/)")
    ap.add_argument("--map", required=True, help="the satellite GeoTIFF")
    ap.add_argument("--csv", default=None,
                    help="the estimate CSV, if the folder holds more than one")
    ap.add_argument("--out", default=None,
                    help="results CSV (default match_results.csv in the folder)")
    ap.add_argument("--every", type=int, default=1, metavar="N",
                    help="match every Nth saved frame. A fix costs about "
                         "270 ms on this Orin with one torch thread, so a "
                         "20-minute flight at 2 Hz is 2400 frames and 11 "
                         "minutes of matching.")
    ap.add_argument("--limit", type=int, default=0, metavar="N",
                    help="stop after N frames")
    ap.add_argument("--min-alt", type=float, default=60.0, metavar="M",
                    help="altitude floor, default the flying path's 60 m. The "
                         "map was measured as not starting at 80 m, which is "
                         "why the floor came DOWN to 60; lowering it here "
                         "measures something the aircraft would not do.")
    ap.add_argument("--radius", type=float, default=150.0, metavar="M",
                    help="largest search window, as MapFixWorker's radius_m")
    ap.add_argument("--min-score", type=float, default=None,
                    help="inlier-ratio gate, default XFeat's 0.25. This is the "
                         "number most likely to be wrong and wrong towards "
                         "rejecting good fixes - sweeping it is what a "
                         "recorded flight is for.")
    ap.add_argument("--min-margin", type=float, default=None,
                    help="cluster-tightness gate, default 0.20")
    ap.add_argument("--pp-correct", action="store_true",
                    help="report the principal point rather than the image "
                         "centre: about 11 m at 110 m altitude after the "
                         "130 px leg crop. OFF by default because the flying "
                         "path does not do it either - see pp_shift.")
    a = ap.parse_args(argv)

    if not os.path.isdir(a.folder):
        print(f"{a.folder} is not a directory. This reads a run FOLDER "
              f"(vps_NNNNN/), the shape live_node --save-frames writes; a run "
              f"without frames wrote a bare CSV and has no images to match.")
        return 1
    try:
        s = run(a.folder, a.map, out_csv=a.out, csv_path=a.csv, every=a.every,
                limit=a.limit, min_alt=a.min_alt, radius_m=a.radius,
                min_score=a.min_score, min_margin=a.min_margin,
                pp_correct=a.pp_correct)
    except RuntimeError as e:
        print(str(e))
        return 1
    return 2 if s["off_sheet_all"] else 0


if __name__ == "__main__":
    sys.exit(main())
