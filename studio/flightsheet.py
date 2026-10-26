#!/usr/bin/env python3
"""Choosing a cut sheet to fly, and proving it will actually open.

"Cut it, then check it loaded, and if it did not, say what is wrong" - this
file is the second half of that. The check is not a re-reading of the
GeoTIFF's header by this program's own rules: it constructs the real
`vps_vio.map_cache.MapCache`, the same class the flying node constructs at
startup, and then slices a window out of it the way the matcher does. If that
works here it will work in the air, and if it does not, it fails here on the
ground where the message can be read.

Nothing in vps_vio is modified by any of this. The node is read, never
written: `map_cache.py` is imported to run the real check, and the choice of
sheet is a symlink the pilot passes with --map.

Every check that can fail carries the remedy with it. A red line that says
only "CRS is not EPSG:4326" is a puzzle; one that says "cut it again from the
Map tab, which reprojects it" is an instruction.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

import geotiff

VPS_VIO = "/data/DRONE/vps_vio"
FLOWN_GSD = 0.5092
MAPS_DIR = "/data/DRONE/maps"
# The chosen sheet is a SYMLINK next to the cuts, and nothing else. Nothing in
# vps_vio is touched: the node is flown with
#
#     python3 vps_vio/live_node.py --map /data/DRONE/maps/active.tif
#
# and "Use this sheet" repoints that one link, so a new site is a click here
# and the same command as always on the aircraft.
LINK = os.path.join(MAPS_DIR, "active.tif")
# MapCache holds the WHOLE sheet in RAM as one greyscale array, on purpose.
# This machine has 8 GB shared with the camera stack and TensorRT, so a sheet
# past this size is a problem to find on the ground, not in the air.
RAM_LIMIT_MB = 700.0
FLY_CMD = "python3 vps_vio/live_node.py --map " + LINK


def active():
    """Which sheet the link points at, if any."""
    out = dict(link=LINK, path=None, exists=False, fly=FLY_CMD)
    if os.path.islink(LINK):
        target = os.path.realpath(LINK)
        out["path"] = target
        out["exists"] = os.path.exists(target)
        if out["exists"]:
            try:
                out["sheet"] = geotiff.Sheet(target).describe()
            except Exception as e:                       # noqa: BLE001
                out["error"] = f"{type(e).__name__}: {e}"
    return out


def set_active(path: str) -> str:
    """Point maps/active.tif at this sheet. Returns the absolute path."""
    p = os.path.abspath(path)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    os.makedirs(MAPS_DIR, exist_ok=True)
    if os.path.islink(LINK) or os.path.exists(LINK):
        os.unlink(LINK)
    os.symlink(p, LINK)
    return p


def clear_active():
    if os.path.islink(LINK) or os.path.exists(LINK):
        os.unlink(LINK)


def load_check(path: str, takeoff=None):
    """Open it the way the node does and report every step, with remedies."""
    checks = []
    t_start = time.time()

    def ck(ok, name, detail, fix="", hard=True):
        # The remedy belongs to the failure. Printing it beside a green line
        # turns a report that should read at a glance into a wall of advice.
        checks.append(dict(ok=bool(ok), name=name, detail=detail,
                           fix="" if ok else fix, hard=hard))
        return bool(ok)

    path = os.path.abspath(path)
    if not ck(os.path.exists(path), "the file is on this machine", path,
              "Cut the sheet again, or copy it back onto the Jetson."):
        return _done(checks, t_start, None)

    size_mb = os.path.getsize(path) / 1e6
    ck(size_mb > 0.01, "the file is not empty", f"{size_mb:.1f} MB",
       "A zero-length file is a copy that was interrupted. Copy it again.")

    # ---- the header, by the rules map_cache.py assumes and never checks
    try:
        rep = geotiff.verify(path, takeoff=takeoff, flying_gsd=FLOWN_GSD)
    except Exception as e:                               # noqa: BLE001
        ck(False, "GDAL can open it", f"{type(e).__name__}: {e}",
           "It is not a GeoTIFF GDAL understands. Cut it from the Map tab "
           "rather than converting it by hand.")
        return _done(checks, t_start, None)
    ck(True, "GDAL opens it",
       f"{rep['width']} x {rep['height']} px, {rep['bands']} x {rep['dtype']}, "
       f"{rep['file_mb']:.1f} MB")

    FIXES = {
        "CRS is geographic WGS 84 (EPSG:4326)":
            "map_cache multiplies pixel size by 111 320 m/deg, so a projected "
            "sheet reads 111 320x too coarse and no fix will ever match. Cut "
            "it again from the Map tab - the cut reprojects it.",
        "north-up: no rotation in the geotransform":
            "GeoRef.from_gdal drops the two rotation terms, so a rotated sheet "
            "is not detected in flight, it is silently mis-georeferenced - and "
            "the matcher already rotates the camera frame north-up by the "
            "aircraft's heading, so the two rotations add instead of "
            "cancelling. Cut it again from the Map tab, which rectifies it.",
        "origin at the top-left (row increases southward)":
            "With a positive gt[5] every latitude the matcher reports is "
            "mirrored about the sheet's origin. Cut it again from the Map tab.",
        "column increases eastward":
            "Longitude would run backwards. Cut it again from the Map tab.",
        "1 or 3 bands, which MapCache can grey-scale":
            "MapCache greyscales 3 bands and takes 1 as it is; anything else "
            "it cannot read. Re-cut choosing colour or greyscale.",
        "8-bit samples":
            "Re-cut it: the cut writes 8-bit whatever the source was.",
        "fits in RAM as one greyscale array":
            "MapCache loads the whole sheet at startup. Cut a smaller area, "
            "or re-cut at a coarser resolution.",
        "resolution close to the sheet flown so far":
            "Not fatal, but every tuned number in the matcher was measured at "
            "0.5092 m/px. Re-cut at that resolution unless you mean to change "
            "it.",
        "the take-off point is on the sheet with 150 m to spare":
            "OFF-SHEET means no map fixes for the whole flight, and the only "
            "sign in the air is that no fix ever lands. Re-cut centred on the "
            "take-off point.",
    }
    for c in rep["checks"]:
        checks.append(dict(ok=c["ok"], name=c["name"], detail=c["detail"],
                           fix="" if c["ok"] else FIXES.get(c["name"], ""),
                           hard=c["hard"]))

    grey_mb = rep["width"] * rep["height"] / 1e6
    if grey_mb > RAM_LIMIT_MB:
        ck(False, "small enough to load here without swapping",
           f"{grey_mb:.0f} MB as grey, over the {RAM_LIMIT_MB:.0f} MB this "
           f"check will attempt",
           "Cut a smaller area or a coarser resolution. The node would load "
           "this at startup on a machine that is also running the camera "
           "stack.")
        return _done(checks, t_start, rep)

    # ---- the real thing: the node's own class, on this file
    if VPS_VIO not in sys.path:
        sys.path.insert(0, VPS_VIO)
    try:
        t0 = time.time()
        from map_cache import MapCache
        cache = MapCache(path)
        load_s = time.time() - t0
    except Exception as e:                               # noqa: BLE001
        ck(False, "the node's own MapCache loads it",
           f"{type(e).__name__}: {e}",
           "This is the class live_node.py constructs at startup, so the "
           "flight would fail here too. Re-cut the sheet from the Map tab.")
        return _done(checks, t_start, rep)
    ck(True, "the node's own MapCache loads it",
       f"{cache.summary()} in {load_s:.1f} s")

    # ---- and a window out of it, which is what the matcher actually asks for
    if takeoff:
        lon, lat = float(takeoff[0]), float(takeoff[1])
    else:
        lon = (rep["bounds"][0] + rep["bounds"][2]) / 2
        lat = (rep["bounds"][1] + rep["bounds"][3]) / 2
    inside = cache.covers(lon, lat)
    ck(inside, "a matcher window can be taken at that position",
       f"{lat:.6f}, {lon:.6f}"
       + ("" if inside else " is outside the sheet"),
       "" if inside else
       "The matcher takes a window centred on where it believes the aircraft "
       "is. Outside the sheet there is nothing to match and no fix can land. "
       "Re-cut centred on the take-off point.")
    if inside:
        try:
            win, geo = cache.window(lon, lat, 150.0)
            ok = win.size > 0
            ck(ok, "that window has pixels in it",
               f"{win.shape[1]} x {win.shape[0]} px for a 150 m radius",
               "" if ok else "Re-cut with more margin around the take-off point.")
            if ok:
                sd = float(np.std(win))
                ck(sd > 3.0, "the imagery there has contrast to match against",
                   f"standard deviation {sd:.1f} grey levels"
                   + ("" if sd > 3.0 else " - this window is nearly blank"),
                   "A blank window is water, cloud, a nodata margin or a "
                   "padded edge. The matcher cannot work on it. Re-cut over "
                   "ground with texture.", hard=False)
        except Exception as e:                           # noqa: BLE001
            ck(False, "that window can be sliced", f"{type(e).__name__}: {e}",
               "Re-cut the sheet from the Map tab.")
    del cache
    return _done(checks, t_start, rep)


def _done(checks, t_start, rep):
    hard_fails = [c for c in checks if not c["ok"] and c["hard"]]
    soft = [c for c in checks if not c["ok"] and not c["hard"]]
    return dict(
        ok=not hard_fails,
        checks=checks,
        took_s=round(time.time() - t_start, 1),
        sheet=rep,
        verdict=("This sheet will load and the matcher can work on it."
                 if not hard_fails and not soft else
                 "It loads, but read the notes below before flying it."
                 if not hard_fails else
                 "It will NOT work as it stands - " + hard_fails[0]["name"]),
    )
