"""Offline map matching has to read what a flight records, and score it honestly.

match_flight.py is the consumer that was missing. Until 20 Oct no flight could
produce its input at all: live_node and record_flight both call dai.Device(),
the OAK-D permits one owner, navigation has to win, so a flight came back with
an estimate CSV and no images. live_node --save-frames now writes the frames
from inside the navigating process into a folder beside the CSV, with each
image's name in its row's frame_file column. This pins what reading that folder
back has to get right.

The failures it exists for, each one real:

  1. THE JOIN, AGAINST THE REAL COLUMN ORDER. The whole value of frame_file is
     that an image arrives with the heading and height it was taken at. Those
     live in columns 9 and 10 of forty, and on 20 Oct seven columns were added
     to live_node's row writer and not to its header - 1162 rows were written
     39 wide under a 32-wide header, a file that opens fine and is wrong in
     every column after the 32nd. So the CSV here is written through
     live_node.csv_row, not by hand: a test with its own header would keep
     passing through exactly that mistake.

  2. SCORING AGAINST THE WRONG POSITION. The CSV carries TWO positions a row.
     lat/lon is the VPS ESTIMATE, which on an injecting flight is partly what
     the aircraft then flew on - scoring a match against it marks the node's
     homework with its own answers. The truth is the receiver, in
     gps_north_m/gps_east_m, metres from an origin the CSV never writes down.
     Here the frames are cut from the map at a KNOWN position and the estimate
     is offset 7.8 m from it, so a tool that scored against the estimate would
     report 7.8 m where the right answer is a third of a metre, and a tool that
     muddled the north/east sign would report 15.6.

  3. A NAME WITH NO IMAGE. FrameWriter drops a frame rather than stall the flow
     loop and can lose a queued one at shutdown; it counts both and says so.
     A row naming a file that is not there is therefore a KNOWN case, and it
     must be skipped and counted, not crash a run that has a hundred good
     frames after it.

  4. THE WRONG SHEET. bhh.tif covers 30.884-30.903 N and the point the FC last
     flew from reads 30.784 - 11 km south. Every frame then fails for one
     reason that has nothing to do with the matcher, and reporting "0 of 240
     matched" would send the next person to sweep thresholds that were never
     consulted. It has to be detected before matching and said plainly.

HOW A KNOWN POSITION IS MANUFACTURED. A frame is cut out of bhh.tif by the
inverse of the geometry north_up_query applies: map pixels to metres about a
centre, to camera pixels at gsd = altitude/focal, rotated by -yaw. The sign is
measured, not assumed - verified 20 Oct at yaw 0, +30, -45 and +120, where -yaw
recovers the centre to 0.22-0.34 m and +yaw gives 0-2 inliers of 127-175.

WHAT THE TOLERANCE HERE IS AND IS NOT. The frame and the map are the SAME
IMAGERY, so this is an upper bound on agreement in every respect and says
nothing about a real camera over real trees against a four-year-old sheet.
Measured 0.24-0.33 m at four points across the sheet with a 7.8 m drifted
prior; the test allows 3.0 m, which is loose enough for OpenCV and XFeat
version drift and tight enough that a broken scale, a flipped yaw or a
misjoined row cannot pass - any of those cost tens of metres or all the
inliers.

Skips if the map is not on this machine - it is gitignored, being 91 MB.
"""
import csv
import math
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "vps_vio"))

MAP = "/data/DRONE/bhh.tif"
if not os.path.exists(MAP):
    print(f"SKIP: {MAP} not present (gitignored, 91 MB)")
    sys.exit(0)

from live_node import CSV_COLUMNS, csv_row                      # noqa: E402
from map_cache import MapCache                                  # noqa: E402
import match_flight                                             # noqa: E402
from match_flight import (MATCH_COLUMNS, match_row, read_camera,  # noqa: E402
                          gps_lonlat, metres_between, run)

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


# The real OAK-D's own numbers, as live_node writes them: fx/fy from the
# calibration at 1280x800, cx already moved by the 130 px left crop where the
# aircraft's landing leg sits (624.89 -> 494.89), and the 14 coefficients of
# the rational model plus thin prism - not the 5 cv2.undistort takes by
# default. Hard-coding fx in the TOOL would be a silent scale error in every
# query, so the tool reads this file and this test writes it.
FX, FY, CX, CY, CROP = 791.72, 791.58, 494.89, 391.80, 130
W, H = 1150, 800                 # the cropped frame, leg removed
D14 = [-4.806814, 13.470192, 0.000253, -0.000486, 15.366786,
       -4.939521, 13.927822, 17.231852, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
M_PER_DEG_LAT = 111_320.0
TOL_M = 3.0                      # see the docstring: 0.24-0.33 measured
DRIFT_N, DRIFT_E = 6.0, 5.0      # how far the estimate sits from the truth


def synth_frame(cache, lon, lat, yaw_deg, alt_m, *, fx=FX, w=W, h=H):
    """The camera frame whose IMAGE CENTRE is the ground point lon/lat.

    The inverse of north_up_query, in one affine: map pixel -> metres about the
    centre -> camera pixel at gsd = alt/fx -> rotated by -yaw. Done as a single
    warpAffine off the whole sheet rather than crop-then-rotate, so the frame
    has no black corners to flatter the matcher with.
    """
    g = alt_m / float(fx)
    c0, r0 = cache.geo.to_pixel(lon, lat)
    th = math.radians(-yaw_deg)
    S = np.array([[cache.gsd_x / g, 0.0], [0.0, cache.gsd / g]])
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    A = R @ S
    t = np.array([w / 2.0, h / 2.0]) - A @ np.array([float(c0), float(r0)])
    return cv2.warpAffine(cache.grey, np.hstack([A, t.reshape(2, 1)]), (w, h),
                          flags=cv2.INTER_AREA)


class Vel:
    """What the velocity bridge would have had on the row. Not under test."""
    vn = ve = speed_mps = sent_vn = sent_ve = 0.0
    deadband_mps = 0.35
    age_s = 0.0
    bias_n = bias_e = 0.0
    flow_updates = 7
    valid = True
    deadbanded = False
    bias_frozen = True
    reason = "ok"


def make_run(folder, cache, specs, *, write_frames=True, bare_camera=False):
    """Build a folder in the shape live_node --save-frames writes.

    specs is a list of dicts: lon/lat are the frame's TRUE centre, yaw and alt
    the prior that goes in its row. "on_disk" False names a frame and does not
    write it (FrameWriter's lost-at-close case). "named" False writes a row
    with an empty frame_file, which is what most rows look like - the gate
    saves a couple a second out of seventeen.
    """
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "camera.txt"), "w") as cf:
        if bare_camera:
            # What a replayed folder gets: ReplaySource knows the focal length
            # and nothing else, so live_node writes this shape instead.
            cf.write(f"fx={FX:.4f}\n")
        else:
            cf.write(f"fx={FX:.4f} fy={FY:.4f} cx={CX:.4f} cy={CY:.4f} "
                     f"crop_left={CROP}\n")
            cf.write("D=" + " ".join(f"{x:.6f}" for x in D14) + "\n")

    stem = os.path.basename(folder.rstrip("/"))
    truths = []
    with open(os.path.join(folder, stem + ".csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(CSV_COLUMNS)
        for i, sp in enumerate(specs):
            lon_t, lat_t = sp["lon"], sp["lat"]
            mlon = M_PER_DEG_LAT * math.cos(math.radians(lat_t))
            # The estimate sits DRIFT away from the truth, as a flight's does.
            lat_p = lat_t + DRIFT_N / M_PER_DEG_LAT
            lon_p = lon_t + DRIFT_E / mlon
            name = ""
            if sp.get("named", True):
                name = f"f{i:06d}.jpg"
                if sp.get("on_disk", True) and write_frames:
                    img = synth_frame(cache, lon_t, lat_t, sp["yaw"], sp["alt"])
                    cv2.imwrite(os.path.join(folder, name), img,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            est = dict(t=i * 0.5, north=0.0, east=0.0, lon=lon_p, lat=lat_p,
                       sigma=4.5, sigma_abs=20.5, tracks=250, fix_used=False, alt=sp["alt"],
                       yaw=sp["yaw"], fc_d_roll=0.01, fc_d_pitch=-0.02,
                       gyro_d_roll=0.011, gyro_d_pitch=-0.019, gyro_n=500,
                       accel_n=480, depth_med_m=12.0, depth_dropped=0, n_dr=1,
                       src="VPS", frame_file=name)
            # The receiver, in metres from the origin - and north/east are 0 on
            # the row, so the origin IS the estimate and the GPS offset is the
            # whole truth-minus-estimate vector. gps_lonlat has to put it back.
            gN, gE = -DRIFT_N, -DRIFT_E
            err = math.hypot(gN, gE)
            wr.writerow(csv_row(est, dict(rngfnd=0.26, alt_amsl=300.7), Vel(),
                                gN, gE, err))
            truths.append((lon_t, lat_t, name))
    return truths


# --------------------------------------------------------------------------- #
print("the results header and a results row are the same width")
# For the reason test_csv_columns.py exists: a row wider than its header is a
# file that opens fine and is wrong in every column after the header ends.
class _Res:
    lon = 74.61
    lat = 30.89
    score = 0.85
    margin = 0.75
    ok = True
    inliers = 400
    total_matches = 470


row = match_row({"t": "1.0", "frame_file": "f000001.jpg", "alt_m": "110.0",
                 "yaw_deg": "3.2"}, 7, _Res(), lon=74.61, lat=30.89,
                gps=(74.61, 30.89), d_vps=1.0, d_gps=0.5, reason="", fix_s=0.27)
check("widths match", len(row) == len(MATCH_COLUMNS),
      f"header {len(MATCH_COLUMNS)}, row {len(row)}")
row = match_row({}, 1, None, lon=None, lat=None, gps=None, d_vps=None,
                d_gps=None, reason="named frame not on disk", fix_s=None)
check("and with no result and no GPS at all", len(row) == len(MATCH_COLUMNS),
      f"header {len(MATCH_COLUMNS)}, row {len(row)}")
check("missing values are empty, not the string None",
      "None" not in [str(x) for x in row],
      str([x for x in row if str(x) == "None"]))

print("\ncamera.txt is read, never assumed")
tmp = tempfile.mkdtemp(prefix="mf_cam_")
try:
    cache = MapCache(MAP)
    make_run(os.path.join(tmp, "vps_00001"), cache, [], write_frames=False)
    cam = read_camera(os.path.join(tmp, "vps_00001"))
    check("fx comes from the file", abs(cam["fx"] - FX) < 1e-4, f"{cam['fx']}")
    check("cx too, crop already applied", abs(cam["cx"] - CX) < 1e-4,
          f"{cam['cx']} (uncropped would be {CX + CROP})")
    check("and the crop it was adjusted by", cam["crop_left"] == CROP,
          str(cam["crop_left"]))
    check("all 14 distortion coefficients, not 5",
          cam["D"] is not None and len(cam["D"]) == 14,
          str(None if cam["D"] is None else len(cam["D"])))
    make_run(os.path.join(tmp, "vps_00002"), cache, [], write_frames=False,
             bare_camera=True)
    cam2 = read_camera(os.path.join(tmp, "vps_00002"))
    check("a replayed folder's bare fx= line still reads",
          abs(cam2["fx"] - FX) < 1e-4 and cam2["D"] is None, str(cam2))
    check("and fy falls back to fx rather than to zero",
          abs(cam2["fy"] - FX) < 1e-4, str(cam2.get("fy")))
    try:
        read_camera(tmp)
        check("a folder with no camera.txt is refused", False, "it was accepted")
    except RuntimeError as e:
        check("a folder with no camera.txt is refused, by name",
              "camera.txt" in str(e))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\nthe GPS truth is recovered from the metres-from-origin columns")
r = {"lon": "74.6150000", "lat": "30.8940000", "north_m": "0.00",
     "east_m": "0.00", "gps_north_m": "-6.00", "gps_east_m": "-5.00"}
g = gps_lonlat(r)
d = metres_between(74.615, 30.894, g[0], g[1])
check("the GPS sits the recorded offset from the estimate",
      abs(d - math.hypot(6.0, 5.0)) < 0.01, f"{d:.3f} m")
check("and to the south-west of it, not the north-east",
      g[1] < 30.894 and g[0] < 74.615, f"{g}")
check("a row with no GPS fix returns nothing, not a zero",
      gps_lonlat({**r, "gps_north_m": "", "gps_east_m": ""}) is None)

# --------------------------------------------------------------------------- #
print("\nmatching a folder of frames cut from the map at known positions")
# One folder holding all four cases at once, because that is how they arrive:
# good frames, a frame whose image was lost at shutdown, and the rows - most of
# them - that name no frame at all.
SPECS = [
    dict(lon=74.6100, lat=30.8930, yaw=0.0, alt=110.0),
    dict(lon=74.6180, lat=30.8960, yaw=37.0, alt=110.0),
    dict(lon=74.6050, lat=30.8900, yaw=-112.0, alt=95.0),
    dict(lon=74.6150, lat=30.8980, yaw=205.0, alt=130.0),
    dict(lon=74.6120, lat=30.8950, yaw=15.0, alt=110.0, on_disk=False),
    dict(lon=74.6120, lat=30.8950, yaw=15.0, alt=110.0, named=False),
    dict(lon=74.6120, lat=30.8950, yaw=15.0, alt=0.3),     # on the ground
]
GOOD = 4

tmp = tempfile.mkdtemp(prefix="mf_run_")
try:
    folder = os.path.join(tmp, "vps_00007")
    cache = MapCache(MAP)
    truths = make_run(folder, cache, SPECS)
    del cache
    s = run(folder, MAP, quiet=True)

    check("every row naming a frame was picked up, and only those",
          s["named"] == len([x for x in SPECS if x.get("named", True)]),
          f"{s['named']} of {len(SPECS)} rows")
    check("the row that names no frame is not in the results",
          s["named"] == len(SPECS) - 1, str(s["named"]))
    check("the frame that is not on disk was counted, not fatal",
          s["missing_file"] == 1, str(s["missing_file"]))
    check("the frame on the ground was held below the altitude floor",
          s["below_floor"] == 1, str(s["below_floor"]))
    check("so the map was asked about the good frames only",
          s["attempted"] == GOOD, f"{s['attempted']}, expected {GOOD}")
    check("and every one of them passed the gate the aircraft flies with",
          s["matched"] == GOOD, f"{s['matched']} of {GOOD}, "
          f"rejected {s['rejected']}: {s['reasons']}")

    e = s["errs_gps"]
    check(f"each match landed within {TOL_M:.0f} m of the known position",
          len(e) == GOOD and max(e) <= TOL_M,
          f"n={len(e)} median {np.median(e):.2f} max {max(e):.2f} m"
          if e else "no errors recorded")
    check("the error is against the GPS, not against the estimate",
          bool(e) and np.median(e) < math.hypot(DRIFT_N, DRIFT_E) / 2,
          f"median {np.median(e):.2f} m against a {math.hypot(DRIFT_N, DRIFT_E):.1f} m drift"
          if e else "")
    v = s["errs_vps"]
    check("and the offset from the estimate is that drift, so the join is right",
          bool(v) and abs(np.median(v) - math.hypot(DRIFT_N, DRIFT_E)) < TOL_M,
          f"median {np.median(v):.2f} m, drift {math.hypot(DRIFT_N, DRIFT_E):.2f} m"
          if v else "")

    print("\n  the results CSV")
    out = os.path.join(folder, "match_results.csv")
    check("was written into the run's own folder", os.path.exists(out))
    lines = list(csv.reader(open(out)))
    check("header first, then one row per named frame",
          len(lines) == 1 + s["named"], f"{len(lines)} lines")
    check("every line the same width", len(set(len(x) for x in lines)) == 1,
          str(set(len(x) for x in lines)))
    res = list(csv.DictReader(open(out)))
    by_name = {r["frame_file"]: r for r in res}
    check("the missing frame has a row that says why, and no match",
          "not on disk" in by_name["f000004.jpg"]["reason"]
          and by_name["f000004.jpg"]["match_lon"] == "",
          by_name["f000004.jpg"]["reason"])
    check("the grounded frame's row names its altitude and the floor",
          "0.3" in by_name["f000006.jpg"]["reason"]
          and "floor" in by_name["f000006.jpg"]["reason"],
          by_name["f000006.jpg"]["reason"])
    ok_rows = [r for r in res if r["ok"] == "1"]
    check("a matched row carries its inlier count and its error",
          len(ok_rows) == GOOD
          and all(int(r["inliers"]) >= 12 and r["d_gps_m"] for r in ok_rows),
          str([(r["inliers"], r["d_gps_m"]) for r in ok_rows]))
    check("and the row index points back into the estimate CSV",
          all(1 <= int(r["row"]) <= len(SPECS) for r in res),
          str([r["row"] for r in res]))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# --------------------------------------------------------------------------- #
print("\na take-off point off the sheet is said plainly, not reported as zero")
tmp = tempfile.mkdtemp(prefix="mf_off_")
try:
    folder = os.path.join(tmp, "vps_00009")
    cache = MapCache(MAP)
    # 30.7844 N, 74.6072 E: what SCR_USER1/2 reads in log 00000195.BIN, which
    # is 11 km south of this sheet. The normal case on this machine.
    off = [dict(lon=74.6072, lat=30.7844 + i * 1e-4, yaw=0.0, alt=110.0)
           for i in range(3)]
    make_run(folder, cache, off, write_frames=False)
    del cache
    s = run(folder, MAP, quiet=True)
    check("it is recognised as the wrong sheet", s["off_sheet_all"] is True)
    check("no frame was matched, and none was attempted",
          s["attempted"] == 0 and s["matched"] == 0,
          f"attempted {s['attempted']}")
    check("every named row is accounted for as off the sheet",
          s["off_sheet"] == len(off), str(s["off_sheet"]))
    check("the exit code distinguishes it from a run that found nothing",
          match_flight.main([folder, "--map", MAP]) == 2)
    # And it must not be confused with the bench case, which is a different
    # sentence: 0 matches because nothing was high enough, not because the
    # flight is somewhere else.
    check("and it is not reported as an altitude problem",
          s["below_floor"] == 0, str(s["below_floor"]))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
