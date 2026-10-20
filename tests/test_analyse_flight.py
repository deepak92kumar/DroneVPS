"""Offline map matching must read what a FLIGHT now records, and say what it cannot.

analyse_flight.py has only ever been able to read record_flight.py's folder -
index.csv plus mono/ - and record_flight has never run on a flight. Both it
and live_node called dai.Device(), the OAK-D permits one owner, and navigation
has to win, so every flight came back with an estimate CSV and no images. On
20 Oct live_node learned to save frames itself (--save-frames, frame_writer),
which produces a DIFFERENT shape: vps_NNNNN.csv with a frame_file column, the
images beside it, one folder per run. This pins that the tool reads that shape
and gets the geometry right.

The three failures it exists for:

  1. SCORING AGAINST THE WRONG POSITION. live_node's CSV carries TWO positions
     per row. lat/lon is the VPS estimate - which on an injecting flight is
     partly what the aircraft flew on, so scoring a match against it would be
     marking the node's homework with its own answers. The receiver is in
     gps_north_m/gps_east_m, metres from an origin the CSV never writes down.
     The origin has to be recovered (lat - north/M) or there is no truth here
     at all, and getting that wrong moves every error by the whole offset
     between the two.

  2. A NAME WITH NO IMAGE. The writer drops a frame rather than stall the flow
     loop, and can lose a queued one at shutdown. A row can therefore name a
     file that is not there, and the run must skip it and count it rather than
     die a third of the way through a flight.

  3. REPORTING THE WRONG SHEET AS A BAD MATCHER. bhh.tif spans 30.884-30.903 N
     and the FC's last take-off point (SCR_USER1/2 in log 00000195.BIN) is
     30.7844 - 11.1 km south of it. On the wrong sheet nothing matches, and
     "0% matched" read as a matcher failure would send someone tuning XFeat
     thresholds against a map that does not contain the ground.
"""
import csv
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "vps_vio"))

import numpy as np  # noqa: E402
import cv2  # noqa: E402
from live_node import CSV_COLUMNS, csv_row  # noqa: E402
from analyse_flight import read_flight, focal_from_camera_txt, sheet_covers  # noqa: E402

MAP = "/data/DRONE/bhh.tif"
M_PER_DEG_LAT = 111_320.0
FOCAL = 791.72
ALT = 110.0

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


class Vel:
    vn = ve = speed_mps = sent_vn = sent_ve = 0.0
    deadband_mps = 0.35
    age_s = 0.0
    bias_n = bias_e = 0.0
    flow_updates = 7
    valid = True
    deadbanded = False
    bias_frozen = True
    reason = "ok"


def est_row(t, north, east, origin, alt=ALT, yaw=12.0, frame=""):
    olon, olat = origin
    mlon = M_PER_DEG_LAT * np.cos(np.radians(olat))
    return dict(t=t, north=north, east=east,
                lon=olon + east / mlon, lat=olat + north / M_PER_DEG_LAT,
                sigma=4.5, tracks=250, fix_used=False, alt=alt, yaw=yaw,
                fc_d_roll=0.01, fc_d_pitch=-0.02, gyro_d_roll=0.011,
                gyro_d_pitch=-0.019, gyro_n=500, accel_n=480,
                depth_med_m=12.0, depth_dropped=0, n_dr=0, src="VPS",
                frame_file=frame)


def write_live_folder(root, frames, origin, gps_offset=(3.0, -4.0)):
    """A live_node --save-frames run folder, written through csv_row itself.

    Through csv_row on purpose: a hand-written header would pass this test
    while the real column order moved underneath it, which is the exact way
    the 20 Oct 39-columns-under-a-32-column-header bug got onto the aircraft.
    """
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "camera.txt"), "w") as f:
        f.write(f"fx={FOCAL:.4f} fy=791.5829 cx=494.8892 cy=391.8030 crop_left=130\n")
        f.write("D=24.087360 -39.140041 0.000364\n")
    with open(os.path.join(root, "vps_00004.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_COLUMNS)
        for i in range(1, 25):
            name = frames.get(i, "")
            e = est_row(i / 17.0, 10.0 * i, -2.0 * i, origin, frame=name)
            # The receiver, offset from the estimate by a known amount, on
            # every row but one: that one has no fix and must not be scored.
            gN = None if i == 5 else 10.0 * i + gps_offset[0]
            gE = None if i == 5 else -2.0 * i + gps_offset[1]
            err = None if gN is None else float(np.hypot(10.0 * i - gN, -2.0 * i - gE))
            w.writerow(csv_row(e, dict(rngfnd=0.26, alt_amsl=300.7), Vel(),
                               gN, gE, err))
    return root


tmp = tempfile.mkdtemp(prefix="analyse_flight_")
try:
    ORIGIN = (74.615, 30.894)          # inside bhh.tif
    print("the live_node run folder reads, and the two positions stay apart")
    root = write_live_folder(os.path.join(tmp, "vps_00004"),
                             {3: "f000003.jpg", 7: "f000007.jpg", 5: "f000005.jpg"},
                             ORIGIN)
    shape, frames, notes = read_flight(root)
    check("recognised as a live_node folder", shape == "live_node", shape)
    check("only the rows that name a frame are frames", len(frames) == 3,
          f"{len(frames)}")
    check("fx comes off camera.txt", abs(focal_from_camera_txt(root) - FOCAL) < 1e-3)

    f3 = next(f for f in frames if f["label"] == "f000003.jpg")
    # The origin was never written to the CSV; it has to come back out of a
    # row. If it does, the receiver's metres land where we put them.
    mlon = M_PER_DEG_LAT * np.cos(np.radians(ORIGIN[1]))
    want = (ORIGIN[0] + (-2.0 * 3 - 4.0) / mlon,
            ORIGIN[1] + (10.0 * 3 + 3.0) / M_PER_DEG_LAT)
    check("the origin was recovered and the receiver placed within 1 cm",
          abs(f3["truth"][0] - want[0]) * mlon < 0.01
          and abs(f3["truth"][1] - want[1]) * M_PER_DEG_LAT < 0.01,
          f"{f3['truth']} vs {want}")
    # And the estimate is kept separately - 5 m away here, by construction.
    sep = np.hypot((f3["truth"][0] - f3["est"][0]) * mlon,
                   (f3["truth"][1] - f3["est"][1]) * M_PER_DEG_LAT)
    check("the VPS estimate is not mistaken for the truth", abs(sep - 5.0) < 0.02,
          f"{sep:.3f} m apart")
    f5 = next(f for f in frames if f["label"] == "f000005.jpg")
    check("a row with no receiver fix carries no truth", f5["truth"] is None)
    check("and that is said out loud", any("no receiver fix" in n for n in notes),
          str(notes))

    print("\nthe record_flight folder still reads, unchanged")
    rf = os.path.join(tmp, "flight_20251020")
    os.makedirs(os.path.join(rf, "mono"))
    with open(os.path.join(rf, "camera.txt"), "w") as f:
        f.write(f"fx={FOCAL} fy=791.6 cx=624.9 cy=391.8\n")
    with open(os.path.join(rf, "index.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "wall_s", "mono_file", "lat", "lon", "alt_rel_m", "yaw_deg"])
        w.writerow([0, "1.0", "m000000.png", "30.8965", "74.6132", "105.0", "11.0"])
        w.writerow([1, "1.5", "m000001.png", "0.0", "0.0", "106.0", "12.0"])
    shape2, frames2, _ = read_flight(rf)
    check("recognised as a record_flight folder", shape2 == "record_flight", shape2)
    check("both frames found", len(frames2) == 2, str(len(frames2)))
    check("its images are looked for in mono/",
          frames2[0]["path"].endswith(os.path.join("mono", "m000000.png")))
    check("a 0.0 latitude is no fix, not the equator",
          frames2[1]["truth"] is None and frames2[0]["truth"] is not None)

    print("\na folder that is neither shape is refused with both names")
    try:
        read_flight(tmp)
        check("raises", False)
    except RuntimeError as e:
        check("raises, naming index.csv and vps_NNNNN.csv",
              "index.csv" in str(e) and "vps_" in str(e), str(e))

    print("\nthe wrong sheet is a sheet problem, not a matcher problem")

    class FakeGeo:
        origin_lon, origin_lat = 74.59762, 30.90332
        dlon_dcol, dlat_drow = 4.57e-5, -4.57e-6

        def to_lonlat(self, col, row):
            return (self.origin_lon + col * self.dlon_dcol,
                    self.origin_lat + row * self.dlat_drow)

    class FakeDb:
        image = np.zeros((4200, 757), np.uint8)
        geo = FakeGeo()

    db = FakeDb()
    check("a point on the sheet is covered", sheet_covers(db, 74.615, 30.894))
    # The real case: the FC's last take-off point, 11.1 km south of bhh.tif.
    check("SCR_USER1/2's 30.7844 is NOT", not sheet_covers(db, 74.6072, 30.7844))

    if not os.path.exists(MAP):
        print(f"\nSKIP the end-to-end match: {MAP} not present (gitignored, 91 MB)")
    else:
        print(f"\nend to end on {os.path.basename(MAP)}: a frame cut from the "
              f"map at a known point must come back to it")
        from map_db import MapDatabase
        db = MapDatabase("AKAZE").load_geotiff(MAP)
        H, W = db.image.shape[:2]
        # A window of the map, upscaled to the GSD a camera has at 110 m -
        # 12.6 cm/px against the map's 50.9 - which is the real ratio the
        # matcher works at, not a 1:1 crop that would flatter it.
        gsd_cam = ALT / FOCAL
        fw, fh = 1150, 800                       # the cropped OAK-D frame
        cw, ch = int(fw * gsd_cam / db.gsd), int(fh * gsd_cam / db.gsd)
        c0, r0 = W // 2 - cw // 2, H // 2 - ch // 2
        crop = db.image[r0:r0 + ch, c0:c0 + cw]
        frame = cv2.resize(crop, (fw, fh), interpolation=cv2.INTER_CUBIC)
        lon_t, lat_t = db.geo.to_lonlat(c0 + cw / 2.0, r0 + ch / 2.0)
        lon_t, lat_t = float(lon_t), float(lat_t)

        run = os.path.join(tmp, "vps_00009")
        # The CSV puts the VPS estimate ON the frame's true centre and the
        # receiver 8 m north of it. That pair is the test: an error near 8 m
        # with an offset-from-VPS near 0 proves the match found the ground AND
        # was scored against the RECEIVER. Had it been scored against the
        # estimate the two numbers would be the other way round, and on an
        # injecting flight that mistake marks the node with its own answers.
        olat = lat_t - (10.0 * 3) / M_PER_DEG_LAT
        mlon9 = M_PER_DEG_LAT * np.cos(np.radians(olat))
        write_live_folder(run, {3: "f000003.jpg", 7: "f000007.jpg"},
                          (lon_t - (-2.0 * 3) / mlon9, olat),
                          gps_offset=(8.0, 0.0))
        cv2.imwrite(os.path.join(run, "f000003.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        # f000007.jpg is named by a row and deliberately NOT written: the
        # writer drops frames, so this is the normal case, not corruption.
        out = os.path.join(tmp, "res.csv")
        p = subprocess.run([sys.executable,
                            os.path.join(HERE, "vps_vio", "analyse_flight.py"),
                            run, "--map", MAP, "--every", "1", "--min-alt", "0",
                            "--out", out], capture_output=True, text=True,
                           timeout=900)
        tail = "\n".join(p.stdout.strip().splitlines()[-6:])
        check("the run finished", p.returncode == 0, p.stderr.strip()[-300:])
        check("the missing frame was skipped and counted",
              "1 named frames were not on disk" in p.stdout,
              tail)
        res = list(csv.DictReader(open(out))) if os.path.exists(out) else []
        check("one match was written to the results CSV", len(res) == 1,
              f"{len(res)} rows\n{tail}")
        if res:
            r = res[0]
            err_to_truth = float(r["err_vs_gps_m"])
            d = np.hypot((float(r["match_lon"]) - lon_t) * mlon9,
                         (float(r["match_lat"]) - lat_t) * M_PER_DEG_LAT)
            check("the match landed within 15 m of where the frame was cut",
                  d < 15.0, f"{d:.1f} m")
            check("and it was scored against the receiver, 8 m away",
                  abs(err_to_truth - 8.0) < 3.0, f"err_vs_gps_m {err_to_truth:.1f}")
            check("not against the VPS estimate, which sat on the frame",
                  r["offset_vs_vps_m"] not in ("", None)
                  and float(r["offset_vs_vps_m"]) < 3.0, r["offset_vs_vps_m"])
            check("the row keeps the frame's name", r["frame_file"] == "f000003.jpg")
            check("and its heading, for the prior", abs(float(r["yaw_deg"]) - 12.0) < 1e-6)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
