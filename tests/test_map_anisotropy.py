"""Does correcting the map's east-west scale actually remove the error.

The 12 Oct report found a 16.5% mismatch behind XFEAT_ANISO and blamed it for
"16 m that has never been explained". A geographic GeoTIFF has equal pixel size
in DEGREES, so in METRES it is anisotropic by cos(latitude) - 1/cos(30.89) is
1.1653, which is that 16.5%. This measures whether correcting it closes the gap.

Method, on the real map: take a patch at a known lon/lat and build the camera
frame it would have produced. That means inverting the geometry - a camera's
pixels ARE square in metres (gsd = altitude/focal) while the map's are not, so
the patch is stretched by gsd_x/gsd_drone in x and gsd_y/gsd_drone in y. Then
run locate() twice, once told about the east-west GSD and once not, and compare
how far each lands from the truth.

Skips if the map is not on this machine - it is gitignored, being 91 MB.
"""
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "vps_vio"))

MAP = "/home/abc/Desktop/DRONE/bhh.tif"
if not os.path.exists(MAP):
    print(f"SKIP: {MAP} not present (gitignored, 91 MB)")
    sys.exit(0)

from map_cache import MapCache          # noqa: E402
from map_match_ncc import locate as ncc_locate   # noqa: E402

ALT, FOCAL = 110.0, 791.72
GSD_DRONE = ALT / FOCAL
RADIUS_M = 0.60 * ALT                   # the flown window, 4 Oct

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


M = MapCache(MAP)
print(M.summary())
print(f"gsd north-south {M.gsd:.4f}  east-west {M.gsd_x:.4f}  "
      f"ratio {M.gsd/M.gsd_x:.4f} = {100*(M.gsd/M.gsd_x-1):.2f}%")
print(f"camera gsd at {ALT:.0f} m: {GSD_DRONE:.4f} m/px\n")


FRAME_W, FRAME_H = 1280, 800            # the mono pair, measured on this OAK-D
GROUND_W = FRAME_W * GSD_DRONE          # 177.8 m across at 110 m
GROUND_H = FRAME_H * GSD_DRONE          # 111.1 m down


def synth_frame(lon, lat, yaw_deg=0.0):
    """The camera frame that this ground would have produced.

    It has to be the real frame shape, not a square: north_up_query keeps
    0.95 of the SMALLER side, so a square query would produce a template far
    larger than the search window and simply refuse to match. 1280x800 at
    0.1389 m/px is 177.8 x 111.1 m of ground.

    The camera's pixels are square in metres; the map's are not. So the ground
    patch is cut using each axis's own GSD and then resampled to the frame
    shape - that is the forward model the matcher has to invert.
    """
    c, r = M.geo.to_pixel(lon, lat)
    c, r = int(c), int(r)
    hw = int(round(GROUND_W / M.gsd_x / 2))     # east-west, in map px
    hh = int(round(GROUND_H / M.gsd / 2))       # north-south, in map px
    patch = M.grey[r - hh:r + hh, c - hw:c + hw]
    f = cv2.resize(patch, (FRAME_W, FRAME_H), interpolation=cv2.INTER_CUBIC)
    if yaw_deg:
        rot = cv2.getRotationMatrix2D((FRAME_W / 2, FRAME_H / 2), yaw_deg, 1.0)
        f = cv2.warpAffine(f, rot, (FRAME_W, FRAME_H))
    return f


def err_m(lon_got, lat_got, lon_true, lat_true):
    mlon = 111_320.0 * math.cos(math.radians(lat_true))
    return (abs(lon_got - lon_true) * mlon,          # east-west component
            abs(lat_got - lat_true) * 111_320.0,     # north-south component
            math.hypot((lon_got - lon_true) * mlon,
                       (lat_got - lat_true) * 111_320.0))


# Several points across the sheet, away from the edges.
pts = []
for fy in (0.35, 0.5, 0.65):
    for fx in (0.35, 0.5, 0.65):
        lon, lat = M.geo.to_lonlat(M.w * fx, M.h * fy)
        pts.append((float(lon), float(lat)))

print(f"{len(pts)} points, each matched with and without the correction\n")
rows = []
for lon, lat in pts:
    frame = synth_frame(lon, lat)
    win, wg = M.window(lon, lat, RADIUS_M, kind="edges")
    common = dict(altitude_m=ALT, focal_px=FOCAL, yaw_deg=0.0,
                  map_gsd_m=M.gsd, window_is_edges=True)
    r_off = ncc_locate(frame, win, wg, **common)
    r_on = ncc_locate(frame, win, wg, map_gsd_x_m=M.gsd_x, **common)
    if not (r_off.ok and r_on.ok):
        print(f"  (skipped a point: off.ok={r_off.ok} on.ok={r_on.ok})")
        continue
    e_off = err_m(r_off.lon, r_off.lat, lon, lat)
    e_on = err_m(r_on.lon, r_on.lat, lon, lat)
    rows.append((e_off, e_on))
    print(f"  off: E {e_off[0]:6.2f}  N {e_off[1]:5.2f}  tot {e_off[2]:6.2f} m"
          f"   |   on: E {e_on[0]:5.2f}  N {e_on[1]:5.2f}  tot {e_on[2]:5.2f} m")

check("enough points matched", len(rows) >= 5, f"{len(rows)} of {len(pts)}")
if rows:
    off_tot = float(np.median([r[0][2] for r in rows]))
    on_tot = float(np.median([r[1][2] for r in rows]))
    off_e = float(np.median([r[0][0] for r in rows]))
    on_e = float(np.median([r[1][0] for r in rows]))
    print(f"\n  median total error   off {off_tot:6.2f} m   on {on_tot:5.2f} m")
    print(f"  median east-west     off {off_e:6.2f} m   on {on_e:5.2f} m")
    check("the correction reduces total error", on_tot < off_tot,
          f"{off_tot:.2f} -> {on_tot:.2f} m")
    check("and it is the EAST-WEST component that moves", on_e < off_e,
          f"{off_e:.2f} -> {on_e:.2f} m")
    check("corrected error is small", on_tot < 5.0, f"{on_tot:.2f} m")

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
