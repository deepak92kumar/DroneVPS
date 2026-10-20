"""A map fix must report the point under the CAMERA, not under the middle of the image.

Found 20 Oct and measured on 30 frames cut from bhh.tif: applying this
correction moves the answer 10.91 m median, 15.22 m max, scaling with altitude
(~7 m at 70 m, ~15 m at 150 m). The cause is arithmetic, not a matcher:
map_query.north_up_query rotates about (w/2, h/2) and never read cx, so the
translation it hands back is about the image's geometric centre, while
live_node feeds the result into MapFuser as the AIRCRAFT's position. Those are
different ground points. The aircraft is under the pixel the optical axis
pierces - the principal point - and these frames are the 130 px leg-crop, so
cx 494.89 sits 80.1 px from the geometric centre of 575. At 110 m the camera's
GSD is 0.139 m/px: 11.1 m.

Why it was never visible: the offset is in the same direction on every frame,
so every frame still matches, with the same inlier count, at the same score.
It moves the whole flight sideways. flow_odom has used K's cx/cy as its centre
since 28 Sep, so for as long as both paths have run they have disagreed about
where the aircraft was, by 11 m, and nothing in either one could say so.

What this pins:

  1. THE SIGN. A correction applied backwards doubles the error to 22 m and
     the frame still matches just as well. There is no feedback that catches
     it, so the direction is pinned here against a frame whose true position
     is known by construction.
  2. THE MAGNITUDE, and that it scales with altitude - it is a pixel offset
     on the ground, so metres per pixel is the altitude.
  3. THE ORDER. north_up_query rescales, rotates north-up, then stretches x
     for the map's east-west anisotropy (16.5% at this latitude - the "16 m
     that has never been explained" of 12 Oct). The tracked point must go
     through the same three stages in the same order as the pixels; through
     the rotation in the wrong order it comes out wrong by this error's own
     size again.
  4. YAW. The offset is fixed in the camera's frame, so in the map's frame it
     rotates with the aircraft. At yaw 180 it points the other way.
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "vps_vio"))

import numpy as np  # noqa: E402
from map_query import north_up_query  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


# This camera, measured 20 Oct, after the 130 px leg crop.
W, H = 1150, 800
FOCAL = 791.72
CX, CY = 494.8892, 391.8030
MAP_GSD = 0.5092                 # bhh.tif north-south
MAP_GSD_X = 0.4369               # and east-west: the pixels are not square
ALT = 110.0

img = np.zeros((H, W), np.uint8)


def tracked(alt=ALT, yaw=0.0, aniso=True, xy=(CX, CY)):
    q, pt = north_up_query(img, altitude_m=alt, focal_px=FOCAL, yaw_deg=yaw,
                           map_gsd_m=MAP_GSD,
                           map_gsd_x_m=MAP_GSD_X if aniso else None,
                           track_xy=xy)
    centre = np.array([q.shape[1] / 2.0, q.shape[0] / 2.0])
    return q, np.array(pt), centre


print("the principal point is not the middle of the image, and the gap is 11 m")
q, pt, centre = tracked()
d_px = pt - centre
# Metres per pixel in the query is NOT the same on both axes, and getting
# that backwards is how this test first read 13.3 m instead of 11.4. The
# anisotropy stretch puts the query's x axis on the MAP's east-west pixel
# pitch - 0.4369 m - while y stays at the north-south 0.5092. The stretch
# exists precisely so x matches the map, not so both axes match each other.
def metres(d):
    return float(np.hypot(d[0] * MAP_GSD_X, d[1] * MAP_GSD))


d_m = metres(d_px)
check("the tracked point is off the query centre", np.hypot(*d_px) > 1.0,
      f"{d_px[0]:+.2f}, {d_px[1]:+.2f} px")
check("by 11 m of ground at 110 m", 10.0 < d_m < 12.5, f"{d_m:.2f} m")
# 80.1 px of camera frame at 0.1389 m/px = 11.12 m, independently of the query.
want = np.hypot((CX - W / 2.0) * ALT / FOCAL, (CY - H / 2.0) * ALT / FOCAL)
check("which is the camera-frame offset times the camera's GSD",
      abs(d_m - want) < 0.5, f"query {d_m:.2f} m vs camera {want:.2f} m")

print("\nthe SIGN: cx is LEFT of centre, so the point is left of centre")
check("x is negative, not positive", d_px[0] < 0, f"{d_px[0]:+.2f} px")
check("and y too, by the smaller cy offset", d_px[1] < 0, f"{d_px[1]:+.2f} px")
# Applied backwards this doubles the error, and every frame still matches.
mirrored = np.array([W - 1 - CX, H - 1 - CY])
_, pt_m, c_m = tracked(xy=mirrored)
check("a mirrored principal point lands the other side, equally far",
      (pt_m - c_m)[0] > 0
      and abs(np.hypot(*(pt_m - c_m)) - np.hypot(*d_px)) < 1.5,
      f"{(pt_m - c_m)[0]:+.2f} px")

print("\nit scales with altitude, because it is a pixel offset on the ground")
for alt, lo, hi in ((70.0, 6.5, 8.0), (110.0, 10.5, 12.5), (150.0, 14.5, 16.5)):
    _, p2, c2 = tracked(alt=alt)
    m = metres(p2 - c2)
    check(f"at {alt:.0f} m it is {m:.1f} m", lo <= m <= hi, f"{m:.2f} m")

print("\nin the MAP's frame it rotates with the aircraft")
_, p0, c0 = tracked(yaw=0.0)
_, p180, c180 = tracked(yaw=180.0)
v0, v180 = p0 - c0, p180 - c180
check("at yaw 180 it points the opposite way",
      np.dot(v0, v180) < 0, f"{v0} vs {v180}")
check("and is the same distance out",
      abs(np.hypot(*v0) - np.hypot(*v180)) < 1.5,
      f"{np.hypot(*v0):.2f} vs {np.hypot(*v180):.2f} px")
_, p90, c90 = tracked(yaw=90.0)
v90 = p90 - c90
check("at yaw 90 it is perpendicular", abs(np.dot(v0, v90)) < 0.06 * np.dot(v0, v0),
      f"cos {np.dot(v0, v90)/max(np.dot(v0,v0),1e-9):+.3f}")

print("\nthe ORDER through the transform is the same as the pixels take")
# The anisotropy stretch is applied after the rotation. If the tracked point
# took it before, at yaw 90 the x and y offsets would be stretched the wrong
# way round - so compare the ratio between aniso on and off at yaw 90.
_, pa, ca = tracked(yaw=90.0, aniso=True)
_, pb, cb = tracked(yaw=90.0, aniso=False)
kx = MAP_GSD / MAP_GSD_X
check("x is stretched by the map's ratio, y is not",
      abs((pa - ca)[0] - (pb - cb)[0] * kx) < 1.0
      and abs((pa - ca)[1] - (pb - cb)[1]) < 1.0,
      f"aniso {(pa-ca)[0]:+.2f} vs plain {(pb-cb)[0]:+.2f} x {kx:.4f}")
# And the geometric centre must still be the geometric centre: tracking the
# middle of the image has to come back to the middle of the query, or the
# chain has an offset in it that would move every answer.
_, pm, cm = tracked(xy=(W / 2.0, H / 2.0))
check("tracking the image centre returns the query centre",
      np.hypot(*(pm - cm)) < 1.0, f"{np.hypot(*(pm - cm)):.3f} px")

print("\nasking for no tracking still returns just the query, unchanged")
plain = north_up_query(img, altitude_m=ALT, focal_px=FOCAL, yaw_deg=12.0,
                       map_gsd_m=MAP_GSD, map_gsd_x_m=MAP_GSD_X)
check("one array, not a tuple", isinstance(plain, np.ndarray), str(type(plain)))
q2, _ = north_up_query(img, altitude_m=ALT, focal_px=FOCAL, yaw_deg=12.0,
                       map_gsd_m=MAP_GSD, map_gsd_x_m=MAP_GSD_X,
                       track_xy=(CX, CY))
check("and the query is identical either way", plain.shape == q2.shape
      and bool((plain == q2).all()), f"{plain.shape} vs {q2.shape}")

print("\nthe flying path passes K's cx/cy in, so the two paths agree")
import inspect  # noqa: E402
from map_fix_worker import MapFixWorker  # noqa: E402
import live_node  # noqa: E402
check("MapFixWorker takes a principal point",
      "principal_xy" in inspect.signature(MapFixWorker.__init__).parameters)
src = inspect.getsource(live_node.VpsNode.__init__)
check("live_node gives it one from K", "principal_xy" in src)
check("taken from K's cx and cy, not the frame's middle",
      "[0, 2]" in src and "[1, 2]" in src)
w = MapFixWorker.__new__(MapFixWorker)
MapFixWorker.__init__(w, None, focal_px=FOCAL, principal_xy=(CX, CY))
check("and it keeps it", w.principal_xy == (CX, CY), str(w.principal_xy))
w2 = MapFixWorker.__new__(MapFixWorker)
MapFixWorker.__init__(w2, None, focal_px=FOCAL)
check("a caller with no K gets the old behaviour, not a different answer",
      w2.principal_xy is None)

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
