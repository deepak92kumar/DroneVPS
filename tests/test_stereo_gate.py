"""The stereo pair, and the one thing it is good for here.

Both mono cameras already ran and StereoDepth already computed a range map on
the camera's own VPU - it reached live_node's frame dict and was then dropped on
the floor. Nothing consumed it.

What it is NOT for: height. That was tried on 9 Oct and the report's own verdict
was "worse than the baro it was meant to help". Nothing here divides by depth.

What it IS for: keeping the flow on one surface. shift_to_ne turns pixels into
metres by dividing by ONE altitude, so it assumes every tracked feature lies on
one plane. A feature on a roof is a perfectly consistent track - RANSAC is
happy with it - and simply scaled wrong. That is the parallax this module's own
notes blame, and the pair can see it where a single altitude cannot.

The envelope is the baseline's, 7.5 cm with fx 791.72: 0.2 px of disparity
noise is 3.4% of range at 10 m, 8.4% at 25 m, and 20.2% at 60 m where the
disparity has fallen under one pixel. So the gate runs below 30 m, and the map
matcher needs 60 m - they cover different bands, which is the point.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "vps_vio"))
from flow_odom import FlowTracker  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def pts(n, w=1280, h=800, seed=3):
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(10, w - 10, n),
                     rng.uniform(10, h - 10, n)], axis=1).reshape(-1, 1, 2).astype(np.float32)


def flat(depth_m, w=1280, h=800):
    return np.full((h, w), int(depth_m * 1000), np.uint16)


tr = FlowTracker(max_corners=400)
P = pts(200)

print("a flat surface keeps everything")
g = tr._depth_gate(P, flat(10.0))
check("gate returns a decision", g is not None)
keep, med, tol = g
check("nothing dropped on flat ground", int(keep.sum()) == 200, str(int(keep.sum())))
check("median is the surface", abs(med - 10.0) < 0.01, f"{med:.3f} m")
check("tolerance is at least a metre", tol >= 1.0, f"{tol:.3f} m")

print("\na rooftop is dropped, and the ground is not")
d = flat(10.0)
d[:, :300] = 4000          # a 4 m-range block: 6 m of parallax
g = tr._depth_gate(P, d)
keep, med, tol = g
on_roof = (P.reshape(-1, 2)[:, 0] < 300)
check("the roof tracks went", int(keep[on_roof].sum()) == 0,
      f"{int(keep[on_roof].sum())} of {int(on_roof.sum())} kept")
check("the ground tracks stayed", int(keep[~on_roof].sum()) == int((~on_roof).sum()),
      f"{int(keep[~on_roof].sum())} of {int((~on_roof).sum())}")
check("the median followed the majority", abs(med - 10.0) < 0.1, f"{med:.3f} m")

print("\nunmatched pixels are not a surface at zero metres")
d = flat(10.0)
d[:, :400] = 0             # StereoDepth writes 0 where it found no match
g = tr._depth_gate(P, d)
keep, med, tol = g
check("zeros are excluded, not read as 0 m", abs(med - 10.0) < 0.1, f"{med:.3f} m")
no_match = (P.reshape(-1, 2)[:, 0] < 400)
check("and those tracks are not kept", int(keep[no_match].sum()) == 0,
      str(int(keep[no_match].sum())))

print("\nbeyond the baseline's reach it refuses to gate at all")
check("60 m is out of range", tr._depth_gate(P, flat(60.0)) is None)
check("40 m is out of range", tr._depth_gate(P, flat(40.0)) is None)
check("25 m is in range", tr._depth_gate(P, flat(25.0)) is not None)
check("10 m is in range", tr._depth_gate(P, flat(10.0)) is not None)

print("\nit refuses rather than gating down to nothing")
d = np.zeros((800, 1280), np.uint16)
check("an all-zero depth map is no gate", tr._depth_gate(P, d) is None)
# Scattered depths with no majority surface: tolerance cannot save it.
rng = np.random.default_rng(11)
d = rng.integers(1000, 25000, (800, 1280), dtype=np.uint16)
g = tr._depth_gate(P, d)
check("noise with no surface keeps few or refuses",
      g is None or int(g[0].sum()) < 120, "refused" if g is None else str(int(g[0].sum())))
check("too few points to judge is no gate",
      tr._depth_gate(pts(10), flat(10.0)) is None)
check("no depth at all is no gate", tr._depth_gate(P, None) is None)

print("\na gate that would drop most of the frame refuses instead")
# This is the first live run, 20 Oct: sitting on the floor the pair reported
# 1.3-1.8 m and the gate threw away 190 of 300 tracks. At that range the depth
# map is mostly noise, so the gate was making the flow worse, not better - and
# an absolute floor of 30 survivors passed it happily.
d = flat(10.0)
d[:, :700] = 3000                 # 55% of the frame at another depth
g = tr._depth_gate(P, d)
check("no majority surface -> refuse, not a hard gate", g is None,
      "refused" if g is None else f"kept {int(g[0].sum())} of 200")
d = flat(10.0)
d[:, :200] = 3000                 # 16%: a real rooftop, still a clear majority
g = tr._depth_gate(P, d)
check("a genuine minority rooftop is still gated", g is not None
      and int(g[0].sum()) < 200, "refused" if g is None else str(int(g[0].sum())))

print("\nmostly-unmatched depth is refused, however many survive")
d = flat(10.0)
d[:, :800] = 0                    # 62% of the frame has no depth at all
check("too little valid depth -> no gate", tr._depth_gate(P, d) is None)
d = flat(10.0)
d[:, :200] = 0                    # 16% unmatched, the rest is a clean surface
check("a little unmatched depth is fine", tr._depth_gate(P, d) is not None)

print("\nthe tolerance widens with range, because the sensor's noise does")
_, _, t10 = tr._depth_gate(P, flat(10.0))
_, _, t25 = tr._depth_gate(P, flat(25.0))
check("25 m is looser than 10 m", t25 > t10, f"{t10:.2f} m vs {t25:.2f} m")

print("\nend to end: step() uses it and says what it did")
def texture(seed=5, w=1280, h=800):
    import cv2
    rng = np.random.default_rng(seed)
    i = rng.integers(0, 255, (h // 8, w // 8), dtype=np.uint8)
    return cv2.GaussianBlur(cv2.resize(i, (w, h)), (3, 3), 0)


base = texture()
moved = np.roll(base, 12, axis=1)
t2 = FlowTracker(max_corners=400)
t2.step(base, depth=flat(10.0))
d2, n2 = t2.step(moved, depth=flat(10.0))
check("flat depth still measures the translation",
      d2 is not None and abs(d2[0] - 12.0) < 1.5, f"{d2}")
check("and reports the surface it used", t2.depth_med_m is not None
      and abs(t2.depth_med_m - 10.0) < 0.2, str(t2.depth_med_m))
check("with nothing dropped", t2.depth_dropped == 0, str(t2.depth_dropped))

dd = flat(10.0)
dd[:, :400] = 3000
t3 = FlowTracker(max_corners=400)
t3.step(base, depth=dd)
d3, n3 = t3.step(moved, depth=dd)
check("a rooftop block is reported as dropped tracks", t3.depth_dropped > 0,
      f"{t3.depth_dropped} dropped")

print("\nno depth is the path every flight before 20 Oct ran")
t4 = FlowTracker(max_corners=400)
t4.step(base)
d4, n4 = t4.step(moved)
check("it still works with no stereo at all",
      d4 is not None and abs(d4[0] - 12.0) < 1.5, f"{d4}")
check("and says the stereo was not used", t4.depth_med_m is None)

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
