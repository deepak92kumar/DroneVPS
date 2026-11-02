"""XFeat fits a similarity, not a translation.

Flight 406, 2 Nov: all 777 attempts reported 0 inliers while XFeat was
finding 67-166 matches on every frame. The fit asked every pair to agree on
ONE offset within tol_px=4, and the query reaches the matcher already rotated
by the compass yaw and already scaled by the barometric altitude - so the
error in those two is still in the correspondences. 2 deg of yaw is 5 px at a
150 px radius; 3% of scale is 4.5 px. Nothing could pass.

The numbers this is holding in place, measured on that flight's own frames:
inliers went from 0-2 to 8-15 (p50 10) with the same matches and the same
tolerance.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "vps_vio"))

fails = []


def check(what, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + what
          + (f"   [{detail}]" if detail else ""))
    if not ok:
        fails.append(what)


import map_match_xfeat as X  # noqa: E402

rng = np.random.default_rng(7)
# A square of true points, then the same points under a small rotation and
# scale - exactly what a 2 deg compass error and a 3% altitude error leave
# behind - plus a third of the set replaced by noise.
src = rng.uniform(0, 300, size=(90, 2))
ROT, SCALE, OFF = 2.5, 1.03, np.array([17.0, -9.0])
c, s = math.cos(math.radians(ROT)), math.sin(math.radians(ROT))
Mtrue = SCALE * np.array([[c, -s], [s, c]])
dst = (src @ Mtrue.T) + OFF
n_out = 30
dst[:n_out] = rng.uniform(0, 300, size=(n_out, 2))

print("ek chhoti rotation+scale ke saath, ek tihai outlier")
off, keep_t, mad_t = X._translation(src, dst, 4.0)
M, keep, mad, rot, scale = X._similarity(src, dst, 4.0)
check("similarity fit chalta hai", M is not None)
# The comparison is the point, not an absolute count: 60 of these pairs are
# real and the old fit keeps a fraction of them, which is what produced 0-2
# inliers in flight where this one gets 8-15.
check("purana translation-only fit isme se zyadatar asli point khota hai",
      int(keep_t.sum()) < 0.25 * int(keep.sum()),
      f"translation {int(keep_t.sum())} vs similarity {int(keep.sum())} "
      f"(60 asli)")
check("saare asli point milte hai", int(keep.sum()) >= 55,
      f"{int(keep.sum())}/60 asli")
check("ek bhi outlier andar nahi aaya", not keep[:n_out].any(),
      f"{int(keep[:n_out].sum())} outliers andar")
check("rotation wapas mili", abs(rot - ROT) < 0.2, f"{rot:.3f} vs {ROT}")
check("scale wapas mila", abs(scale - SCALE) < 0.01, f"{scale:.4f} vs {SCALE}")
check("residual tight hai", mad < 1.0, f"mad {mad:.3f} px")

print("\nek sidhi translation par bhi utna hi sahi")
dst2 = src + np.array([11.0, 4.0])
dst2[:20] = rng.uniform(0, 300, size=(20, 2))
M2, keep2, mad2, rot2, scale2 = X._similarity(src, dst2, 4.0)
check("translation bhi wapas milti hai", M2 is not None
      and abs(rot2) < 0.2 and abs(scale2 - 1.0) < 0.01,
      f"rot {rot2:.3f} scale {scale2:.4f}")
check("itne hi point bachte hai", int(keep2.sum()) >= 65,
      f"{int(keep2.sum())}/70")

print("\nbahut kam point par fit nahi karta, crash bhi nahi karta")
M3, keep3, mad3, rot3, scale3 = X._similarity(src[:3], dst[:3], 4.0)
check("3 point par None", M3 is None)
check("mask phir bhi sahi lambai ka", len(keep3) == 3)

print("\nresult rot/scale le kar jata hai")
r = X.XFeatResult(True)
check("residual_rot_deg default 0", r.residual_rot_deg == 0.0)
check("residual_scale default 1", r.residual_scale == 1.0)
src_txt = open(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "vps_vio", "map_match_xfeat.py")).read()
check("locate() similarity hi istemaal karta hai",
      "_similarity(src, dst, tol_px)" in src_txt
      and "off, keep, mad = _translation" not in src_txt)

print("\nALL PASS" if not fails else f"\n{len(fails)} FAILED: {', '.join(fails)}")
sys.exit(1 if fails else 0)
