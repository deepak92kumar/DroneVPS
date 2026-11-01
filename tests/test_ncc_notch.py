"""The periodic-furrow notch: what it is, and why it is OFF by default.

Flight 378 matched its own frames offline against demo/bam1.tif - the sheet
that DOES cover the site - and the matcher failed in a way no threshold could
catch: 57% of 2069 attempts passed its gate and landed 33.9 m from the
receiver, only 2.2% inside 10 m, and the top 1% by margin still sat at 35.9 m.

The cause is visible in the imagery. Camera and sheet agree on the structure -
the same building compound, the same field boundaries - but the ground is
ploughed, and parallel furrows dominate both. A periodic pattern correlates
with itself at every multiple of its period, so the surface grows rival peaks
as tall as the true one and the matcher locks onto the wrong furrow.

_denotch removes the strongest periodic components in the Fourier plane. In a
hand-rolled single-scale harness it looked decisive - the 39 m leg went 58.7
to 15.0 m. Inside the real locate(), with its pyramid and yaw search, it did
NOT reproduce. Measured over 360 frames, window centred on GPS_RAW_INT:

                   passed   err p50   <10m   <20m   ms/frame
    notch off       54%      42.0 m    1%     3%       80
    notch on        49%      37.0 m    4%    16%      191

      84 m leg: <20m 3% -> 19%     39 m leg: <20m 4% -> 0%

So it helps at height, hurts low, and costs 2.4x. Neither setting is usable -
4% inside 10 m is not a position aid - so the default stays off and the
function stays, with its numbers, for whoever comes back to this with a sheet
whose furrows are not four years out of date.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vps_vio"))
import map_match_ncc as NCC  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        fails.append(name)


print("the default is OFF, and that is a measurement, not an oversight")
import inspect  # noqa: E402
sig = inspect.signature(NCC.locate)
check("locate's notch defaults to False", sig.parameters["notch"].default is False,
      "on 378 it took the 39 m leg from 4% inside 20 m to 0%")
check("_edges' does too", inspect.signature(NCC._edges).parameters["notch"].default is False)

print("\nit actually removes a periodic pattern")
h = w = 192
yy, xx = np.mgrid[0:h, 0:w]
stripes = (120 + 100 * np.sin(2 * np.pi * xx / 8.0)).astype(np.uint8)
blob = stripes.copy()
blob[70:110, 70:110] = 20                      # the non-repeating structure
out = NCC._denotch(blob)
# the stripe amplitude away from the blob should collapse; the blob should not
bg_before = float(np.std(blob[0:40, :].astype(float)))
bg_after = float(np.std(out[0:40, :].astype(float)))
check("the stripes lose most of their amplitude", bg_after < 0.5 * bg_before,
      f"std {bg_before:.1f} -> {bg_after:.1f}")
check("and the square is still there",
      abs(float(out[85:95, 85:95].mean()) - float(out[0:10, 0:10].mean())) > 10.0,
      "a notch that flattens everything would be no use")

print("\nshape and type survive it")
for shape in ((64, 64), (101, 203)):
    a = (np.random.default_rng(0).random(shape) * 255).astype(np.uint8)
    o = NCC._denotch(a)
    check(f"{shape} in, same out", o.shape == shape and o.dtype == np.uint8)
    e = NCC._edges(a, True)
    check(f"{shape} through _edges(notch=True)", e.shape == shape and e.dtype == np.uint8)

print("\nDC is kept - a notch that killed the mean would be a different filter")
flat = np.full((96, 96), 137, np.uint8)
check("a flat image survives", int(NCC._denotch(flat).std()) == 0)

print()
if fails:
    print(f"{len(fails)} FAILED: " + ", ".join(fails))
    sys.exit(1)
print("ALL PASS")
