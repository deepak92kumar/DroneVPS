"""The 4 Nov findings, held in place.

NCC used to win every frame it accepted. The worker's loop kept the FIRST
result unless a later one was ok where the earlier was not, and the pair
reads ["ncc", "xfeat"], so on flight 478 all 153 published fixes were NCC's
and XFeat's 51 successes were computed, used for the disagreement number,
and discarded. Measured on 675 frames of two flights, both matchers, nadir
corrected, against the M9N:

                      n    NCC p50/p90/max      XFeat p50/p90/max
  both accepted     339    3.80  8.09  85.69     3.55  6.49  11.36
  only NCC          226   25.01 146.55 180.74   (XFeat declined)
  only XFeat          9   (NCC declined)         5.18  7.37   9.00
  neither           101   both about 104 m

                   fix rate   p50    p90      max    >20 m  >50 m
  NCC wins (old)   574/675    4.91  89.83  180.74     132    109
  XFeat trusted    348/675    3.57   6.59   11.36       0      0

So: only a trusted matcher's answer is published, both still run, the source
is named in the result, and the nadir correction is withheld when the
attitude behind it is older than att_age_max_s.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vps_vio"))
import map_fix_worker as W                      # noqa: E402
from map_match_ncc import NccResult             # noqa: E402
from map_match_xfeat import XFeatResult         # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        fails.append(name)


class _Geo:
    def to_lonlat(self, x, y):
        return 74.0 + x * 1e-6, 30.0 + y * 1e-6


class _Cache:
    gsd = 0.5
    gsd_x = 0.5

    def covers(self, lon, lat):
        return True

    def window(self, lon, lat, radius, kind="grey"):
        return np.zeros((256, 256), np.uint8), _Geo()


def worker(*, ncc, xfeat, trust=None, **kw):
    """A worker whose two matchers say exactly what the test tells them to."""
    W.MapFixWorker._warm = lambda self: None
    w = W.MapFixWorker(_Cache(), focal_px=795.0, matcher="both",
                       min_alt_m=0.0, fix_period_s=0.0, trust=trust, **kw)
    w._all_locate = {
        "ncc": lambda *a, **k: NccResult(*ncc),
        "xfeat": lambda *a, **k: XFeatResult(*xfeat),
    }
    return w


def run(w, **kw):
    w.submit(np.zeros((400, 640), np.uint8), prior_lon=74.0, prior_lat=30.0,
             altitude_m=100.0, yaw_deg=0.0, **kw)
    for _ in range(500):
        r = w.poll()
        if r is not None:
            return r
        time.sleep(0.01)
    return None


# NccResult and XFeatResult take (ok, lon, lat, score, margin, inliers,
# total_matches, reason) in that order - they are shaped alike on purpose.
GOOD_NCC = (True, 74.1, 30.1, 0.90, 0.9, 0, 0, "")
GOOD_XF = (True, 74.2, 30.2, 0.30, 0.9, 40, 200, "")
BAD_XF = (False, 0.0, 0.0, 0.01, 0.0, 3, 200, "only 3 inliers")
BAD_NCC = (False, 0.0, 0.0, 0.00, 0.0, 0, 0, "weak peak")

print("both accepted: XFeat is the one published")
w = worker(ncc=GOOD_NCC, xfeat=GOOD_XF)
try:
    r = run(w)
finally:
    w.stop()
check("a result came back", r is not None)
if r is not None:
    check("it is a fix", r.ok)
    check("it is XFeat's", r.source == "xfeat", f"source={r.source!r}")
    check("and XFeat's position, not NCC's", abs(r.lat - 30.2) < 1e-6,
          f"lat={r.lat}")

print("\nonly NCC accepted: nothing becomes a fix")
w = worker(ncc=GOOD_NCC, xfeat=BAD_XF)
try:
    r = run(w)
finally:
    w.stop()
check("a result came back", r is not None)
if r is not None:
    check("it is NOT a fix", not r.ok,
          "an untrusted matcher's answer reached the fuser")
    check("the reason says why", bool(r.reason), f"reason={r.reason!r}")
    check("the suppression is counted", w.n_untrusted == 1,
          f"n_untrusted={w.n_untrusted}")

print("\nthe old behaviour is a setting, not a code change")
w = worker(ncc=GOOD_NCC, xfeat=BAD_XF, trust="ncc,xfeat")
try:
    r = run(w)
finally:
    w.stop()
check("trust='ncc,xfeat' publishes NCC again",
      r is not None and r.ok and r.source == "ncc",
      "" if r is None else f"ok={r.ok} source={r.source!r}")

print("\ntrust has to name a matcher that is running")
raised = False
try:
    W.MapFixWorker(_Cache(), focal_px=795.0, matcher="xfeat", trust="ncc")
except ValueError:
    raised = True
check("trust='ncc' with matcher='xfeat' is an error", raised,
      "a trust list nothing satisfies would silently stop every fix")

print("\nstale attitude withholds the nadir correction")
dN, dE = W.MapFixWorker.nadir_offset_m(100.0, 0.0, 10.0, 0.0)
check("the test's tilt really does shift the answer",
      abs(dN) > 1.0 or abs(dE) > 1.0, f"dN={dN:.2f} dE={dE:.2f}")
res = {}
for tag, age in (("fresh", 0.05), ("stale", 1.50)):
    w = worker(ncc=BAD_NCC, xfeat=GOOD_XF, att_age_max_s=0.30)
    try:
        res[tag] = (run(w, roll_deg=0.0, pitch_deg=10.0, att_age_s=age),
                    w.n_att_stale)
    finally:
        w.stop()
a, n_a = res["fresh"]
b, n_b = res["stale"]
check("both produced a result", a is not None and b is not None)
if a is not None and b is not None:
    check("the age is carried", abs(a.att_age_s - 0.05) < 1e-9
          and abs(b.att_age_s - 1.50) < 1e-9,
          f"fresh={a.att_age_s} stale={b.att_age_s}")
    check("the fresh fix was corrected", abs(a.lat - 30.2) > 1e-9,
          f"lat={a.lat}")
    check("the stale one was not", abs(b.lat - 30.2) < 1e-9, f"lat={b.lat}")
    check("the stale one says so", "attitude" in b.reason.lower()
          and "withheld" in b.reason.lower(), f"reason={b.reason!r}")
    check("the fresh one does not", "withheld" not in (a.reason or "").lower())
    check("and it is counted", n_a == 0 and n_b == 1,
          f"fresh={n_a} stale={n_b}")

print("\nthe source and the attitude age reach the log")
src = open(os.path.join(os.path.dirname(__file__), "..",
                        "vps_vio", "live_node.py")).read()
for col in ("fix_source", "fix_att_age_s", "fix_reason"):
    check(f"{col} is a column", f'"{col}"' in src)
    check(f"{col} is written", f"{col}=" in src)
check("the attitude's age is measured from att_t",
      'time.time() - s["att_t"]' in src)
check("and handed to the worker", "att_age_s=f.get(\"att_age_s\")" in src)
rec = open(os.path.join(os.path.dirname(__file__), "..",
                        "vps_vio", "record_flight.py")).read()
check("record_flight timestamps ATTITUDE", "att_t=time.time()" in rec)

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
