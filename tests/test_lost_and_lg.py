"""LighterGlue as XFeat's fallback, and the aircraft knowing it is lost.

Both from 4 Nov measurements on 338 frames of two flights, scored against
the M9N:

    mutual NN      accept 168/338 (50%)  err p50 3.53  WORST  10.95  >20 m 0
    LighterGlue    accept 213/338 (63%)  err p50 3.94  WORST  14.79  >20 m 0
    MNN, else LG   accept 236/338 (70%)  err p50 3.72  WORST  14.79  >20 m 0

and by how wrong the prior was, which is what this is for:

    prior  20- 50 m   MNN 22%   LG 94%
    prior  50+    m   MNN  0%   LG 15%

The lost detector fired on flight 478 at t=409 with the error at 24.7 m -
before the run from 25 m to 124 m - with zero false alarms on flight 476,
which never got lost.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vps_vio"))
import map_fix_worker as W                      # noqa: E402
import map_match_xfeat as MX                    # noqa: E402
from live_node import VpsNode                   # noqa: E402

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


print("LighterGlue is registered and gated for itself")
W.MapFixWorker._warm = lambda self: None
w = W.MapFixWorker(_Cache(), focal_px=795.0, matcher="xfeat,xfeat_lg",
                   min_alt_m=0.0, fix_period_s=0.0)
check("the comma list is accepted", w._pair == ["xfeat", "xfeat_lg"],
      str(w._pair))
check("both are trusted, in that order", w._trust == ["xfeat", "xfeat_lg"],
      str(w._trust))
check("xfeat_lg has a gate", "xfeat_lg" in w._gate, str(sorted(w._gate)))
w.stop()
raised = False
try:
    W.MapFixWorker(_Cache(), focal_px=795.0, matcher="xfeat,nonsense")
except ValueError:
    raised = True
check("a bad name in the list is an error", raised)

print("\nLighterGlue's own floor, which is the whole difference")
import inspect
src = inspect.getsource(MX.locate_lg)
check("min_inliers defaults to 20", 'setdefault("min_inliers", 20)' in src,
      "12 let six fixes over 20 m through, worst 152.67 m")
# 90 -> 85 at the pilot's instruction: NCC is trusted only below 80 m and
# LighterGlue only above 90 left 80-90 m on XFeat alone at about 38%
# coverage. 85 closes half that gap, and it is measured as free: in the
# 85-100 m band the two accept the same 36% of frames, so nothing is given
# up by starting there. The remaining 80-85 m band has XFeat at 62% with a
# worst of 6.7 m.
check("it is not asked below 85 m", 'setdefault("min_alt_m", 85.0)' in src,
      "LG accepts 19% at 55-85 m against mutual NN's 46%, so it still has "
      "a floor - just a lower one")
check("it selects the lg matching step", 'kw["pairs"] = "lg"' in src)
ls = inspect.getsource(MX.locate)
check("locate still defaults to mutual NN", 'pairs: str = "mnn"' in ls,
      "the flown path must be untouched")
check("the altitude floor is enforced in locate",
      "is not asked below" in ls)

print("\nthe expensive matcher runs only when the cheap one refuses")
ws = inspect.getsource(W.MapFixWorker._loop)
check("the loop stops on a trusted answer",
      "if any(m in got for m in self._trusted_at(alt)):" in ws,
      "running both every frame measured 2587 ms, past the 2 s period")

print("\nrelocation is tiled, bounded, and returns a hypothesis")
rs = inspect.getsource(W.MapFixWorker.relocate)
check("it tiles rather than widening", "tile_m" in rs and "step_m" in rs,
      "widening recovered 0% past 100 m")
check("it can search a ring", "ring_min_m" in rs)
check("it has a time budget", "budget_s" in rs)
check("the winner is the most inliers, not the best score",
      "res.inliers > best.inliers" in rs,
      "corr(inlier ratio, error) = +0.052, so score decides nothing")
check("it defaults to LighterGlue", 'matcher="xfeat_lg"' in rs,
      "mutual NN found 0 of 4 lost frames across 49 tiles each")


def node():
    n = VpsNode.__new__(VpsNode)
    n.lost_since = None
    n.lost_reason = ""
    n._lost_gap = 0
    n.lost_gap_n = 5
    n.lost_score_max = 0.05
    n.lost_alt_min_m = 75.0
    n.lost_sigma_m = 50.0
    n.n_lost = 0
    n.n_lost_cleared = 0
    return n


class _Res:
    def __init__(self, ok, score):
        self.ok = ok
        self.score = score


print("\nthe lost detector: five refusals, each with nothing in the window")
n = node()
for i in range(4):
    n._lost_check(_Res(False, 0.02), dict(t=100.0 + i, alt=110.0))
check("four refusals is not lost yet", n.lost_since is None,
      f"gap={n._lost_gap}")
n._lost_check(_Res(False, 0.02), dict(t=104.0, alt=110.0))
check("the fifth declares it", n.lost_since == 104.0, str(n.lost_since))
check("and says why", "refusals" in n.lost_reason, n.lost_reason)
n._lost_check(_Res(True, 0.3), dict(t=105.0, alt=110.0))
check("an accepted fix clears it", n.lost_since is None)
check("and the clear is counted", n.n_lost_cleared == 1)

print("\nand it does not fire on the cases that made it false-alarm")
n = node()
for i in range(12):
    n._lost_check(_Res(False, 0.06), dict(t=float(i), alt=110.0))
check("refusals NEAR the gate are not lost", n.lost_since is None,
      "score 0.06 means 'nearly had it', not 'nothing there'")
n = node()
for i in range(12):
    n._lost_check(_Res(False, 0.01), dict(t=float(i), alt=65.0))
check("refusals below 75 m are not lost", n.lost_since is None,
      "the 60 m matcher floor depresses the accept rate on its own")
n = node()
for i in range(12):
    n._lost_check(_Res(False if i % 3 else True, 0.01),
                  dict(t=float(i), alt=110.0))
check("an occasional success keeps the run broken", n.lost_since is None,
      f"gap={n._lost_gap}")

print("\nthe declared accuracy RAMPS, it does not step")
import map_fuse                                    # noqa: E402
n = node()
n.fuser = map_fuse.MapFuser()
n.origin_sigma_m = 2.0
n.lost_ramp_mps = 0.6
n.lost_since = 100.0
seq = []
for dt in (0.0, 10.0, 30.0, 60.0, 120.0):
    n.prev_t = 100.0 + dt
    seq.append(n.absolute_sigma_m)
check("at the moment it fires nothing jumps", seq[0] < 5.0,
      f"{seq[0]:.2f} m - a step to 50 m is a glitch, and a glitch is ALT_HOLD")
check("it rises steadily", seq[0] < seq[1] < seq[2] < seq[3],
      " -> ".join(f"{v:.1f}" for v in seq))
check("at 0.6 m/s, matching the 0.546-0.625 m/s the error really grew at",
      abs((seq[2] - seq[0]) / 30.0 - 0.6) < 0.05,
      f"{(seq[2]-seq[0])/30.0:.3f} m/s")
check("and stops at the ceiling", abs(seq[4] - 50.0) < 1e-6,
      f"{seq[4]:.2f} m")
n.lost_ramp_mps = 0.0
n.prev_t = 100.0
check("ramp 0 goes back to the step", n.absolute_sigma_m >= 50.0)

print("\nbeing lost reaches the flight controller as horiz_accuracy")
import map_fuse
n = node()
n.fuser = map_fuse.MapFuser()
n.origin_sigma_m = 2.0
clean = n.absolute_sigma_m
n.lost_since = 104.0
lost = n.absolute_sigma_m
check("sigma is normal when found", clean < 10.0, f"{clean:.2f} m")
check("and large once lost", lost >= 50.0, f"{lost:.2f} m")
_as = inspect.getsource(VpsNode.absolute_sigma_m.fget)
check("it is the declared number, not an internal one",
      "horiz_accuracy" in _as.lower() or "flight controller" in _as,
      "flight 478 declared 8.5 m while 124.7 m wrong")

print("\nthe detector can be switched off")
n = node()
n.lost_gap_n = 0
for i in range(20):
    n._lost_check(_Res(False, 0.0), dict(t=float(i), alt=110.0))
check("map_lost_gap=0 never declares lost", n.lost_since is None)

print("\nit is all in the log")
ln = open(os.path.join(os.path.dirname(__file__), "..",
                       "vps_vio", "live_node.py")).read()
for col in ("lost", "lost_reason"):
    check(f"{col} is a column", f'"{col}"' in ln)
check("lost is written as 0/1", 'int(bool(est.get("lost")))' in ln)


print("\nNCC comes back below 80 m, and only there")
from config import VpsConfig                    # noqa: E402
_c = VpsConfig()
w = W.MapFixWorker(_Cache(), focal_px=795.0, matcher=_c.matcher,
                   trust=_c.map_trust, ncc_below_m=_c.map_ncc_below_m,
                   min_alt_m=0.0, fix_period_s=0.0)
check("all three matchers run", w._pair == ["xfeat", "xfeat_lg", "ncc"],
      str(w._pair))
check("NCC is not in the base trust", w._trust == ["xfeat", "xfeat_lg"],
      str(w._trust))
check("at 70 m NCC may publish", "ncc" in w._trusted_at(70.0),
      "XFeat accepts 48% there and LighterGlue is not asked at all")
check("at 79 m it still may", "ncc" in w._trusted_at(79.0))
check("at 85 m it may not", "ncc" not in w._trusted_at(85.0),
      "the pilot flies this and reports NCC good to 80 m, best at 70")
check("at 110 m it may not", "ncc" not in w._trusted_at(110.0),
      "NCC reached 180.74 m up there")
check("and it is appended LAST at 70 m",
      w._trusted_at(70.0)[-1] == "ncc",
      "XFeat is still believed wherever it speaks - that is what takes the "
      "worst case from 68.92 m to 45.81 m")
w.stop()
w = W.MapFixWorker(_Cache(), focal_px=795.0, matcher=_c.matcher,
                   trust=_c.map_trust, ncc_below_m=0.0,
                   min_alt_m=0.0, fix_period_s=0.0)
check("map_ncc_below_m=0 switches the fallback off",
      "ncc" not in w._trusted_at(10.0))
w.stop()
check("map_trust is explicit in config, not None",
      _c.map_trust is not None,
      "a None trust with three matchers would trust NCC at every height")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
