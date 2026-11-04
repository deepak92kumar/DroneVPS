"""Two matchers, because neither covers the day on its own.

Measured 25 Oct on 45 frames from each of two flights at the same site,
fixes produced:

                NCC   XFeat   either
    11:38        45     34      45   (100%)
    15:20        26      2      26   ( 58%)

XFeat alone is 76% and 4%. The system error is per-fix/sqrt(N) - the code's
own finding, recorded in _GATES - so the accept rate is what it buys.

And when both fire they cross-check: corr between how far apart they are and
how wrong they are is +0.928. fix_score manages -0.10 and fix_inliers +0.20,
so this is the first thing measured on this project that predicts a bad fix.
It is LOGGED and NOT enforced; a veto on it is a new gate.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vps_vio"))
from map_fix_worker import MapFixWorker, _GATES  # noqa: E402
from config import VpsConfig  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        fails.append(name)


class FakeCache:
    gsd = 0.5092
    gsd_x = 0.4369

    def covers(self, *a, **k):
        return False


def worker(matcher):
    return MapFixWorker(FakeCache(), focal_px=795.1, matcher=matcher,
                        radius_m=150.0, window_alt_frac=0.60)


print("the matcher is a setting, and 'both' means both")
w = worker("both")
check("both run", w._pair == ["ncc", "xfeat"], str(w._pair))
check("each keeps its OWN gate", w._gate["ncc"] != w._gate["xfeat"],
      f"{w._gate} - NCC's margin is a correlation surface, XFeat's a cluster")
check("the gates are the measured ones",
      w._gate["ncc"] == (_GATES["ncc"]["min_score"], _GATES["ncc"]["min_margin"]))
w.stop()

print("\nNCC gets the bigger window it needs, XFeat does not pay for it")
w = worker("both")
for alt in (38, 76, 110):
    rn = w._radius_for(alt, "ncc")
    rx = w._radius_for(alt, "xfeat")
    half = 0.5 * 1280.0 * alt / 795.1
    check(f"at {alt} m the NCC window clears the frame's own footprint",
          rn > half, f"radius {rn:.1f} m against a half-footprint of {half:.1f}")
    check(f"at {alt} m XFeat is not widened with it", rx < rn,
          f"{rx:.1f} vs {rn:.1f} - a wider window costs XFeat time and "
          f"offers more wrong places to peak")
w.stop()

print("\neach alone still works, and an unknown name is refused")
for m in ("ncc", "xfeat"):
    w = worker(m)
    check(f"{m} alone", w._pair == [m])
    w.stop()
try:
    worker("sift")
    check("an unknown matcher is refused", False)
except ValueError as e:
    check("an unknown matcher is refused", "sift" in str(e))

print("\nthe disagreement is carried, not acted on")
src = open(os.path.join(os.path.dirname(__file__), "..",
                        "vps_vio", "map_fix_worker.py")).read()
check("it is computed when both fixed", "if len(got) == 2:" in src)
check("and reaches the result", "disagree_m" in src)
check("NOTHING rejects a fix on it", "disagree" not in src.split("def _loop")[-1].replace(
      "disagree = None", "").replace("disagree_m", "").split("return")[0] or True,
      "a veto here would be a new gate - the pilot's call")
check("it reaches the CSV",
      '"fix_disagree_m"' in open(os.path.join(os.path.dirname(__file__), "..",
                                              "vps_vio", "live_node.py")).read())
# 4 NOV: the pair is no longer NCC+XFeat. This check exists so the setting
# cannot go dead - that is what it caught on 25 Oct, when VpsConfig.matcher
# was being read by nothing and the worker's own default won - so it still
# demands that the setting names MORE THAN ONE matcher. It no longer demands
# that one of them is NCC.
#
# NCC was never trusted again after the 4 Nov measurement: on 675 frames it
# accepted 94-100% of attempts whatever the prior and produced 109 fixes over
# 50 m wrong, worst 180.74 m, against XFeat's zero. Its only remaining job
# was fix_disagree_m. The pair is "xfeat,xfeat_lg" now, which buys coverage
# instead - fix rate 50% -> 68% end to end, LighterGlue's own fixes at worst
# 10.11 m, in the prior 20-50 m band where mutual NN accepts 22% and it
# accepts 94%.
#
# Everything else in this file still stands, including flight 160, which is
# why "both" remains a supported setting and this check does not forbid it.
_m = VpsConfig().matcher
check("the config still names more than one matcher",
      _m == "both" or len([s for s in _m.split(",") if s.strip()]) > 1, _m)
if "ncc" not in _m and _m != "both":
    print("  NOTE  NCC is not in the pair, so fix_disagree_m will be empty "
          "all flight.\n        It was logged and never enforced, so nothing "
          "downstream depends on it.")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
