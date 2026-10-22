"""Every map-matcher number, pinned to the report that set it.

The rebuild drifted several of these, and the pattern each time was the same:
a number changed for a sound reason that had nothing to do with why the
original was chosen. This file names the source for each one so the next change
has to argue with the measurement rather than with the code.

Read against all fifteen status reports on 21 Oct, not grepped - two of the
findings below only appear when the surrounding paragraph is read.
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "vps_vio"))

from map_fuse import MapFuser  # noqa: E402
from map_fix_worker import MapFixWorker, _GATES  # noqa: E402
import map_match_xfeat as MX  # noqa: E402
import inspect  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


live = open(os.path.join(HERE, "vps_vio", "live_node.py")).read()
W = inspect.signature(MapFixWorker.__init__).parameters
L = inspect.signature(MX.locate).parameters

# ------------------------------------------------- the search window

print("the search window - 4 Oct, flown and kept")
check("0.60 x altitude", W["window_alt_frac"].default == 0.60,
      '"fixed 100 m -> 0.60. The matcher was guessing outside what the '
      'frame could see at low altitude and cropping the search too tight at '
      'high"')
check("floored at 40 m and capped at radius_m",
      "np.clip(self.window_alt_frac * altitude_m, 40.0, self.radius_m)"
      in open(os.path.join(HERE, "vps_vio", "map_fix_worker.py")).read())
check("radius_m 150", W["radius_m"].default == 150.0)

# ------------------------------------------------- the gates

print("\nthe three gates")
check("min_inliers 12", L["min_inliers"].default == 12,
      "UNVERIFIED against real imagery; 5 Oct measured 14-32 above 140 m")
# 0.25 -> 0.10 -> 0.07, each step swept rather than guessed. 0.25 came from
# 12 Oct and was never swept; at 0.25 the 21 Oct flight passed 0 of 477
# matches at 90-100 m, its score topping out at 0.248. 0.10 -> 0.07 was swept
# on 22 Oct over 78 frames in three altitude bands:
#
#     band      min_score   accept   per-fix err   system err
#     75-90 m     0.10       34.6%      22.3 m       6.92 m
#                 0.07       42.3%      22.3 m       6.26 m
#     90-100 m    0.10       15.4%      18.2 m       8.48 m
#                 0.07       19.2%      18.1 m       7.54 m
#
# The per-fix error does not degrade, so the fixes between 0.07 and 0.10 are
# as good as those already passing. It matters because the SYSTEM error is set
# by the fix RATE, not by any one fix: measured across our own flights, 0 fixes
# gave 38-68 m above 60 m and 19 fixes at 0.27/s gave 4.50 m.
check("min_score 0.07, the swept value", L["min_score"].default == 0.07,
      "swept 0.25 -> 0.10 -> 0.07, each against real imagery")
check("the worker agrees", _GATES["xfeat"]["min_score"] == 0.07)
# Below 0.07 nothing changes, because min_inliers 12 becomes binding: at the
# ~124 matches this matcher returns, a ratio of 0.10 and a count of 12 are the
# same constraint. Lowering min_inliers alone did nothing at all.
check("and min_inliers is the floor underneath it",
      L["min_inliers"].default / 124.0 > L["min_score"].default,
      "12/124 = 0.097, above the 0.07 ratio gate")
check("but there is still a gate, which is what 12 Oct was about",
      L["min_score"].default > 0.0,
      'the XFeat branch once hard-coded ok=True and "189 of 189 fixes were '
      'accepted including ones 85.66 m out"')
check("min_margin 0.20", L["min_margin"].default == 0.20,
      "no report sets this; it is a code-level value and unchanged")
check("tol_px 4.0", L["tol_px"].default == 4.0, "likewise")

# min_score is pinned at its report value above, but the report that set it
# also doubted it in the same paragraph, and the 21 Oct measurement agrees:
#
#   12 Oct: "Its rejections track our own error - 1, 13, 3, 15 against errors
#   of 17, 33, 17 and 74 m - because the matcher takes its window from our
#   estimate, so the gate closes exactly when it needs to be open. map_fuse's
#   own outlier gate already concedes after a run of rejections; a score gate
#   placed before it bypasses that."
#
#   21 Oct, 477 matches at 90-100 m: the score's correlation with the actual
#   error is -0.118, the inliers' -0.158, the margin's +0.034. None of the
#   three predicts how wrong a fix is, and no frame in the flight reached 0.25
#   - the maximum was 0.248.
#
# Changing it is the pilot's call, so this pins what is set rather than what
# might be better.

# ------------------------------------------------- the altitude floor

print("\nthe altitude floor - and the proposal to lower it was RULED OUT")
check("min_alt_m 60", W["min_alt_m"].default == 60.0)
check("not 30", W["min_alt_m"].default != 30.0,
      '9 Oct overturned 8 Oct: "Ruled out. That recommendation rested on the '
      'map being accurate at 30-45 m. With the fix\'s own position logged, '
      'the map is 20 m wrong overall and 33.5 m wrong when the aircraft is '
      'stationary, which is exactly the landing case it was meant to improve"')

# ------------------------------------------------- the fix period

print("\nthe fix period - 2.0, and 1.0 was flown and reverted on 22 Oct")
# The full history, because 1.0 looked like recovered knowledge and was not:
#
#   0.4 s  5 Oct: "It now runs every 2 s instead of every 0.4, which keeps
#          the bound and removes most of the shove" - 1.18 m/s of push on a
#          stationary aircraft above 80 m.
#   2.0 s  what 5 Oct moved to, and what the post-SSD rebuild took.
#   1.0 s  what the pilot's archived 11 Oct reports record - "the step cap is
#          0.500 m per fix at 1 fix/s" - and later evidence than 5 Oct, so it
#          was tried on 22 Oct. Flown as log 227 and reverted the same day.
#
# What log 227 measured, against log 226 of the day before at the same height:
#
#     at 82-83 m     XKF4.SP max     fix_period
#       log 226          0.13            2.0 s
#       log 227          1.17            1.0 s     against a 1.2 gate
#
# Nine times worse, and the mechanism is plain: map_fuse.update moves the
# POSITION by up to max_correction_m and does not touch the velocity, which
# is built only from flow. At 1 Hz the EKF is told the aircraft jumped
# without moving, twice as often, and its position and velocity states
# disagree. 5 Oct's 1.18 m/s of push is the same fault seen from outside.
#
# So an archived value from the old airframe does not override a measurement
# on this one, however much later the archive is.
check("MapFixWorker's default is 2.0", W["fix_period_s"].default == 2.0)
check("the CLI agrees",
      'ap.add_argument("--fix-period", type=float, default=2.0,' in live)
check("and so does VpsNode", "fix_period_s: float = 2.0" in live,
      "three defaults for one number is how 0.5 survived")

# And the other half of the same lever, which must NOT move with it.

# ------------------------------------------------- the fuser

print("\nwhat the fuser does with a fix that passes")
f = MapFuser()
check("fix_sigma_m 20", abs(f.fix_var ** 0.5 - 20.0) < 1e-6,
      '9 Oct: "the map\'s median error is 20.0 m against our own 6.6 m"')
check("max_correction_m 0.5", abs(f.max_correction_m - 0.5) < 1e-9,
      '3 Oct: raised to 1.5-3.0 once and "the hold fell from 11-14 s to '
      '4.5-9.0 s while map_move_m sat at exactly 1.50". Reverted the same '
      'afternoon, and 11 Oct still records "0.500 m per fix" - so the rate '
      'comes from fix_period, which 11 Oct flew at 1 fix/s, not from here')
check("gate_sigmas 3.0", abs(f.gate - 3.0) < 1e-9)

# ------------------------------------------------- what must NOT come back

print("\nthree things that were flown and REVERTED, and must stay out")
src_all = live + open(os.path.join(HERE, "vps_vio", "map_fix_worker.py")).read() \
    + open(os.path.join(HERE, "vps_vio", "map_fuse.py")).read()
# Matched on the IDENTIFIER, not on the word. It was "RAMP" anywhere in three
# files until 22 Oct, when a comment about the velocity baseline's altitude
# ramp - a different mechanism entirely, in live_node - tripped it. What must
# stay out is spreading the MAP correction over time, not the word.
check("no MAP_RAMP_S", "MAP_RAMP" not in src_all.upper()
      and "ramp_s" not in src_all,
      '5 Oct: "It killed the implied-speed spike and DOUBLED the hover box '
      'above 60 m, 6.19 -> 11.26 m... a spread lie is more believable than a '
      'sharp one"')
check("no MAP_FIX_BIAS", "FIX_BIAS" not in src_all.upper()
      and "fix_bias" not in src_all,
      '13 Oct: "With the bias constant gone the map\'s offset is now N -1.69 '
      '/ E -0.22 m... what was removed was the error we were adding to it"')
check("anisotropic resampling stays behind an env var",
      'os.environ.get("XFEAT_ANISO")' in
      open(os.path.join(HERE, "vps_vio", "map_match_xfeat.py")).read(),
      '12 Oct: switching it on "made the total error worse, 12.21 to 14.44 m. '
      'Ruled out"')

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
