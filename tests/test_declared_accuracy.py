"""What the node DECLARES about its own error, and why both numbers moved.

On 20 Oct the pilot asked for the live VPS output to be compared against log
00000194, the last flight that worked. Four of the six fields matched. Two did
not, and the reasons turned out to be different faults.

THE ORIGIN WAS NEVER IN THE SIGMA. Measured over the three VPS flights of
20 Oct, which flew with the origin left at a stale SCR_USER1/2:

    declared sigma   p50  14.00 m    p95  22.30 m
    TRUE error       p50 151.11 m    p95 154.53 m

and the true error exceeded the declared sigma in 76.3% of samples. Take the
constant offset out and what is left - the drift - is covered properly:

    drift            p50   0.61 m    p95  16.11 m   against declared 14.10
    exceeded the declared sigma 8.9% of the time (honest 1-sigma: about 32%)

So the SHAPE was never wrong. The fuser's sigma is the uncertainty of the
DISPLACEMENT and it models that well. What was missing is that the position is
origin + displacement, and the origin's own error - the entire 151 m - was
declared as nothing at all.

The origin's error is measurable when it comes from the receiver: GPS1 reported
1.08 to 3.01 m with a lock across logs 190-220, and UINT32_MAX with no lock
(the reader discards that). A point typed into SCR_USER1/2 states nothing, and
that is exactly the case that was 147-157 m out - so unknown falls back to the
map's own error, which is the best this system can ever know absolutely.

THE RATE GATE LOST 9% TO ITS OWN ARITHMETIC. `_next_send = t + period` starts
each period from the moment the send happened, not from the deadline it was
due. The gate can only fire on a frame and frames arrive every 30 ms, so every
send lands a little late and that remainder is thrown away each time. Replayed
on the real frame timestamps of all thirteen recorded VPS flights:

    _next_send = t + period      3.913 Hz
    _next_send += period         4.295 Hz        target 4.30, log 194 = 4.31
"""
import inspect
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "vps_vio"))

from map_fuse import MapFuser  # noqa: E402
from live_node import ORIGIN_UNKNOWN_SIGMA_M  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


src = open(os.path.join(HERE, "vps_vio", "live_node.py")).read()

# ------------------------------------------------- the rate gate's arithmetic

print("the gate advances from the deadline, not from the moment it fired")
check("the period is added to _next_send, not to now",
      "self._next_send += self.send_period_s" in src)
check("and a deadline already in the past is restarted, not chased",
      "if self._next_send < t:" in src,
      "otherwise a stalled camera is followed by a 30 Hz burst")

PER = 1.0 / 4.3


def replay(timestamps, form):
    nxt = None
    sent = 0
    for t in timestamps:
        if nxt is not None and t < nxt:
            continue
        if nxt is None or form == "now":
            nxt = t + PER
        else:
            nxt += PER
            if nxt < t:
                nxt = t + PER
        sent += 1
    return (sent - 1) / (timestamps[-1] - timestamps[0])


# The REAL frame timestamps where they are on disk, because the whole finding
# came out of them and a synthetic loop understates the jitter: an 8% chance of
# one extra 25 ms frame gives the old form 4.14 Hz, where the recordings give
# 3.91. The real loop's dt is p50 0.030 with p99 0.050-0.060 and occasional
# multi-second gaps, and that tail is what costs the rate.
import csv as _csv
import glob as _glob

real = []
for path in sorted(_glob.glob(os.path.join(HERE, "flight_logs", "vps",
                                           "vps_000*", "vps_*.csv"))):
    if path.endswith("_imu.csv"):
        continue
    col = []
    for r in _csv.DictReader(open(path)):
        try:
            col.append(float(r["t"]))
        except (KeyError, TypeError, ValueError):
            pass
    if len(col) > 2000:
        real.append(np.array(col))

if real:
    now = [replay(x, "now") for x in real]
    dl = [replay(x, "deadline") for x in real]
    print(f"  on {len(real)} recorded flights' own timestamps: "
          f"t+period {np.mean(now):.3f} Hz, deadline {np.mean(dl):.3f} Hz")
    check("the old form fell short on every recorded flight",
          max(now) < 4.1, f"worst {max(now):.3f}, mean {np.mean(now):.3f}")
    check("the new form holds 4.3 Hz on all of them",
          abs(np.mean(dl) - 4.30) < 0.03,
          f"mean {np.mean(dl):.3f} against log 194's 4.31")
else:
    # No recordings in this checkout. The synthetic loop still separates the
    # two forms, just by less, so the assertion is loosened to match what it
    # can actually show rather than to the real figure.
    rng = np.random.default_rng(20261020)
    t = 0.0
    ts = []
    for _ in range(30 * 180):
        t += 0.030 + (0.025 if rng.random() < 0.08 else 0.0)
        ts.append(t)
    ts = np.array(ts)
    r_now, r_dl = replay(ts, "now"), replay(ts, "deadline")
    print(f"  no recordings here; synthetic 30 Hz loop: "
          f"t+period {r_now:.3f} Hz, deadline {r_dl:.3f} Hz")
    check("the old form falls short of 4.3 Hz", r_now < 4.25,
          f"{r_now:.3f} Hz - the thirteen real flights measured 3.913")
    check("the new form holds the rate", abs(r_dl - 4.30) < 0.05,
          f"{r_dl:.3f} Hz against log 194's 4.31")

# A real gap must not be followed by a burst. A clean 30 Hz loop, a 3 s stall
# where the camera says nothing, then 30 Hz again - which is what a reconnect
# looks like. Chasing the missed deadlines would send the whole backlog at
# frame rate, and the 30 Hz burst is exactly what this gate exists to stop.
clean = [0.030 * i for i in range(1, 301)]
gap = clean + [clean[-1] + 3.0 + 0.030 * i for i in range(200)]
nxt = None
after = []
for t in gap:
    if nxt is not None and t < nxt:
        continue
    if nxt is None:
        nxt = t + PER
    else:
        nxt += PER
        if nxt < t:
            nxt = t + PER
    after.append(t)
tail = [x for x in after if x > clean[-1]]
burst = max(sum(1 for x in tail if w <= x < w + 1.0) for w in tail) if tail else 0
check("after a 3 s gap it does not burst", burst <= 5,
      f"most sends in any one second afterwards: {burst}")

# ------------------------------------------------- the origin is in the sigma

print("\nthe accuracy sent to the FC includes the origin's own error")
check("set_origin takes the origin's uncertainty",
      "def set_origin(self, lon: float, lat: float, sigma_m: float | None = None)" in src)
check("there is a separate absolute sigma", "def absolute_sigma_m" in src)
check("it combines the origin with the drift in quadrature",
      "np.hypot(origin, drift)" in src)
check("an unknown origin falls back to the map's own error",
      "origin = ORIGIN_UNKNOWN_SIGMA_M" in src,
      "a hand-typed SCR_USER1/2 states no accuracy, and was 147-157 m out")
check("and that fallback is its own constant, not the fuser's weight",
      "ORIGIN_UNKNOWN_SIGMA_M = 20.0" in src,
      "until 23 Oct this read the fuser's fix_sigma_m, so tuning how hard "
      "the filter pulls towards the map also changed what the aircraft was "
      "told its absolute position was worth")
check("and that is what the send is given, not the drift alone",
      'est["sigma_abs"], est["yaw"], vel=est["vel"]' in src)
check("both are logged so a flight can be audited",
      '"sigma_abs_m"' in src and '"sigma_m"' in src)

# The arithmetic itself, without a camera.
class FakeNode:
    def __init__(self, fuser):
        self.fuser = fuser
        self.origin_sigma_m = None
    absolute_sigma_m = property(
        lambda self: float(np.hypot(
            self.origin_sigma_m if self.origin_sigma_m is not None
            else ORIGIN_UNKNOWN_SIGMA_M,
            float(self.fuser.sigma_m))))


f = MapFuser(flow_drift_frac=0.07, fix_sigma_m=8.0)
n = FakeNode(f)
check("with no origin accuracy it never claims better than the map",
      n.absolute_sigma_m >= 20.0,
      f"{n.absolute_sigma_m:.2f} m against the map's own 20 m error")
check("and log 194, which flew, declared 21.36 m p50",
      19.0 <= n.absolute_sigma_m <= 25.0, f"{n.absolute_sigma_m:.2f} m")

n.origin_sigma_m = 2.46          # what GPS1 reported in log 00000194
fresh = n.absolute_sigma_m
check("a receiver origin is worth much better than that",
      fresh < 5.0, f"{fresh:.2f} m from a 2.46 m origin and a 2 m displacement")

# Drive it and watch which term owns the answer. The variance adds per step, so
# the drift grows as the square root of distance: with flow_drift_frac 0.07 and
# a 0.15 m noise floor, a 1 m step adds 0.22^2 of variance, which needs about
# 8 km before the drift alone reaches the map's own 20 m. That is the honest
# shape of it - over the 3-5 km mission this aircraft is for, a receiver origin
# at 2.46 m is small and the drift is the whole number.
for d, expect in ((383, "the longest flight so far"),
                  (3000, "the short end of the mission"),
                  (8000, "where the drift reaches the map's own error")):
    g = MapFuser(flow_drift_frac=0.07, fix_sigma_m=8.0)
    nn = FakeNode(g)
    nn.origin_sigma_m = 2.46
    for _ in range(d):
        g.predict(1.0, 0.0)
    print(f"    after {d:5d} m ({expect}): drift {g.sigma_m:5.2f} m, "
          f"declared {nn.absolute_sigma_m:5.2f} m")
g = MapFuser(flow_drift_frac=0.07, fix_sigma_m=8.0)
n2 = FakeNode(g)
n2.origin_sigma_m = 2.46
for _ in range(383):
    g.predict(1.0, 0.0)
check("at 383 m - the longest flight flown - the drift is the larger term",
      g.sigma_m > 2.46, f"drift {g.sigma_m:.2f} m against the origin's 2.46")
check("and the declared figure is above both", 
      n2.absolute_sigma_m > max(g.sigma_m, 2.46),
      f"{n2.absolute_sigma_m:.2f} m")
check("it is never below the origin's own error",
      n2.absolute_sigma_m >= 2.46)

# ------------------------------------------------- the number that was wrong

print("\nthe fuser the flying code builds")
check("fix_sigma_m is 8, matching MapFuser's default and the log 232 sweep",
      "MapFuser(flow_drift_frac=0.07, fix_sigma_m=8.0)" in src
      and inspect.signature(MapFuser).parameters["fix_sigma_m"].default == 8.0,
      "this check exists because live_node once said 12 while the default, "
      "the docstring and the tests all said 20 - the value may move, the "
      "disagreement may not")
check("and what the FC is told is still the map's measured 20 m",
      ORIGIN_UNKNOWN_SIGMA_M == 20.0,
      f"{ORIGIN_UNKNOWN_SIGMA_M} m - a weight that helps the filter is not "
      f"a claim about where the aircraft is")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
