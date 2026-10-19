"""The TF03 spike filter, in both directions.

The pilot: "tf03 bich bich mein jump bhi karta hai... uppar aur niche bhi".
Log 214 has it on an aircraft that never left the floor - RFND ranging 0.05 to
0.77 m with a median of 0.41, which is 70 cm of spread on a stationary drone
with a 0.26 m ground clearance.

cuav_mavlink_bridge already had a median of 5. A median only rejects a spike
that is in the MINORITY of its window, so three bad samples in a row walk
through it untouched, and it has no notion of how fast the ground can actually
approach. live_node's telemetry path had no filter at all.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "vps_vio"))
from lidar_filter import LidarFilter  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def run(vals, hz=20.0, **kw):
    f = LidarFilter(**kw)
    out = []
    for i, v in enumerate(vals):
        out.append(f.push(i / hz, v))
    return f, out


print("a steady reading passes straight through")
f, out = run([0.26] * 20)
check("the level is the level", abs(out[-1] - 0.26) < 1e-9, str(out[-1]))
check("nothing was rejected", f.spikes_up == 0 and f.spikes_down == 0)
check("and nothing was held", f.held == 0, str(f.held))

print("\na single spike UP is not a climb")
vals = [0.26] * 10 + [40.0] + [0.26] * 10
f, out = run(vals)
check("the spike never reaches the output", max(o for o in out if o) < 1.0,
      f"max {max(o for o in out if o):.2f} m")
check("it is counted as an up spike", f.spikes_up >= 1, str(f.spikes_up))
check("the level is unchanged afterwards", abs(out[-1] - 0.26) < 0.05, str(out[-1]))

print("\na single spike DOWN is not a landing")
vals = [12.0] * 10 + [0.30] + [12.0] * 10
f, out = run(vals)
check("the drop never reaches the output", min(o for o in out if o) > 5.0,
      f"min {min(o for o in out if o):.2f} m")
check("it is counted as a down spike", f.spikes_down >= 1, str(f.spikes_down))

print("\nthree bad samples in a row - what the median alone let through")
vals = [12.0] * 10 + [40.0, 40.5, 39.8] + [12.0] * 10
f, out = run(vals, confirm_n=5)
check("a run of three is still refused with confirm_n 5",
      max(o for o in out if o) < 20.0, f"max {max(o for o in out if o):.2f} m")

print("\nbut a REAL step is eventually believed - flying over a wall")
vals = [12.0] * 10 + [4.0] * 20
f, out = run(vals)
check("the new level is accepted", abs(out[-1] - 4.0) < 0.5, f"{out[-1]:.2f} m")
check("and it is recorded as a step, not a spike", f.steps >= 1, str(f.steps))
check("it was held while it proved itself", f.held >= 1, str(f.held))

print("\nthe held value is a real measurement, never an extrapolation")
vals = [12.0] * 10 + [40.0] + [12.0] * 5
f, out = run(vals)
seen = set(round(o, 4) for o in out if o is not None)
check("every output was once an input",
      all(any(abs(o - v) < 1e-6 for v in vals) for o in seen), str(sorted(seen)))

print("\ngarbage never enters the window")
f, out = run([0.26] * 5 + [float("nan"), float("inf"), -5.0, 500.0, None] + [0.26] * 5)
check("NaN, inf, negative and over-range all refused", f.n_out_of_range == 5,
      str(f.n_out_of_range))
check("and the level survived them", abs(out[-1] - 0.26) < 1e-9, str(out[-1]))
check("a non-number does not raise", True)

print("\nnothing is claimed before the first good reading")
f = LidarFilter()
check("no value yet", f.push(0.0, float("nan")) is None)
check("still none", f.push(0.05, None) is None)
check("the first real one defines the level", abs(f.push(0.1, 7.5) - 7.5) < 1e-9)

print("\na normal descent is not fought")
# 2 m/s down at 20 Hz is 0.10 m a sample, well inside the 8 m/s gate.
vals = [12.0 - 0.10 * i for i in range(40)]
f, out = run(vals)
check("the descent is followed", abs(out[-1] - vals[-1]) < 0.3,
      f"{out[-1]:.2f} vs {vals[-1]:.2f}")
check("and nothing was called a spike", f.spikes_up + f.spikes_down == 0,
      f"{f.spikes_up}/{f.spikes_down}")

print("\nlog 214's own numbers, on an aircraft that never moved")
# RFND 0.05..0.77, median 0.41, ground clearance 0.26.
import random  # noqa: E402
random.seed(214)
vals = []
for i in range(200):
    vals.append(0.41 if random.random() > 0.12
                else random.choice([0.05, 0.77, 0.70, 0.08]))
f, out = run(vals)
spread = max(o for o in out if o) - min(o for o in out if o)
raw_spread = max(vals) - min(vals)
print(f"      raw spread {raw_spread:.2f} m  ->  filtered spread {spread:.2f} m")
check("the spread is cut hard", spread < raw_spread / 2,
      f"{raw_spread:.2f} -> {spread:.2f} m")
check("and it settles on the real surface", abs(out[-1] - 0.41) < 0.2, str(out[-1]))

print("\nthe counters say what happened, so a flight can be reviewed")
st = f.stats()
for k in ("n_in", "out_of_range", "spikes_up", "spikes_down", "held", "steps"):
    check(f"stats has {k}", k in st)

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
