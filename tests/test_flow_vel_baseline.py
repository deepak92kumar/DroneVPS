"""The flow velocity is measured over a baseline, and the baseline breaks on a gap.

Velocity is pixels x altitude / focal, so at 80 m one pixel is 80/791.72 =
0.101 m and at 30 fps that is 3.03 m/s PER PIXEL of frame-to-frame jitter.
Measuring over one frame, which is what live_node did until 22 Oct 2026, is
therefore the worst available choice. Per-axis error against the raw receiver
on log 226, by baseline length:

    N frames   lag     5-25m   25-45m   45-70m   70-85m
       1      0.02 s    1.79     8.45     9.67    17.79 m/s
      20      0.33 s    0.25     1.76     1.31     3.75
      30      0.50 s    0.15     1.31     0.83     2.61

Summing N displacements and dividing by their span divides coherent motion by
N but white jitter only by sqrt(N). The cost is lag, and it is NOT compensated
- GPS2_DELAY_MS stays 0, because one GPS delay applies to the position too and
the position is the part that works (SP never left 0.56 against a 1.2 gate).

Two properties matter and they pull against each other:

  1. Real, sustained motion must come back at full magnitude. The 8 Oct
     module read 40% of truth at sustained speed and nothing aboard could
     tell; a baseline that averaged motion away would repeat that.
  2. The span must never straddle a gap. The displacement of a refused or
     lost frame is missing while its time still passes, so a span across it
     reports less motion than there was - the same failure, arrived at from
     the other direction.
"""
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "vps_vio"))

from config import VpsConfig

FAILED = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}   {detail}")
    if not ok:
        FAILED.append(label)


class Baseline:
    """Exactly what live_node.step does with the accepted displacements."""

    def __init__(self, n):
        from collections import deque
        self.n = n
        self.cum_n = self.cum_e = 0.0
        self.hist = deque(maxlen=n + 1)

    def push(self, t, dn, de, dt):
        if not self.hist:
            self.hist.append((t - dt, self.cum_n, self.cum_e))
        self.cum_n += dn
        self.cum_e += de
        self.hist.append((t, self.cum_n, self.cum_e))
        t0, n0, e0 = self.hist[0]
        span = t - t0
        if span <= 0.0:
            return None
        return (self.cum_n - n0) / span, (self.cum_e - e0) / span

    def gap(self):
        self.hist.clear()


print(__doc__)

cfg = VpsConfig()
N = cfg.flow_vel_frames
print(f"VpsConfig.flow_vel_frames = {N}")
check("the baseline is longer than one frame", N > 1, f"{N}")
# The bound was 0.5 s and moved to 0.75 on 22 Oct with the sweep behind it.
# Below 60 m, where the map never runs, the velocity error at 0.5-2 m/s is
# 0.90-1.22 at N=20 and 0.51-0.60 at N=40 - the latter under the 0.66 m/s
# deadband at 31 m, so the band LOIT_SPEED_MS=3.0 commands is suppressed.
# N=60 reads better still (0.36-0.49) and costs 1.0 s, which a position
# controller acting on stale velocity can turn into oscillation; it is not
# taken without a flight behind it.
check("and not so long that the lag passes three quarters of a second",
      N / 2 / 30.0 <= 0.75, f"lag {N / 2 / 30.0:.2f} s at 30 fps")

DT = 1.0 / 30.0

print("\n1. sustained motion comes back at full magnitude")
for truth in (0.5, 2.0, 6.0):
    b = Baseline(N)
    out = None
    for i in range(N * 3):
        out = b.push((i + 1) * DT, truth * DT, 0.0, DT)
    check(f"a steady {truth:.1f} m/s reads {out[0]:.3f}",
          abs(out[0] - truth) < 1e-6)

print("\n2. white jitter falls, and roughly as 1/sqrt(N)")
rnd = random.Random(5)
JIT = 0.101          # one pixel at 80 m, in metres
res = {}
for n in (1, 5, 20):
    b = Baseline(n)
    vals = []
    for i in range(4000):
        v = b.push((i + 1) * DT, rnd.gauss(0.0, JIT), 0.0, DT)
        if i > n:
            vals.append(v[0])
    res[n] = math.sqrt(sum(x * x for x in vals) / len(vals))
    print(f"        N={n:<3d} noise {res[n]:6.3f} m/s")
check("N=20 is quieter than one frame", res[20] < res[1],
      f"{res[20]:.3f} < {res[1]:.3f}")
ratio = res[1] / res[20]
check("and by something close to sqrt(20) = 4.47", 2.5 < ratio < 7.0,
      f"{ratio:.2f}x")
check("one frame alone is about 3 m/s on a single pixel at 80 m",
      2.0 < res[1] < 4.5, f"{res[1]:.2f} m/s")

print("\n3. the span never straddles a gap")
# Move steadily, lose a frame's displacement, and keep moving. Without the
# clear, the span covers time that carries no displacement and the speed
# reads low - which is the 8 Oct failure mode.
b = Baseline(N)
for i in range(N * 2):
    out = b.push((i + 1) * DT, 2.0 * DT, 0.0, DT)
before = out[0]
b.gap()                        # what live_node does when flowed is False
out = b.push((N * 2 + 5) * DT, 2.0 * DT, 0.0, DT)
check("before the gap it reads the truth", abs(before - 2.0) < 1e-6,
      f"{before:.3f} m/s")
# This assertion was ">= 2.0" first and it PASSED on 82.000 m/s, because the
# history was being re-seeded with a zero endpoint while the running total
# stayed large. One-sided assertions hide the direction they do not test.
check("right after the gap it reads the truth, not a spike",
      abs(out[0] - 2.0) < 1e-6, f"{out[0]:.3f} m/s")

# And the thing the clear prevents, shown directly.
b2 = Baseline(N)
for i in range(N * 2):
    b2.push((i + 1) * DT, 2.0 * DT, 0.0, DT)
stale = b2.push((N * 2 + 30) * DT, 2.0 * DT, 0.0, DT)   # 30 frames of silence
check("without a clear the same gap would under-report", stale[0] < 2.0 * 0.6,
      f"{stale[0]:.3f} m/s instead of 2.0 - this is why it clears")

print("\n4. a velocity exists from the very first frame")
b = Baseline(N)
first = b.push(DT, 1.0 * DT, 0.0, DT)
check("the first accepted frame already yields one", first is not None,
      f"{first[0]:.3f} m/s")
check("and it equals the single-frame value there", abs(first[0] - 1.0) < 1e-6)

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
    sys.exit(1)
print("ALL PASS")
