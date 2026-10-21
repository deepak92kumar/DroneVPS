"""Derotation runs off the camera's gyro, and that is a RESTORATION.

Gyro derotation is not a new idea on this project. It was built, flown and
carried at 95% from 28 Sep until the NVMe went, and the rebuild lost it:

    28 Sep  "Tilt taken from the camera's own 400 Hz gyro instead of the
             flight controller's attitude over MAVLink"       80%
    29 Sep  "Availability 39% to 100% on the bench, 96% in flight"   85%
    30 Sep  "1.5% of frames in the afternoon, 97.7% by evening"      95%
     4 Oct  "The largest single reduction in the chain"              95%

WHY THE FC'S ATTITUDE CANNOT DO IT. It arrives at about 4 Hz against a 30 Hz
loop, so it reports no rotation on most frames and several frames' worth on the
rest - which then gets subtracted from ONE frame's shift. 30 Sep has the
extreme: "the FC's attitude had sat frozen for four frames and then jumped 4.67
deg of roll and 11.82 of pitch in one, so the derotation subtracted 167 px from
a 1 px shift - 24 m at 114 m over 0.03 s, 802 m/s."

Measured on the 21 Oct flight, where the FC reported 0.000 deg on 88% of
frames, per-frame speed error against the real receiver:

    no derotation at all       p50 2.50  p95 24.01 m/s
    the FC's attitude          p50 2.97  p95 31.05     <- WORSE than nothing
    the gyro, sign corrected   p50 1.15  p95  7.54

and through the whole velocity path, 11398 samples above 5 m:

    FC attitude                p50 0.725  p95 3.576   1.9% over 5 m/s
    gyro                       p50 0.155  p95 1.243   none over 5 m/s

That velocity is what tripped XKF4.SV to 7.95 and 10.92 against a 1.2 gate and
dropped the aircraft into ALT_HOLD on both VPS handovers of that flight.

THE COEFFICIENTS come from the 3 Oct report, not from this one flight:
"Roll re-measured on hover only, pooled over 15 flights and 39723 samples:
-14.8 px/deg, steady when restricted to the strongest rotations, so -13.82 was
6% short. Pitch deliberately left at 13.82 - its fit climbs toward theory as
signal grows, which is attenuation, not gain." 13.82 is 791.72*tan(1 deg).

Replaying the 21 Oct flight cannot separate them - 13.82/13.82 gives p50 0.154
against 14.80/13.82's 0.155 - so 39723 samples decide it rather than 11398.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "vps_vio"))

from flow_odom import (ATT_PX_PER_DEG_PITCH_Y, ATT_PX_PER_DEG_ROLL_X,  # noqa: E402
                       GYRO_PX_PER_DEG_PITCH_Y, GYRO_PX_PER_DEG_ROLL_X,
                       attitude_shift_px, gyro_shift_px)

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


src = open(os.path.join(HERE, "vps_vio", "live_node.py")).read()

# ------------------------------------------------- the coefficients

print("the coefficients are the 3 Oct report's, not this flight's")
check("roll is -14.80 px/deg", abs(GYRO_PX_PER_DEG_ROLL_X + 14.80) < 1e-9,
      "15 flights, 39723 samples, hover only")
check("pitch is -13.82 px/deg", abs(GYRO_PX_PER_DEG_PITCH_Y + 13.82) < 1e-9,
      "left at theory on purpose - its fit was attenuation, not gain")
theory = 791.72 * np.tan(np.radians(1.0))
check("13.82 really is the geometry", abs(abs(GYRO_PX_PER_DEG_PITCH_Y) - theory) < 0.01,
      f"791.72 * tan(1 deg) = {theory:.3f}")
check("roll sits about 7% above theory, as the report says",
      1.05 < abs(GYRO_PX_PER_DEG_ROLL_X) / theory < 1.09,
      f"{abs(GYRO_PX_PER_DEG_ROLL_X) / theory:.3f}x")

print("\nand the SIGN is the fault that was being fixed")
check("the gyro's coefficients are NEGATIVE",
      GYRO_PX_PER_DEG_ROLL_X < 0 and GYRO_PX_PER_DEG_PITCH_Y < 0,
      "measured against the FC over matched intervals: scale -0.983 to -1.008")
check("the FC's are positive, so the two really do oppose",
      ATT_PX_PER_DEG_ROLL_X > 0 and ATT_PX_PER_DEG_PITCH_Y > 0)
a = attitude_shift_px((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
g = gyro_shift_px(1.0, 0.0)
check("one degree of roll moves them opposite ways",
      a[0] * g[0] < 0, f"FC {a[0]:+.2f} px, gyro {g[0]:+.2f} px")

print("\nthe shift is linear in the angle and the axes do not mix")
for deg in (0.1, 0.5, 1.0, 5.0):
    r = gyro_shift_px(deg, 0.0)
    p = gyro_shift_px(0.0, deg)
    if abs(r[0] - GYRO_PX_PER_DEG_ROLL_X * deg) > 1e-9 or r[1] != 0.0:
        check(f"roll {deg} deg", False, str(r))
    if abs(p[1] - GYRO_PX_PER_DEG_PITCH_Y * deg) > 1e-9 or p[0] != 0.0:
        check(f"pitch {deg} deg", False, str(p))
check("roll moves only x and pitch only y, at every angle tried", True,
      "0.1 to 5 deg")
check("zero rotation is zero shift", tuple(gyro_shift_px(0.0, 0.0)) == (0.0, 0.0))

# The 30 Sep failure, in numbers: a frozen FC attitude releasing four frames of
# rotation into one.
print("\nthe 30 Sep case the FC path cannot avoid")
stale = attitude_shift_px((0.0, 0.0, 0.0), (4.67, 11.82, 0.0))
check("four frames of rotation arriving at once is 167 px",
      160 < float(np.hypot(*stale)) < 220,
      f"{np.hypot(*stale):.0f} px subtracted from a 1 px shift")
one_frame = gyro_shift_px(4.67 / 4.0, 11.82 / 4.0)
check("the gyro would have spread it over the four frames it happened in",
      float(np.hypot(*one_frame)) < float(np.hypot(*stale)) / 3.0,
      f"{np.hypot(*one_frame):.0f} px in the frame that rotated")

# ------------------------------------------------- the wiring

print("\nthe wiring")
check("the gyro is the default", 'derotate: str = "gyro"' in src)
check("and the flag can still reproduce an old flight",
      'choices=("gyro", "fc")' in src)
check("the gyro path is taken when the frame carries a delta",
      'if self.derotate == "gyro" and gd_r is not None and gd_p is not None:' in src)
check("the FC remains the fallback, not an equal option",
      "d = d - attitude_shift_px(self.prev_att, att)" in src
      and src.index("d = d - gyro_shift_px(gd_r, gd_p)")
      < src.index("d = d - attitude_shift_px(self.prev_att, att)"))
check("both are SUBTRACTED, so the sign lives in the constant",
      "d = d - gyro_shift_px(gd_r, gd_p)" in src,
      "a negative coefficient is the inversion, applied in one place")
check("which source ran is counted",
      "self.n_derot_gyro += 1" in src and "self.n_derot_fc += 1" in src)
check("and reported at the end of a run",
      "frames from the camera gyro" in src,
      "a flight should say which path it took, not be assumed")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
