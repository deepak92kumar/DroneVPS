"""A tilt source is used only after it has earned it, and with the sign it earned.

21 Oct is why this exists. Derotation was switched to the camera gyro on a
static measurement - gyro against the FC's attitude over matched intervals,
r = -0.994, scale -0.99 - and the sign that implied was hard-coded. Flown, the
reported velocity came out 160 degrees from the truth, east r = -0.884, and in
Loiter the aircraft ran away: told it was moving one way while it drifted the
other, the controller pushed it further. Log 00000223, two VPS windows,
position error growing 0.5 -> 21.4 m and 18.2 -> 43.3 m while the aircraft
really moved 10.6 and 22.3 m.

The pre-SSD-loss daemon never had that hole. aruco_flow_rpi/flight_daemon.py:

    "Compare ~0.4 s windows of gyro vs rotation-vector tilt. Gyro derotation
     turns ON only when both axes agree (>=15 moving windows, |corr|>=0.85,
     0.7<=|slope|<=2.0) and OFF if agreement is lost."

and it DERIVED the sign instead of assuming it:

    gyro_derot.update(ok=True, sign_r=sign(verdict[0][0]),
                               sign_p=sign(verdict[1][0]))

Thresholds here are that daemon's, unchanged. The reference is the FC's
attitude, because it is the only tilt source measured to track the real motion
on this airframe - velocity slope +0.795 north and +0.892 east against the raw
receiver, 33 degrees from the truth, 73% of samples in the right half-plane.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "vps_vio"))

from flow_odom import TiltValidator, gyro_shift_px  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


DT = 1.0 / 30.0
rng = np.random.default_rng(21)


def feed(v, n, scale_r, scale_p, noise=0.0, ref=None):
    """n frames where the candidate is scale * the reference, plus noise."""
    for i in range(n):
        r = rng.normal(0, 0.35) if ref is None else ref[i]
        p = rng.normal(0, 0.35)
        v.update(scale_r * r + rng.normal(0, noise),
                 scale_p * p + rng.normal(0, noise), r, p, DT)


# ------------------------------------------------- a clean source is adopted

print("a source that agrees is adopted, and its sign is LEARNED not assumed")
v = TiltValidator("same sign")
feed(v, 900, +1.0, +1.0, noise=0.01)
check("it turned on", v.ok, v.describe())
check("and learned a POSITIVE sign", v.sign_roll == 1.0 and v.sign_pitch == 1.0)

vi = TiltValidator("inverted")
feed(vi, 900, -1.0, -1.0, noise=0.01)
check("an inverted source is also adopted", vi.ok, vi.describe())
check("but with a NEGATIVE sign - which is the whole point",
      vi.sign_roll == -1.0 and vi.sign_pitch == -1.0,
      "21 Oct hard-coded this and got it backwards")

# ------------------------------------------------- the gate itself

print("\nwhat it refuses")
vn = TiltValidator("noise")
for _ in range(900):
    vn.update(rng.normal(0, 0.35), rng.normal(0, 0.35),
              rng.normal(0, 0.35), rng.normal(0, 0.35), DT)
check("a source uncorrelated with the reference never turns on", not vn.ok,
      vn.describe())

vs = TiltValidator("wrong scale")
feed(vs, 900, +0.2, +0.2, noise=0.01)
check("a source at 0.2x the reference is refused", not vs.ok,
      f"slope must be 0.7-2.0; {vs.describe()}")

vb = TiltValidator("too big")
feed(vb, 900, +4.0, +4.0, noise=0.01)
check("a source at 4x is refused too", not vb.ok, vb.describe())

vm = TiltValidator("mixed axes")
feed(vm, 900, +1.0, -1.0, noise=0.01)
check("one axis positive and one negative is still adopted, per axis",
      vm.ok and vm.sign_roll == 1.0 and vm.sign_pitch == -1.0,
      "the two axes are judged separately, as the daemon did")

print("\nand it refuses to decide on too little")
# 12 windows: 150 frames at 30 Hz is 5 s, and the window is 0.4 s.
ve = TiltValidator("short")
feed(ve, 150, +1.0, +1.0, noise=0.01)
check("under the window count it stays off", not ve.ok,
      f"{ve.n_windows} windows, needs {TiltValidator.MIN_WINDOWS} MOVING ones")
check("one more window than the minimum IS enough, so the bar is real",
      (lambda v: (feed(v, 200, +1.0, +1.0, noise=0.01), v.ok)[1])(
          TiltValidator("just enough")),
      "200 frames is 16 windows")

vz = TiltValidator("still")
for _ in range(900):
    vz.update(0.0, 0.0, 0.0, 0.0, DT)
check("a stationary aircraft teaches it nothing", not vz.ok,
      "windows with no rotation in them are dropped")

# ------------------------------------------------- it lets go again

print("\na source that loses agreement is dropped MID-FLIGHT")
vd = TiltValidator("goes bad")
feed(vd, 900, +1.0, +1.0, noise=0.01)
check("on at first", vd.ok)
feed(vd, 900, -1.0, -1.0, noise=0.01)       # the sign flips under it
check("and off once the sign flips under it", not vd.ok, vd.describe())
check("the transition was counted", vd.turned_on >= 1 and vd.turned_off >= 1,
      f"on {vd.turned_on}, off {vd.turned_off}")

print("\nON is harder than OFF, so a source does not flicker")
check("arm at |corr| 0.85", TiltValidator.ON_CORR == 0.85)
check("stay at |corr| 0.60", TiltValidator.OFF_CORR == 0.60)
check("and the window is the daemon's 0.4 s", TiltValidator.WINDOW_S == 0.4)
check("needing 15 moving windows", TiltValidator.MIN_WINDOWS == 15)

# ------------------------------------------------- the shift carries the sign

print("\nthe sign reaches the pixels")
a = gyro_shift_px(1.0, 0.0, +1.0, +1.0)
b = gyro_shift_px(1.0, 0.0, -1.0, +1.0)
check("the same rotation moves opposite ways under opposite signs",
      a[0] * b[0] < 0, f"{a[0]:+.2f} vs {b[0]:+.2f} px")
check("the magnitude is the report's 14.80 either way",
      abs(abs(a[0]) - 14.80) < 1e-9 and abs(abs(b[0]) - 14.80) < 1e-9,
      "3 Oct, 15 flights, 39723 samples")
# The constants carry their own sign from 23 Oct, so "+1" now means "as
# measured" (negative) rather than "positive". The property being pinned is
# unchanged: the two axes do not contaminate each other.
check("pitch is independent of roll's sign",
      abs(gyro_shift_px(0.0, 1.0, -1.0, +1.0)[1]
          - gyro_shift_px(0.0, 1.0, +1.0, +1.0)[1]) < 1e-9,
      "flipping roll's sign must not move pitch")

# ------------------------------------------------- the wiring

print("\nthe wiring")
src = open(os.path.join(HERE, "vps_vio", "live_node.py")).read()
check("both candidates are judged against the FC",
      "self.val_gyro.update(gd_r, gd_p, fc_r, fc_p, dt_v)" in src
      and "self.val_quat.update(qd_r, qd_p, fc_r, fc_p, dt_v)" in src)
# 23 OCT, AT THE PILOT'S INSTRUCTION: the validator no longer withholds the
# gyro. It could never adopt when it mattered - it needs fifteen 0.4 s windows
# with real rotation in them, so on the ground it never adopts at all
# (measured: vps_00130, 1139 of 1142 frames on 'fc' sitting still) and the
# first forty seconds of every flight ran on the FC path, which delivers
# 0.000 deg on 88% of frames and measured worse than no derotation.
#
# It still RUNS and still reports, which is what the next two checks pin. What
# it no longer does is gate.
check("the validator still judges the gyro against the FC",
      "self.val_gyro.update(gd_r, gd_p, fc_r, fc_p, dt_v)" in src)
check("but it no longer gates the gyro",
      "self.val_gyro.ok\n" not in src.replace("self.val_gyro.ok ", "self.val_gyro.ok\n")
      or "and gd_r is not None and gd_p is not None):" in src,
      "the gyro is used whenever it reported; the FC is the fallback for when "
      "it did not")
check("GAME_ROTATION_VECTOR sits between them",
      "gyro_shift_px(qd_r, qd_p)" in src)
check("the FC is the fallback", "d = d - attitude_shift_px(self.prev_att, att)" in src)
check("there is NO way to force an unverified source",
      'choices=("auto", "fc")' in src,
      "the 'gyro' option is gone - forcing it is the 21 Oct mistake")
check("and the verdict is printed at the end of a run",
      "node.val_gyro.describe()" in src)

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
