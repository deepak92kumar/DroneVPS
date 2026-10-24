"""The gyro is integrated over the FRAME's interval, not the drain's.

Flight 240 measured what the drain window cost. Fitting the per-frame pixel
displacement against the rotation derotation had just removed gives
step_px = 0.157 x rotation_px + 1.48 - 16% of the rotation survived, a 0.64 px
residual that is the same at every height and becomes 0.75 m/s at 38 m and
2.19 m/s at 110 m. That is the hover phantom, and corr(rotation, pixel error)
was +0.870 while the pixel error itself was flat across altitude.

cos(angle between the residual and the correction) was +0.07 and -0.005 -
near zero - so it is not a sign or gain error in GYRO_PX_PER_DEG, which would
leave the residual pointing along the correction. A window offset leaves the
right magnitude in a scattered direction, which is what was measured.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vps_vio"))

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        fails.append(name)


def integrate(buf, w0, w1):
    """The same overlap weighting live_node applies, isolated for testing."""
    wr = wp = win = 0.0
    for (a, b, r, p) in buf:
        ov = min(b, w1) - max(a, w0)
        if ov > 0.0:
            wr += r * ov
            wp += p * ov
            win += ov
    return wr, wp, win


print("a sample wholly inside the window contributes all of itself")
# one sample, 10 deg/s for 10 ms, window covers it entirely
wr, wp, win = integrate([(1.00, 1.01, 10.0, 0.0)], 0.99, 1.02)
check("the whole interval is counted", abs(win - 0.01) < 1e-9, f"{win:.4f} s")
check("and the rotation is rate x time", abs(wr - 0.1) < 1e-9, f"{wr:.4f} deg")

print("\na sample wholly outside contributes nothing")
wr, wp, win = integrate([(2.00, 2.01, 10.0, 0.0)], 0.99, 1.02)
check("nothing is counted", win == 0.0 and wr == 0.0)

print("\na sample straddling the start contributes only its overlap")
# 10 deg/s from 0.995 to 1.005; the window starts at 1.000, so half of it
wr, wp, win = integrate([(0.995, 1.005, 10.0, 0.0)], 1.000, 1.020)
check("half the interval", abs(win - 0.005) < 1e-9, f"{win:.4f} s")
check("half the rotation", abs(wr - 0.05) < 1e-9,
      f"{wr:.4f} deg - all-or-nothing would give 0.10 or 0.00")

print("\na sample straddling the end contributes only its overlap")
wr, wp, win = integrate([(1.015, 1.025, 10.0, 0.0)], 1.000, 1.020)
check("half the interval", abs(win - 0.005) < 1e-9, f"{win:.4f} s")

print("\nthe window is covered exactly once by back-to-back samples")
buf = [(1.000 + 0.006 * i, 1.000 + 0.006 * (i + 1), 10.0, -4.0) for i in range(10)]
wr, wp, win = integrate(buf, 1.000, 1.040)      # a 40 ms frame at 25 fps
check("the window length is the frame interval", abs(win - 0.040) < 1e-9,
      f"{win:.4f} s against 0.0400")
check("roll is the rate over that interval", abs(wr - 0.4) < 1e-9, f"{wr:.4f} deg")
check("pitch carries its own sign", abs(wp + 0.16) < 1e-9, f"{wp:.4f} deg")

print("\nthe two frames of a skipped-frame gap do not double-count")
# frame A ends at 1.040, frame B at 1.080 - a skipped frame doubles B's span
a = integrate(buf + [(1.000 + 0.006 * i, 1.000 + 0.006 * (i + 1), 10.0, 0.0)
                     for i in range(10, 20)], 1.000, 1.040)
b = integrate(buf + [(1.000 + 0.006 * i, 1.000 + 0.006 * (i + 1), 10.0, 0.0)
                     for i in range(10, 20)], 1.040, 1.080)
check("each window gets its own span", abs(a[2] - 0.040) < 1e-9 and abs(b[2] - 0.040) < 1e-9,
      f"{a[2]:.4f} + {b[2]:.4f}")
check("and the total is the two frames, not three",
      abs((a[0] + b[0]) - 0.8) < 1e-9, f"{a[0] + b[0]:.4f} deg over 80 ms at 10 deg/s")

print("\nthe node wires it the way this models it")
src = open(os.path.join(os.path.dirname(__file__), "..",
                        "vps_vio", "live_node.py")).read()
check("the window is bounded by the previous FRAME, not the drain",
      "if self._ts_prev_frame is not None and ts > self._ts_prev_frame:" in src)
check("samples are weighted by overlap, not all-or-nothing",
      "ov = min(b, w1) - max(a, w0)" in src)
check("a sample reaching past the window is KEPT for the next frame",
      "self._gyro_buf = [g for g in self._gyro_buf if g[1] > w1]" in src,
      "dropping it would lose rotation at every frame boundary")
check("the previous frame timestamp advances every frame",
      "self._ts_prev_frame = ts" in src)
check("the old drain integral is still logged for comparison",
      '"gyro_drain_roll"' in src and '"gyro_win_s"' in src)

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
