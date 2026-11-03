#!/usr/bin/env python3
"""Evidence for (or against) applying the mount yaw offset to accel_to_ne.

accel_to_ne's own docstring says "Camera-frame acceleration -> (north, east)"
and it rotates by the FULL FC attitude. That is correct only while the camera
is aligned with the body. The mount is not: +4.42 deg, measured three ways.

Same situation shift_to_ne was in, and shift_to_ne was corrected on 2 Nov.
This checks the correction on real recorded IMU samples before any flight
code is touched.
"""
import csv, math, sys
import numpy as np
sys.path.insert(0, '/data/DRONE/vps_vio')
from live_node import accel_to_ne

OFF = -4.42

print("1) zero offset par output bilkul wahi hona chahiye")
rng = np.random.default_rng(3)
same = True
for _ in range(2000):
    ax, ay, az = rng.normal(0, 3, 3)
    yaw, r, p = rng.uniform(-180, 180), rng.uniform(-25, 25), rng.uniform(-25, 25)
    a = accel_to_ne(ax, ay, yaw, az, r, p)
    b = accel_to_ne(ax, ay, yaw + 0.0, az, r, p)
    if abs(a[0] - b[0]) > 1e-12 or abs(a[1] - b[1]) > 1e-12:
        same = False
print(f"   2000 random samples: {'BILKUL BARABAR' if same else 'ALAG - GADBAD'}")

print("\n2) 0/90/180/270 par, +ve aur -ve offset (unit vector forward = camera +x)")
print("   %7s %9s %9s %9s %9s %9s" % ("heading", "N(off 0)", "E(off 0)",
                                      "N(-4.42)", "E(-4.42)", "ghuma"))
for yaw in (0.0, 90.0, 180.0, 270.0):
    n0, e0 = accel_to_ne(1.0, 0.0, yaw, 0.0, 0.0, 0.0)
    n1, e1 = accel_to_ne(1.0, 0.0, yaw + OFF, 0.0, 0.0, 0.0)
    d = math.degrees(math.atan2(e1, n1) - math.atan2(e0, n0))
    d = (d + 180) % 360 - 180
    print("   %7.0f %9.4f %9.4f %9.4f %9.4f %+9.3f" % (yaw, n0, e0, n1, e1, d))
for yaw in (0.0, 90.0):
    n1, e1 = accel_to_ne(1.0, 0.0, yaw - OFF, 0.0, 0.0, 0.0)
    n0, e0 = accel_to_ne(1.0, 0.0, yaw, 0.0, 0.0, 0.0)
    d = (math.degrees(math.atan2(e1, n1) - math.atan2(e0, n0)) + 180) % 360 - 180
    print("   %7.0f  +4.42 offset -> ghuma %+.3f (ulta hona chahiye)" % (yaw, d))

print("\n3) asli recorded IMU samples par: old vs corrected")
for run in ('vps_00418', 'vps_00406'):
    try:
        I = [r for r in csv.DictReader(
            open(f'/data/DRONE/flight_logs/vps/{run}/{run}_imu.csv'))
            if r['src'] == 'a']
        M = list(csv.DictReader(
            open(f'/data/DRONE/flight_logs/vps/{run}/{run}.csv')))
    except Exception as e:
        print(f"   {run}: {e}")
        continue
    if len(I) < 1000:
        continue
    t = np.array([float(r['t_dev']) for r in I])
    A = np.array([[float(r['x']), float(r['y']), float(r['z'])] for r in I])
    # attitude per sample, from the estimate CSV
    mt, my, mr, mp = [], [], [], []
    for r in M:
        try:
            mt.append(float(r['t'])); my.append(float(r['yaw_deg']))
            mr.append(float(r['fc_d_roll'] or 0)); mp.append(float(r['fc_d_pitch'] or 0))
        except (TypeError, ValueError):
            pass
    if len(mt) < 50:
        continue
    t0 = t - t[0]
    mt = np.array(mt) - mt[0]
    yaw = np.interp(t0, mt, my)
    dt = np.diff(t0, prepend=t0[0])
    dt = np.clip(dt, 0, 0.05)
    out = {}
    for off in (0.0, OFF):
        N = np.array([accel_to_ne(A[i, 0], A[i, 1], yaw[i] + off, A[i, 2],
                                  0.0, 0.0) for i in range(len(A))])
        vn = np.cumsum(N[:, 0] * dt); ve = np.cumsum(N[:, 1] * dt)
        pn = np.cumsum(vn * dt); pe = np.cumsum(ve * dt)
        out[off] = (N, vn, ve, pn, pe)
    a0, a1 = out[0.0], out[OFF]
    da = np.hypot(a1[0][:, 0] - a0[0][:, 0], a1[0][:, 1] - a0[0][:, 1])
    dv = math.hypot(a1[1][-1] - a0[1][-1], a1[2][-1] - a0[2][-1])
    dp = math.hypot(a1[3][-1] - a0[3][-1], a1[4][-1] - a0[4][-1])
    print(f"   {run}: {len(A)} accel samples, {t0[-1]:.0f} s")
    print(f"      per-sample accel ka fark : p50 {np.median(da):.4f}  "
          f"max {da.max():.3f} m/s2")
    print(f"      integrate ki hui velocity ka fark : {dv:.3f} m/s")
    print(f"      integrate ki hui position ka fark : {dp:.1f} m")
