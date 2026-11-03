#!/usr/bin/env python3
"""Score a flight's velocity as VECTORS, by altitude band.

This exists because scoring it as a MAGNITUDE hid the fault that mattered.
On 27 Oct the sent speed matched the receiver to 0.05-0.29 m/s at every
height and the aircraft still wandered, because the magnitude was right and
the DIRECTION was 25 degrees wrong. memory/dronevps-measure-fairly.md was
written about exactly this and I did it again anyway.

So: angle between our vector and the receiver's, the per-axis slope, and the
correlation - never |v|. Both sides are differenced over the same 1 s window,
because comparing a raw signal against a smoothed one is the other way this
project has fooled itself.

Truth is gps1_north_m / gps1_east_m, the raw receiver, filtered to within
400 m of the origin. Never gps_north_m, which is the EKF's fused output and
on VPS is our own injection coming back.

Columns, per band:

    accel   vel_accel_n/e   the camera IMU, rotated into NED
    flow    vel_flow_vn/ve  the optical flow before the blend
    blend   vel_vn/ve       after the accel blend
    sent    vel_sent_vn/ve  what the flight controller was given

    python3 vel_vectors.py vps_00238
"""
import csv
import math
import sys

import numpy as np


def load(run):
    p = f"/data/DRONE/flight_logs/vps/{run}/{run}.csv"
    return list(csv.DictReader(open(p)))


def col(rows, k):
    out = []
    for x in rows:
        try:
            out.append(float(x[k]))
        except Exception:
            out.append(np.nan)
    return np.array(out)


def truth_series(t, gn, ge, window=1.0):
    """The receiver's velocity VECTOR, differenced over `window` seconds."""
    ok = np.isfinite(gn) & np.isfinite(ge) & (np.abs(gn) < 400) & (np.abs(ge) < 400)
    ii = np.flatnonzero(ok)
    tt, nn, ee = t[ii], gn[ii], ge[ii]
    tn, te, ts = [], [], []
    for i in range(len(ii)):
        j = np.searchsorted(tt, tt[i] - window)
        if i - j < 3 or tt[i] - tt[j] <= 0:
            continue
        dt = tt[i] - tt[j]
        tn.append((nn[i] - nn[j]) / dt)
        te.append((ee[i] - ee[j]) / dt)
        ts.append(tt[i])
    return np.array(ts), np.array(tn), np.array(te)


def score(t, ON, OE, m, ts, tn, te, floor=0.15):
    """Median angle, per-axis slope, correlation. None when too little motion."""
    if m.sum() < 40:
        return None
    TN, TE = np.interp(t[m], ts, tn), np.interp(t[m], ts, te)
    on, oe = ON[m], OE[m]
    mo, mt = np.hypot(on, oe), np.hypot(TN, TE)
    g = (mo > floor) & (mt > floor)
    if g.sum() < 20:
        return None
    dot = (on[g] * TN[g] + oe[g] * TE[g]) / (mo[g] * mt[g])
    ang = np.degrees(np.arccos(np.clip(dot, -1, 1)))
    sl = np.polyfit(TN[g], on[g], 1)[0]
    cr = np.corrcoef(TN[g], on[g])[0, 1]
    return float(np.median(ang)), float(sl), float(cr), int(g.sum())


BANDS = ((5, 20), (20, 35), (35, 50), (50, 70), (70, 95))


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        print(__doc__)
        return 1
    run = argv[0]
    r = load(run)
    t, alt = col(r, "t"), col(r, "alt_m")
    src = np.array([x.get("src", "") for x in r])
    vps = src == "VPS"
    fx = np.array([bool((x.get("fix_lon") or "").strip()) for x in r])
    ts, tn, te = truth_series(t, col(r, "gps1_north_m"), col(r, "gps1_east_m"))
    stages = (("accel", "vel_accel_n", "vel_accel_e"),
              ("flow", "vel_flow_vn", "vel_flow_ve"),
              ("blend", "vel_vn", "vel_ve"),
              ("sent", "vel_sent_vn", "vel_sent_ve"))
    print(f"=== {run}   {len(r)} rows, VPS {int(vps.sum())}, "
          f"alt max {np.nanmax(alt):.1f} m, map fixes {int(fx.sum())}")
    print(f"\n  {'band':>9} " + "".join(f"{s[0]:>19}" for s in stages))
    print(f"  {'':>9} " + "".join(f"{'ang slope corr':>19}" for _ in stages))
    for lo, hi in BANDS:
        m = vps & (alt >= lo) & (alt < hi)
        line = f"  {lo:4d}-{hi:3d} "
        hit = False
        for _, kn, ke in stages:
            s = score(t, col(r, kn), col(r, ke), m, ts, tn, te)
            if s:
                line += f"{s[0]:7.0f} {s[1]:5.2f} {s[2]:5.2f}"
                hit = True
            else:
                line += f"{'--':>19}"
        if hit:
            print(line)
    # position, and how hard the aircraft was really working
    n_, e_ = col(r, "north_m"), col(r, "east_m")
    gn, ge = col(r, "gps1_north_m"), col(r, "gps1_east_m")
    ok = vps & np.isfinite(gn) & np.isfinite(ge) & (np.abs(gn) < 400) & (np.abs(ge) < 400)
    err = np.hypot(n_ - gn, e_ - ge)
    spd = np.interp(t, ts, np.hypot(tn, te))
    print(f"\n  {'band':>9} {'n':>6} {'pos p50':>8} {'p90':>8} {'fixes':>6} "
          f"{'drift m/s':>10} {'real spd p50':>13} {'p90':>6}")
    for lo, hi in BANDS:
        m = ok & (alt >= lo) & (alt < hi)
        if m.sum() < 40:
            continue
        i = np.flatnonzero(m)
        rate = np.polyfit(t[i] - t[i][0], err[i], 1)[0]
        print(f"  {lo:4d}-{hi:3d} {len(i):6d} {np.percentile(err[i], 50):8.2f} "
              f"{np.percentile(err[i], 90):8.2f} {int(fx[m].sum()):6d} {rate:+10.3f} "
              f"{np.nanmedian(spd[m]):13.2f} {np.nanpercentile(spd[m], 90):6.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
