#!/usr/bin/env python3
"""Compare our velocity against the receiver at SHORT smoothing, and look at
how fast each one changes.

THE BLIND SPOT THIS EXISTS FOR. Every comparison in vel_vectors.py uses a 1 s
window on both sides, which is right for asking "is the direction correct" and
wrong for asking "why does the aircraft not settle". LOITER does not run at
1 Hz. If what we send carries jitter at 2-5 Hz, a 1 s comparison averages it
away completely while the position controller feels all of it and answers with
tilt - and the FC was measured commanding 2.7 deg of roll and 4.3 of pitch on
a flight where the pilot never touched the stick.

So: the same angle at several smoothing widths, the implied acceleration of
what we send against the receiver's own, and where each one keeps its energy.

    python3 vel_fast.py vps_00257
"""
import csv
import sys

import numpy as np

DT = 0.04  # the grid everything is resampled onto, 25 Hz


def col(rows, k):
    out = []
    for x in rows:
        try:
            out.append(float(x[k]))
        except Exception:
            out.append(np.nan)
    return np.array(out)


def smooth(v, w):
    k = max(3, int(w / DT))
    return np.convolve(v, np.ones(k) / k, "same")


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        print(__doc__)
        return 1
    run = argv[0]
    r = list(csv.DictReader(open(f"/data/DRONE/flight_logs/vps/{run}/{run}.csv")))
    t, alt = col(r, "t"), col(r, "alt_m")
    sn, se = col(r, "vel_sent_vn"), col(r, "vel_sent_ve")
    gn, ge = col(r, "gps1_north_m"), col(r, "gps1_east_m")
    src = np.array([x.get("src", "") for x in r])
    m = ((src == "VPS") & (alt > 4) & np.isfinite(gn) & np.isfinite(ge)
         & (np.abs(gn) < 400) & (np.abs(ge) < 400))
    i = np.flatnonzero(m)
    if len(i) < 500:
        print(f"{run}: only {len(i)} usable rows")
        return 1
    g = np.arange(t[i][0], t[i][-1], DT)
    TN, TE = np.interp(g, t[i], gn[i]), np.interp(g, t[i], ge[i])
    SN, SE = np.interp(g, t[i], sn[i]), np.interp(g, t[i], se[i])
    print(f"=== {run}   {g[-1]-g[0]:.0f}s above 4 m, alt p50 "
          f"{np.nanmedian(alt[i]):.1f} m")

    print("\n  angle between what we SEND and the receiver, by smoothing width")
    print("  (1 s is what every earlier report used)")
    for w in (0.2, 0.4, 1.0, 2.0):
        vN, vE = np.gradient(smooth(TN, w), DT), np.gradient(smooth(TE, w), DT)
        oN, oE = smooth(SN, w), smooth(SE, w)
        mo, mt = np.hypot(oN, oE), np.hypot(vN, vE)
        s = (mo > 0.1) & (mt > 0.1)
        if s.sum() < 100:
            continue
        d = (oN[s] * vN[s] + oE[s] * vE[s]) / (mo[s] * mt[s])
        ang = np.degrees(np.arccos(np.clip(d, -1, 1)))
        cr = np.corrcoef(oN[s], vN[s])[0, 1]
        print(f"    {w:4.1f}s   angle p50 {np.median(ang):5.0f}d   "
              f"corr {cr:+.3f}   n {int(s.sum())}")

    print("\n  how violently does each one change, sample to sample")
    dS = np.diff(SN) / DT
    vN = np.gradient(TN, DT)
    dT = np.diff(vN) / DT
    print(f"    what we send implies   p50 {np.median(np.abs(dS)):6.2f}  "
          f"p90 {np.percentile(np.abs(dS), 90):6.2f}  max {np.abs(dS).max():7.1f} m/s2")
    print(f"    the receiver implies   p50 {np.median(np.abs(dT)):6.2f}  "
          f"p90 {np.percentile(np.abs(dT), 90):6.2f}  max {np.abs(dT).max():7.1f} m/s2")

    print("\n  where each one keeps its energy")
    for lbl, v in (("what we send", SN - SN.mean()), ("real motion", vN - vN.mean())):
        F = np.abs(np.fft.rfft(v * np.hanning(len(v))))
        fr = np.fft.rfftfreq(len(v), DT)
        tot = F[(fr > 0.02) & (fr < 6)].sum()
        if tot <= 0:
            continue

        def band(a, b):
            return 100.0 * F[(fr >= a) & (fr < b)].sum() / tot

        print(f"    {lbl:13s} 0.02-0.3Hz {band(0.02,0.3):4.0f}%   "
              f"0.3-1Hz {band(0.3,1):4.0f}%   1-3Hz {band(1,3):4.0f}%   "
              f"3-6Hz {band(3,6):4.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
