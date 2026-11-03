#!/usr/bin/env python3
"""Cross-check the VPS pipeline against ITSELF, stage by stage.

Everything else in this repo scores one of our outputs against the receiver.
That answers "is it right" and cannot answer "where does it first go wrong",
because a disagreement with the world could enter at any of eight stages.

These checks need no external truth. Each one takes two things the node
produced from the same frames and asks whether they can both be true. Where
they cannot, the fault is upstream of the second one, and that is a place to
look rather than a number to argue about.

    camera   step_dn/step_de        per-frame ground displacement
    flow     vel_flow_vn/ve         velocity from the N-frame baseline
    bridge   vel_vn/ve              after the accelerometer blend
    sent     vel_sent_vn/ve         after deadband, slew and cap
    position north_m/east_m         origin + integrated displacement + map

Checks:

  1. POSITION vs STEPS. north_m should be the origin plus the sum of
     step_dn, plus whatever the map moved. Any gap is the map's doing, and
     the size of it says how hard the map is pulling.

  2. VELOCITY vs POSITION. Differentiating north_m over the same window the
     velocity baseline uses should give vel_flow_vn back. These are two
     views of one measurement; if they disagree, the flight controller is
     being handed a position and a velocity that contradict each other.
     memory/dronevps-height-is-the-wall records corr 0.374 here and nobody
     ever chased it.

  3. SENT vs BRIDGE. What leaves should be the bridge's value, unchanged
     except where the deadband, the slew or the cap say otherwise.

  4. ALTITUDE. The number that turns pixels into metres, against the
     rangefinder when there is one.

    python3 audit.py vps_00238
"""
import csv
import sys

import numpy as np


def col(rows, k):
    out = []
    for x in rows:
        try:
            out.append(float(x[k]))
        except Exception:
            out.append(np.nan)
    return np.array(out)


def pr(label, value, note=""):
    print(f"    {label:<34} {value:>12}  {note}")


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        print(__doc__)
        return 1
    run = argv[0]
    r = list(csv.DictReader(open(f"/data/DRONE/flight_logs/vps/{run}/{run}.csv")))
    t, alt = col(r, "t"), col(r, "alt_m")
    dn, de, ddt = col(r, "step_dn"), col(r, "step_de"), col(r, "step_dt_s")
    n_, e_ = col(r, "north_m"), col(r, "east_m")
    fvn, fve = col(r, "vel_flow_vn"), col(r, "vel_flow_ve")
    bvn, bve = col(r, "vel_vn"), col(r, "vel_ve")
    svn, sve = col(r, "vel_sent_vn"), col(r, "vel_sent_ve")
    db = col(r, "vel_deadbanded")
    rng = col(r, "rngfnd_m")
    src = np.array([x.get("src", "") for x in r])
    fx = np.array([bool((x.get("fix_lon") or "").strip()) for x in r])
    vps = src == "VPS"
    print(f"=== {run}   {len(r)} rows, VPS {int(vps.sum())}, "
          f"alt max {np.nanmax(alt):.1f} m, map fixes {int(fx.sum())}")

    print("\n  1. POSITION vs the steps it is built from")
    good = np.isfinite(dn) & np.isfinite(de) & vps
    if good.sum() > 200:
        i = np.flatnonzero(good)
        walked_n = np.nancumsum(dn[i])
        walked_e = np.nancumsum(de[i])
        rep_n = n_[i] - n_[i][0]
        rep_e = e_[i] - e_[i][0]
        gap = np.hypot(rep_n - walked_n, rep_e - walked_e)
        # Read the END gap with care: step_dn is logged to five decimals and
        # the position to fewer, so a few millimetres of rounding per row
        # random-walks into tens of centimetres over thousands of them. On
        # vps_00253, which took no fixes at all, this read 0.63 m while the
        # PER-ROW difference was p50 0.00000 m with not one row above a
        # centimetre. The per-row figure is the one that means anything.
        pr("flow alone walked", f"{np.hypot(walked_n[-1], walked_e[-1]):.1f} m")
        pr("the node reports having moved", f"{np.hypot(rep_n[-1], rep_e[-1]):.1f} m")
        pr("gap at the end (the map's doing)", f"{gap[-1]:.1f} m",
           f"over {int(fx[i].sum())} fixes")
        pr("biggest gap at any moment", f"{np.nanmax(gap):.1f} m")
        dpos = np.diff(rep_n)
        dstep = dn[i][1:]
        k = np.isfinite(dpos) & np.isfinite(dstep)
        if k.sum() > 100:
            d = dpos[k] - dstep[k]
            pr("per-row, position move minus step", f"{np.median(d):+.5f} m",
               f"sd {d.std():.5f}, {int((np.abs(d)>0.01).sum())} rows over 1 cm")
    else:
        pr("step_dn/step_de", "absent", "this flight predates those columns")

    print("\n  2. VELOCITY vs POSITION - two views of one measurement")
    for W, lbl in ((1.33, "the 40-frame baseline"), (0.5, "half a second")):
        vn = np.full(len(t), np.nan)
        for k in range(len(t)):
            j = np.searchsorted(t, t[k] - W)
            if k - j < 3 or t[k] - t[j] <= 0:
                continue
            vn[k] = (n_[k] - n_[j]) / (t[k] - t[j])
        m = vps & np.isfinite(vn) & np.isfinite(fvn)
        if m.sum() < 200:
            continue
        cr = np.corrcoef(vn[m], fvn[m])[0, 1]
        sl = np.polyfit(vn[m], fvn[m], 1)[0]
        pr(f"over {lbl}", f"corr {cr:+.3f}", f"slope {sl:.2f}, n {int(m.sum())}")
    print("      (position and velocity come from the SAME steps, so a low")
    print("       correlation here is the node contradicting itself)")

    print("\n  3. SENT vs the bridge's own value")
    m = vps & np.isfinite(bvn) & np.isfinite(svn)
    if m.sum() > 100:
        zero = np.abs(svn[m]) < 1e-6
        same = np.isclose(svn[m], bvn[m], atol=1e-3) & ~zero
        other = ~(same | zero)
        pr("zeroed", f"{100*zero.mean():.1f}%",
           f"deadband flag says {100*np.nanmean(db[m]):.1f}%")
        pr("passed through unchanged", f"{100*same.mean():.1f}%")
        pr("changed but not zeroed", f"{100*other.mean():.1f}%",
           "slew or cap")
        # The three are exclusive on purpose. They overlapped when this was
        # first written - "unchanged" counted every row where both were zero -
        # and the three lines summed to 143%, which is how I noticed.

    print("\n  4. ALTITUDE - what turns pixels into metres")
    m = vps & np.isfinite(rng) & (rng > 0.5) & (rng < 60) & (alt > 2)
    if m.sum() > 100:
        k, c = np.polyfit(rng[m], alt[m], 1)
        pr("alt_m against the rangefinder", f"{k:.3f}x {c:+.2f} m",
           f"n {int(m.sum())}")
    else:
        pr("rangefinder", "not usable", "off, or out of range all flight")
    return 0


if __name__ == "__main__":
    sys.exit(main())
