#!/usr/bin/env python3
"""Is the aircraft fighting a yaw it cannot win, and is the compass why?

The pilot reports two things that belong together: with the VPS cable out the
aircraft hovered on GPS, with it in it drifted, and three of the six motors run
hotter than the other three.

On a hexacopter yaw torque is the difference between the three clockwise props
and the three counter-clockwise ones. Holding a heading against a steady yaw
error means running one group faster than the other for the whole flight, and a
motor that runs faster gets hotter. So a persistent split in RCOU between the
two groups IS a persistent yaw demand, and the heat is its receipt.

The usual cause of a steady yaw error is the compass, and the usual cause of a
compass error is current in a wire near it. COMPASS_MOT_X/Y/Z are all zero on
this aircraft, so the compensation that exists for exactly this has never been
run.

    python3 yaw_and_motors.py /mnt/fccard/APM/LOGS/00000011.BIN
"""
import sys

import numpy as np
from pymavlink import mavutil


def main(path):
    m = mavutil.mavlink_connection(path)
    rcou, mag, att, ctun, vibe, gps = [], [], [], [], [], []
    t0 = None
    while True:
        x = m.recv_match(type=["RCOU", "MAG", "ATT", "CTUN", "VIBE", "GPS"])
        if x is None:
            break
        t = getattr(x, "TimeUS", 0) / 1e6
        if t0 is None:
            t0 = t
        r, ty = t - t0, x.get_type()
        if ty == "RCOU":
            rcou.append([r] + [getattr(x, f"C{i}", np.nan) for i in range(1, 7)])
        elif ty == "MAG":
            mag.append((r, x.MagX, x.MagY, x.MagZ, getattr(x, "I", 0)))
        elif ty == "ATT":
            att.append((r, x.Yaw, x.DesYaw))
        elif ty == "CTUN":
            ctun.append((r, x.Alt, getattr(x, "ThO", np.nan)))
        elif ty == "VIBE":
            vibe.append((x.VibeX, x.VibeY, x.VibeZ))
        elif ty == "GPS":
            gps.append((r, getattr(x, "GCrs", np.nan), getattr(x, "Spd", np.nan)))

    R, C = np.array(rcou), np.array(ctun)
    print(f"=== {path.split('/')[-1]}   {R[-1,0]:.0f} s")

    # Airborne by the MOTORS, not by the barometer. CTUN.Alt is relative to
    # whatever the baro was zeroed at, and after a hard reset that reference
    # has been seen sitting half a metre below the ground - which hid a real
    # flight behind "alt max 0.44 m". Motors above 1200 us are carrying the
    # aircraft whatever the barometer believes.
    spin = R[:, 1:].max(axis=1) > 1200
    if spin.sum() < 50:
        print("  motor kabhi nahi chale - yeh flight nahi hai\n")
        return
    si = np.flatnonzero(spin)
    lo, hi = R[si[0], 0], R[si[-1], 0]
    air_t = C[(C[:, 0] >= lo) & (C[:, 0] <= hi), 1]
    if air_t.size:
        print(f"  (us doran alt {air_t.min():.2f} .. {air_t.max():.2f} m)")
    a = (R[:, 0] >= lo) & (R[:, 0] <= hi)
    print(f"  hawa me {lo:.0f} - {hi:.0f} s  ({a.sum()} samples)\n")

    print("  MOTOR OUTPUT, hawa me (PWM)")
    means = []
    for i in range(1, 7):
        v = R[a, i]
        v = v[np.isfinite(v)]
        means.append(v.mean() if v.size else np.nan)
        print(f"     motor {i}   p50 {np.median(v):7.1f}   mean {v.mean():7.1f}"
              f"   sd {v.std():5.1f}")
    means = np.array(means)
    # The two groups alternate around a hexa, so odd and even motor numbers are
    # the two rotation directions whichever way round the frame is wired.
    odd, even = means[0::2], means[1::2]
    print(f"\n     motors 1,3,5   mean {odd.mean():7.1f}")
    print(f"     motors 2,4,6   mean {even.mean():7.1f}")
    d = odd.mean() - even.mean()
    print(f"     antar          {d:+7.1f} PWM")
    print("     (ek group dusre se lagataar tez = lagataar yaw torque.")
    print("      ~20 PWM se jyada antar ek asli yaw demand hai.)")

    if att:
        A = np.array(att)
        aa = (A[:, 0] >= lo) & (A[:, 0] <= hi)
        err = (A[aa, 1] - A[aa, 2] + 180) % 360 - 180
        print(f"\n  YAW  - DesYaw   p50 {np.median(err):+.2f}  sd {err.std():.2f} deg")
        print(f"         heading  {A[aa,1].min():.0f} .. {A[aa,1].max():.0f} deg")

    if mag:
        M = np.array(mag)
        for inst in sorted(set(M[:, 4].astype(int))):
            s = (M[:, 4] == inst) & (M[:, 0] >= lo) & (M[:, 0] <= hi)
            g = (M[:, 4] == inst) & (M[:, 0] < lo)
            if s.sum() < 20:
                continue
            fa = np.linalg.norm(M[s, 1:4], axis=1)
            print(f"\n  COMPASS {inst}  field strength, hawa me")
            print(f"     p50 {np.median(fa):7.1f}  sd {fa.std():6.1f}  "
                  f"min {fa.min():7.1f}  max {fa.max():7.1f} mGauss")
            if g.sum() > 20:
                fg = np.linalg.norm(M[g, 1:4], axis=1)
                print(f"     zameen par (motor band) p50 {np.median(fg):7.1f}")
                print(f"     ANTAR {np.median(fa)-np.median(fg):+7.1f} mGauss"
                      "   <- motor ka current compass ko kitna hila raha hai")

    if vibe:
        V = np.array(vibe)
        print(f"\n  vibration  p50 {np.percentile(V,50,axis=0)}  "
              f"p95 {np.percentile(V,95,axis=0)}")
    print()


if __name__ == "__main__":
    for p in sys.argv[1:] or ["/mnt/fccard/APM/LOGS/00000011.BIN"]:
        main(p)
