#!/usr/bin/env python3
"""Split a bench log at the ch6 switch flips and score the receiver in each part.

The question is whether the Jetson, or its camera's USB 3 link, raises the
noise floor the M9N has to hear through. GPS L1 is 1575.42 MHz and its
wavelength is 19 cm; the antenna sits 10-15 cm above the Jetson, which is
inside one wavelength, so the coupling is near-field and a good cable does not
settle it either way. Only a measurement does.

The aircraft stands still for the whole log, so every metre the receiver
reports is its own. The pilot marks each change by flicking ch6, which lands
in RCIN and needs no clock.

    python3 emi_test.py /mnt/fccard/APM/LOGS/00000015.BIN
"""
import sys

import numpy as np
from pymavlink import mavutil


def main(path):
    m = mavutil.mavlink_connection(path)
    gps, rc, mav = [], [], []
    t0 = None
    while True:
        x = m.recv_match(type=["GPS", "RCIN", "MAV"])
        if x is None:
            break
        t = getattr(x, "TimeUS", 0) / 1e6
        if t0 is None:
            t0 = t
        r, ty = t - t0, x.get_type()
        if ty == "GPS":
            # Status 3+ and a real latitude: the receiver emits zero rows while
            # it is between fixes and they read as a 7000 km jump.
            if int(x.Status) >= 3 and int(x.NSats) > 4 and abs(x.Lat) > 1:
                gps.append((r, x.Lat, x.Lng, int(x.NSats), x.HDop))
        elif ty == "RCIN":
            rc.append((r, x.C6))
        elif ty == "MAV":
            mav.append((r, int(getattr(x, "chan", -1)), getattr(x, "rxp", 0)))

    G, R = np.array(gps), np.array(rc)
    print(f"=== {path.split('/')[-1]}   {R[-1,0]:.0f} s,  {len(G)} GPS samples\n")

    # The marks. ch6 rests at 1045 and is flicked up; each flick is one mark.
    up = R[:, 1] > 1400
    edges = R[np.flatnonzero(np.diff(up.astype(int)) == 1) + 1, 0]
    print(f"  ch6 ke nishaan: {len(edges)}  ->  "
          + ", ".join(f"{e:.0f}s" for e in edges) + "\n")
    if len(edges) < 2:
        print("  do se kam nishaan - hisse nahi kaat sakta\n")
        return

    # A segment is the quiet stretch after one mark and before the next, with
    # five seconds trimmed off each end so the flick itself is not counted.
    bounds = list(edges) + [R[-1, 0]]
    segs = [(bounds[i] + 5, bounds[i + 1] - 5) for i in range(len(bounds) - 1)]
    names = ["STEP 1  Jetson chalu + camera",
             "STEP 2  Jetson chalu, camera nikla",
             "STEP 3  Jetson band"]

    print(f"  {'':34s} {'sats':>5} {'HDOP':>6} {'north sd':>9} "
          f"{'east sd':>8} {'p90':>7} {'10s drift p50':>14}")
    print("  " + "-" * 92)
    for i, (a, b) in enumerate(segs):
        s = G[(G[:, 0] >= a) & (G[:, 0] <= b)]
        if len(s) < 30:
            print(f"  {names[i] if i < len(names) else f'part {i+1}':34s} "
                  f"  (sirf {len(s)} samples)")
            continue
        lat0 = np.median(s[:, 1])
        n = (s[:, 1] - lat0) * 111320.0
        e = (s[:, 2] - np.median(s[:, 2])) * 111320.0 * np.cos(np.radians(lat0))
        keep = (np.abs(n) < 400) & (np.abs(e) < 400)
        n, e = n[keep], e[keep]
        r_ = np.hypot(n, e)
        # How far it wanders in ten seconds, which is the number the position
        # controller actually answers to.
        g = np.arange(s[0, 0], s[-1, 0], 0.2)
        ni = np.interp(g, s[:, 0], (s[:, 1] - lat0) * 111320.0)
        ei = np.interp(g, s[:, 0], (s[:, 2] - np.median(s[:, 2]))
                       * 111320.0 * np.cos(np.radians(lat0)))
        k = 50
        d = [np.hypot(ni[j:j + k] - ni[j], ei[j:j + k] - ei[j]).max()
             for j in range(0, len(g) - k)]
        nm = names[i] if i < len(names) else f"part {i+1}"
        print(f"  {nm:34s} {np.median(s[:,3]):5.0f} {np.median(s[:,4]):6.2f} "
              f"{n.std():9.3f} {e.std():8.3f} {np.percentile(r_,90):7.3f} "
              f"{np.median(d) if d else np.nan:14.3f}")
        # Jetson off shows up here as channel 2 going silent.
        if mav:
            M = np.array(mav)
            ch2 = M[(M[:, 1] == 2) & (M[:, 0] >= a) & (M[:, 0] <= b)]
            if len(ch2) > 1:
                print(f"  {'':34s} (Jetson link: {ch2[-1,2]-ch2[0,2]:.0f} "
                      f"packets FC ko mile)")
            else:
                print(f"  {'':34s} (Jetson link: CHUP - Jetson band tha)")
    print("\n  sab metre me. drone poore waqt khada tha, toh har metre receiver ka apna hai.\n")


if __name__ == "__main__":
    for p in sys.argv[1:] or ["/mnt/fccard/APM/LOGS/00000015.BIN"]:
        main(p)
