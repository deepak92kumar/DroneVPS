#!/usr/bin/env python3
"""Calibrate the analog lidar from a flight, with no adapter and no rewiring.

WHY

The TF03 here is the 4-20 mA variant on the X7's ADC pin through 150 Ohms, so
the FC computes distance = (volts - RNGFND1_OFFSET) * RNGFND1_SCALING. Its
analog span is a setting - 180 m from the factory - and narrowing it is what
would make its noise small, but that needs a USB-TTL adapter on the sensor's
own UART. The pilot does not have one and no wiring is being touched, so the
span stays at 180 m and everything that can be fixed has to be fixed from
flight data.

WHAT THIS CAN AND CANNOT DO, STATED PLAINLY

It CAN pin down the scale and the offset exactly, and it can measure the
barometer's offset, which is the number the altitude correction exists to
remove.

It CANNOT make the sensor quieter. At a 180 m span one millivolt is 7.26 cm,
and that is set by the sensor, not by RNGFND1_SCALING. Re-fitting SCALING
moves the whole line; it does not narrow it.

HOW

Everything comes out of the FC's own .BIN, including the parameters, so there
is no live probing of the flight controller - see
memory/dronevps-dont-probe-the-fc-live.md for why that matters.

Two references, and they disagree in a useful way:

  CTUN.Alt   the EKF's barometric height. Smooth, and the thing we suspect.
  GPS.Alt    the receiver's height above the origin. Noisy and biased in
             scale, but INDEPENDENT of both the barometer and the lidar.

Fitting the lidar and the barometer against the same GPS altitude makes the
GPS's own scale error common to both, so the OFFSETS are comparable even
though the slopes are not trustworthy on their own. That is how 26 October
established lidar -0.04 m against baro +5.49 m on log 246.

    python3 lidar_fit.py /path/to/00000246.BIN
"""
import math
import sys

import numpy as np

RESISTOR_OHM = 150.0
MA_LOW, MA_HIGH = 4.0, 20.0


def load(path):
    from pymavlink import mavutil
    m = mavutil.mavlink_connection(path)
    R, C, G, P = [], [], [], {}
    t0 = None
    while True:
        x = m.recv_match(type=["RFND", "CTUN", "GPS", "PARM"])
        if x is None:
            break
        if x.get_type() == "PARM":
            if x.Name.startswith("RNGFND1_"):
                P[x.Name] = x.Value
            continue
        tu = getattr(x, "TimeUS", None)
        if tu and t0 is None:
            t0 = tu
        ts = ((tu - t0) / 1e6) if (tu and t0) else 0.0
        t = x.get_type()
        if t == "RFND" and getattr(x, "I", 0) == 0:
            R.append((ts, x.Dist))
        elif t == "CTUN":
            C.append((ts, x.Alt))
        elif t == "GPS" and getattr(x, "I", 0) == 0:
            G.append((ts, x.Alt))
    return (np.array(R, float), np.array(C, float), np.array(G, float), P)


def fit(x, y):
    """Least squares y = a*x + b, with the correlation."""
    a, b = np.polyfit(x, y, 1)
    return a, b, float(np.corrcoef(x, y)[0, 1])


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        print(__doc__)
        return 1
    R, C, G, P = load(argv[0])
    if len(R) < 100 or len(C) < 100:
        print("not enough rangefinder or CTUN rows in that log")
        return 1
    sc = P.get("RNGFND1_SCALING")
    off = P.get("RNGFND1_OFFSET")
    print(f"=== {argv[0].split('/')[-1]}")
    print(f"  the log flew with RNGFND1_SCALING {sc}  OFFSET {off}")

    alt = np.interp(R[:, 0], C[:, 0], C[:, 1])
    fly = (alt > 5) & (R[:, 1] > 1.0) & (R[:, 1] < 170)
    print(f"  {int(fly.sum())} samples above 5 m with a live reading")
    if fly.sum() < 200:
        print("  too few to fit")
        return 1

    a, b, r = fit(R[fly, 1], alt[fly])
    print(f"\n  barometer against the lidar")
    print(f"    baro = {a:.4f} * lidar {b:+.2f} m     corr {r:.4f}")
    print(f"    slope {a:.4f} means their SCALES agree to "
          f"{abs(1 - a) * 100:.1f}%")
    print(f"    offset {b:+.2f} m is the barometer's error if the lidar is right")

    if len(G) > 100:
        gh = np.interp(R[:, 0], G[:, 0], G[:, 1])
        gh = gh - np.median(gh[alt < 2]) if (alt < 2).sum() > 20 else gh - gh[0]
        s = fly & np.isfinite(gh)
        if s.sum() > 200:
            al, bl, rl = fit(gh[s], R[s, 1])
            ab, bb, rb = fit(gh[s], alt[s])
            print(f"\n  both against the receiver's altitude, which is "
                  f"independent of each")
            print(f"    lidar = {al:.4f} * GPSalt {bl:+.2f} m   corr {rl:.4f}")
            print(f"    baro  = {ab:.4f} * GPSalt {bb:+.2f} m   corr {rb:.4f}")
            print(f"    the slopes are the GPS's own vertical error and are "
                  f"common to both.")
            print(f"    THE OFFSETS ARE THE RESULT: lidar {bl:+.2f} m, "
                  f"baro {bb:+.2f} m")

    # What the analog chain is actually doing, in volts.
    if sc and off:
        v = R[fly, 1] / float(sc) + float(off)
        span_v = (MA_HIGH - MA_LOW) / 1000.0 * RESISTOR_OHM
        print(f"\n  the pin, worked back from the reading")
        print(f"    {np.min(v):.4f} - {np.max(v):.4f} V over the flight")
        print(f"    one millivolt is {float(sc) * 0.1:.2f} cm at this SCALING")
        print(f"    implied analog span {float(sc) * span_v:.0f} m "
              f"(the sensor ships at 180)")
        # If the lidar is the truth, SCALING should make its slope against
        # the barometer 1.000 - any departure is a scale error we can remove.
        want = float(sc) * a
        print(f"\n  SCALING that would make the two agree on scale: "
              f"{want:.2f}  (now {float(sc):.2f})")
        if abs(want - float(sc)) / float(sc) < 0.02:
            print(f"    within 2%, so the scale is already right and the "
                  f"offset is the whole fault")
        else:
            print(f"    {abs(want - float(sc)) / float(sc) * 100:.1f}% out - "
                  f"worth setting, and it costs nothing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
