#!/usr/bin/env python3
"""How steady was the hover, in numbers that compare across flights.

"It wobbles more than last time" is the hardest kind of report to act on,
because the pilot is comparing two memories and the aircraft is changing
between flights. This reads the same window out of every log the same way:
the longest steady hover it can find, and then what the attitude did in it.

What to look at, and what it means:

  actual std / p2p   how much the aircraft moved. The headline number.
  desired std        how much the CONTROLLER ASKED it to move. If actual and
                     desired both have it, the aircraft is following orders -
                     the pilot's stick, or the position controller chasing a
                     drifting estimate. If only actual has it, the aircraft is
                     oscillating on its own and it is the rate/attitude loop.
  dominant frequency the signature. Under ~1.5 Hz is the attitude loop or an
                     integrator; 4-30 Hz is the rate loop; above that is
                     usually vibration getting into the gyro.
  rate error std     how far the rate controller is from what it asked for.
  PID I mean         what the integrator is holding up. A large steady I is a
                     trim the aircraft is fighting - CoG, or motor twist.

    python3 hover_quality.py log1.BIN log2.BIN ...
    python3 hover_quality.py log.BIN --from 493 --to 518

It does not say what to change. It says what happened, in the same words
every time, so that two flights can be put side by side.
"""
from __future__ import annotations
import argparse
import sys

import numpy as np
from pymavlink import mavutil

WANT = ["ATT", "RATE", "CTUN", "VIBE", "PIDR", "PIDP", "RCIN", "PARM", "MODE"]


def load(path):
    m = mavutil.mavlink_connection(path)
    R, P = {}, {}
    while True:
        msg = m.recv_match(type=WANT)
        if msg is None:
            break
        if msg.get_type() == "PARM":
            P[msg.Name] = msg.Value
        else:
            R.setdefault(msg.get_type(), []).append(msg)
    return R, P


def sticks_centred(R, dead=60):
    """Times when the pilot was NOT asking for roll or pitch.

    Without this the window can land on a stretch where the pilot is flying
    the aircraft, and then "the demand is large" means only that the stick
    moved. That happened on log 00000218: the first window picked had the
    pitch stick swinging 1292-1752 and the aircraft travelling 13.4 m, and it
    read as a far worse hover than the flight before it. It was not a hover.
    """
    out = []
    for r in R.get("RCIN", []):
        c1, c2 = getattr(r, "C1", 1500), getattr(r, "C2", 1500)
        if abs(c1 - 1500) <= dead and abs(c2 - 1500) <= dead:
            out.append(r.TimeUS / 1e6)
    return out


def hover_window(R, want=25.0):
    """The longest stretch that is actually a hover, not a climb or a let-down.

    Hands off the sticks, throttle up, not climbing, off the ground.
    """
    if "CTUN" not in R:
        return None
    still = sticks_centred(R)
    if still:
        lo, hi = min(still), max(still)
        quiet = np.array(sorted(still))
    else:
        quiet = None
    c = [(r.TimeUS / 1e6, r.ThO, r.CRt, r.Alt) for r in R["CTUN"]]
    def hands_off(t):
        if quiet is None:
            return True
        i = np.searchsorted(quiet, t)
        return (i < len(quiet) and abs(quiet[i] - t) < 0.5) or \
               (i > 0 and abs(quiet[i - 1] - t) < 0.5)
    ok = [t for t, tho, crt, alt in c
          if tho > 0.2 and abs(crt) < 0.3 and alt > 1.5 and hands_off(t)]
    if not ok:
        return None
    runs, s, p = [], ok[0], ok[0]
    for t in ok[1:]:
        if t - p > 0.5:
            runs.append((s, p))
            s = t
        p = t
    runs.append((s, p))
    runs.sort(key=lambda r: r[1] - r[0], reverse=True)
    a, b = runs[0]
    return (b - want if b - a > want else a), b


def peaks(t, v, n=3, lo=0.15, hi=8.0):
    dt = np.median(np.diff(t))
    v = np.asarray(v, dtype=float)
    v = v - v.mean()
    w = np.hanning(len(v))
    f = np.fft.rfftfreq(len(v), dt)
    a = np.abs(np.fft.rfft(v * w)) * 2 / w.sum()
    out = []
    for i in np.argsort(a)[::-1]:
        if not (lo <= f[i] <= hi):
            continue
        if any(abs(f[i] - q) < 0.25 for q, _ in out):
            continue
        out.append((f[i], a[i]))
        if len(out) >= n:
            break
    return out


def report(path, t0=None, t1=None):
    R, P = load(path)
    if "ATT" not in R:
        print(f"  {path}: no ATT - nothing to measure")
        return
    w = (t0, t1) if (t0 is not None and t1 is not None) else hover_window(R)
    if w is None:
        print(f"  {path}: no steady hover in this log")
        return
    a, b = w
    print("\n=== %s   hover %.1f-%.1f s (%.0f s) ===" % (path.split("/")[-1], a, b, b - a))
    frame = (int(P.get("FRAME_CLASS", -1)), int(P.get("FRAME_TYPE", -1)))
    print("   FRAME_CLASS/TYPE %s   ATC_RAT_RLL_P/I/D %.3f/%.3f/%.4f   "
          "PIT %.3f/%.3f/%.4f" % (
              frame, P.get("ATC_RAT_RLL_P", 0), P.get("ATC_RAT_RLL_I", 0),
              P.get("ATC_RAT_RLL_D", 0), P.get("ATC_RAT_PIT_P", 0),
              P.get("ATC_RAT_PIT_I", 0), P.get("ATC_RAT_PIT_D", 0)))
    print("   AHRS_TRIM %.4f/%.4f rad   INS_ACCSCAL_X %.6f   HNTCH %.0fHz ref %.3f bw %.0f" % (
        P.get("AHRS_TRIM_X", 0), P.get("AHRS_TRIM_Y", 0), P.get("INS_ACCSCAL_X", -1),
        P.get("INS_HNTCH_FREQ", 0), P.get("INS_HNTCH_REF", 0), P.get("INS_HNTCH_BW", 0)))

    sel = lambda n, fs: (
        np.array([r.TimeUS / 1e6 for r in R[n] if a <= r.TimeUS / 1e6 <= b]),
        {f: np.array([getattr(r, f) for r in R[n] if a <= r.TimeUS / 1e6 <= b],
                     dtype=float) for f in fs})
    t, A = sel("ATT", ["Roll", "DesRoll", "Pitch", "DesPitch", "Yaw", "DesYaw"])
    print("\n   axis   actual std   p2p   desired std   error std   dominant")
    for ax in ("Roll", "Pitch"):
        e = A[ax] - A["Des" + ax]
        pk = peaks(t, A[ax], 2)
        print("   %-6s %9.2f deg %6.1f %11.2f %11.2f   %s" % (
            ax, A[ax].std(), A[ax].ptp(), A["Des" + ax].std(), e.std(),
            "  ".join("%.2fHz %.2f" % (q, v) for q, v in pk)))
        pkd = peaks(t, A["Des" + ax], 2)
        print("          %-44s demand: %s" % (
            "", "  ".join("%.2fHz %.2f" % (q, v) for q, v in pkd)))

    if "RATE" in R:
        tr, Rt = sel("RATE", ["R", "RDes", "P", "PDes", "Y", "YDes"])
        print()
        for nm, (x, y) in (("roll", ("R", "RDes")), ("pitch", ("P", "PDes")),
                           ("yaw", ("Y", "YDes"))):
            print("   rate %-6s actual std %6.2f   desired std %6.2f   "
                  "error std %6.2f deg/s" % (nm, Rt[x].std(), Rt[y].std(),
                                             (Rt[x] - Rt[y]).std()))
    if "VIBE" in R:
        v = [r for r in R["VIBE"] if r.IMU == 0 and a <= r.TimeUS / 1e6 <= b]
        if v:
            print("\n   VibeZ p50 %.2f  max %.2f  clip +%d   (30 = failsafe, "
                  "15 healthy, 5 ideal)" % (
                      np.percentile([r.VibeZ for r in v], 50),
                      max(r.VibeZ for r in v), v[-1].Clip - v[0].Clip))
    for nm, lab in (("PIDR", "roll"), ("PIDP", "pitch")):
        if nm not in R:
            continue
        rows = [r for r in R[nm] if a <= r.TimeUS / 1e6 <= b]
        if rows:
            print("   PID %-5s I mean %+.4f (the standing trim it holds up)  "
                  "P std %.4f  D std %.4f" % (
                      lab, np.mean([r.I for r in rows]),
                      np.std([r.P for r in rows]), np.std([r.D for r in rows])))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--from", dest="t0", type=float, default=None)
    ap.add_argument("--to", dest="t1", type=float, default=None)
    a = ap.parse_args()
    for p in a.logs:
        report(p, a.t0, a.t1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
