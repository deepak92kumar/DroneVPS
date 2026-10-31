#!/usr/bin/env python3
"""What the flight controller saw in the seconds before it stopped.

Reads the last part of a dataflash log and lays the timeline out: modes,
errors, the attitude it was asked for against the one it held, what the motors
were being told, the board rail, the battery and the EKF's own doubt. A crash
is usually visible as one of those leaving the others behind.

    python3 crash_report.py /mnt/fccard/APM/LOGS/00000016.BIN
"""
import sys

import numpy as np
from pymavlink import mavutil

MODES = {0: "STABILIZE", 1: "ACRO", 2: "ALT_HOLD", 3: "AUTO", 4: "GUIDED",
         5: "LOITER", 6: "RTL", 7: "CIRCLE", 9: "LAND", 11: "DRIFT",
         16: "POSHOLD", 17: "BRAKE", 20: "GUIDED_NOGPS", 21: "SMART_RTL"}
SUB = {1: "MAIN", 2: "RADIO", 3: "COMPASS", 4: "OPTFLOW", 5: "FS_THROTTLE",
       6: "FS_BATT", 7: "FS_GPS", 8: "FS_GCS", 9: "FS_FENCE", 10: "FLT_MODE",
       11: "GPS_GLITCH", 12: "CRASH_CHECK", 13: "FLIP", 15: "PARACHUTE",
       16: "EKF_CHECK", 17: "FAILSAFE_EKFINAV", 18: "BARO_GLITCH", 19: "CPU",
       22: "NAVIGATION", 24: "EKF_PRIMARY", 25: "THRUST_LOSS",
       26: "FAILSAFE_SENSORS", 29: "FAILSAFE_VIBE", 30: "INTERNAL_ERROR"}
EV = {10: "ARMED", 11: "DISARMED", 15: "AUTO_ARMED", 18: "LAND_COMPLETE",
      19: "LOST_RC", 20: "LOST_RC_RECOVER", 24: "LOST_GPS", 25: "SET_HOME",
      26: "LOST_GPS_RECOVER", 28: "NOT_LANDED", 54: "EKF_ALT_RESET"}


def main(path):
    m = mavutil.mavlink_connection(path)
    d = {k: [] for k in ("MODE", "ERR", "EV", "MSG", "ATT", "RCOU", "POWR",
                         "BAT", "CTUN", "VIBE", "GPS", "RCIN", "XKF1")}
    t0 = None
    while True:
        x = m.recv_match(type=list(d))
        if x is None:
            break
        t = getattr(x, "TimeUS", 0) / 1e6
        if t0 is None:
            t0 = t
        r, ty = t - t0, x.get_type()
        if ty == "MODE":
            d[ty].append((r, MODES.get(int(x.Mode), x.Mode), int(x.Rsn)))
        elif ty == "ERR":
            d[ty].append((r, SUB.get(int(x.Subsys), f"SUBSYS {x.Subsys}"),
                          int(x.ECode)))
        elif ty == "EV":
            d[ty].append((r, EV.get(int(x.Id), f"EV {x.Id}")))
        elif ty == "MSG":
            d[ty].append((r, x.Message))
        elif ty == "ATT":
            d[ty].append((r, x.Roll, x.DesRoll, x.Pitch, x.DesPitch,
                          x.Yaw, x.DesYaw))
        elif ty == "RCOU":
            d[ty].append([r] + [getattr(x, f"C{i}", np.nan)
                                for i in range(1, 7)])
        elif ty == "POWR":
            d[ty].append((r, x.Vcc))
        elif ty == "BAT":
            d[ty].append((r, x.Volt, getattr(x, "Curr", np.nan)))
        elif ty == "CTUN":
            d[ty].append((r, x.Alt, getattr(x, "ThO", np.nan),
                          getattr(x, "DAlt", np.nan)))
        elif ty == "VIBE":
            d[ty].append((r, x.VibeX, x.VibeY, x.VibeZ))
        elif ty == "RCIN":
            d[ty].append((r, x.C1, x.C2, x.C3, x.C4))
        elif ty == "XKF1":
            d[ty].append((r, getattr(x, "PN", np.nan), getattr(x, "PE", np.nan)))

    end = max(v[-1][0] for v in d.values() if v)
    print(f"=== {path.split('/')[-1]}   log {end:.0f} s lamba\n")

    print("  MODES")
    for r, nm, rsn in d["MODE"]:
        print(f"     {r:8.1f} s   {nm:<12} reason {rsn}")
    print(f"\n  EVENTS ({len(d['EV'])})")
    for r, nm in d["EV"]:
        print(f"     {r:8.1f} s   {nm}")
    print(f"\n  ERRORS ({len(d['ERR'])})")
    for r, nm, c in d["ERR"]:
        print(f"     {r:8.1f} s   {nm:<20} code {c}")

    C = np.array(d["CTUN"])
    R = np.array(d["RCOU"])
    print(f"\n  ALTITUDE  min {C[:,1].min():.2f}  max {C[:,1].max():.2f} m")
    spin = R[:, 1:].max(axis=1) > 1200
    if spin.any():
        si = np.flatnonzero(spin)
        print(f"  MOTOR chale   {R[si[0],0]:.1f} - {R[si[-1],0]:.1f} s")

    # The last ten seconds, where the answer usually is.
    lo = end - 10
    print(f"\n  AAKHRI 10 SECOND  ({lo:.0f} - {end:.0f} s)")
    A = np.array(d["ATT"])
    a = A[:, 0] >= lo
    if a.sum():
        print(f"     roll   {A[a,1].min():+7.1f} .. {A[a,1].max():+7.1f}   "
              f"maanga {A[a,2].min():+7.1f} .. {A[a,2].max():+7.1f} deg")
        print(f"     pitch  {A[a,3].min():+7.1f} .. {A[a,3].max():+7.1f}   "
              f"maanga {A[a,4].min():+7.1f} .. {A[a,4].max():+7.1f} deg")
        print(f"     yaw    {A[a,5].min():7.1f} .. {A[a,5].max():7.1f} deg")
    r_ = R[:, 0] >= lo
    if r_.sum():
        print("     motors " + "  ".join(
            f"{R[r_,i].mean():.0f}" for i in range(1, 7)))
    for key, lbl, idx in (("POWR", "Vcc", 1), ("BAT", "battery", 1),
                          ("BAT", "current", 2)):
        V = np.array(d[key])
        if not len(V):
            continue
        s = V[:, 0] >= lo
        if s.sum():
            print(f"     {lbl:9s} {V[s,idx].min():7.2f} .. {V[s,idx].max():7.2f}")
    V = np.array(d["VIBE"])
    if len(V):
        s = V[:, 0] >= lo
        if s.sum():
            print(f"     vibe      {np.percentile(V[s,1:],95,axis=0)}")
    I = np.array(d["RCIN"])
    if len(I):
        s = I[:, 0] >= lo
        if s.sum():
            print(f"     stick     roll {I[s,1].min():.0f}..{I[s,1].max():.0f}  "
                  f"pitch {I[s,2].min():.0f}..{I[s,2].max():.0f}  "
                  f"thr {I[s,3].min():.0f}..{I[s,3].max():.0f}  "
                  f"yaw {I[s,4].min():.0f}..{I[s,4].max():.0f}")

    print(f"\n  AAKHRI MESSAGES")
    for r, s in d["MSG"][-18:]:
        print(f"     {r:8.1f} s   {s[:72]}")
    print()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
