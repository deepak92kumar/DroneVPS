#!/usr/bin/env python3
"""Where is the hover imbalance - CoG, or motor twist? Read it off a log.

ZD850 hexa-X, Tarot MT4008 340 kV x6, Hobbywing 40 A ESCs. No ESC telemetry on
this aircraft, so motor PWM is the only per-arm signal there is - but that is
enough, because six motors holding a steady hover are solving a four-unknown
problem (total thrust, roll, pitch, yaw) and the system is over-determined.
Fit the six thrusts onto those four and the residual says whether the fit is
real. On the 20 Oct hover the residual came out at 0.0% of the spread, which
is what makes the rest of the numbers worth reading.

Why it matters. "The drone wobbles" and "the drone is out of balance" are
different problems with different fixes, and a hover log can tell them apart:

  * a ROLL or PITCH share means the centre of gravity is off centre. In a
    steady hover the moments balance, so the thrust centroid IS the CoG.
    Move the battery.
  * a YAW share means the motors are fighting each other about heading -
    mounts not level, an arm twisted in its clamp, or the CW and CCW props
    not matched. No amount of battery shifting fixes it, and it eats the yaw
    authority that holding heading needs.

20 Oct, hover 108-121 s of log 00000216:

    yaw   50.2%          the biggest single piece, and not a CoG problem
    roll  29.5%          CoG 16.8 mm LEFT
    pitch 20.3%          CoG 11.5 mm FORWARD
    pwm spread 125 us, thrust spread 44% of the mean

Run it again after every mechanical change. Two numbers that got smaller is
progress; a number that moved from yaw to roll is a battery that moved.

    python3 hover_balance.py /data/DRONE/flight_logs/00000216.BIN
    python3 hover_balance.py <log> --from 108 --to 121

With no window given it finds the steadiest stretch of hover itself: throttle
up, climb rate near zero, level.
"""
from __future__ import annotations
import argparse
import sys

import numpy as np
from pymavlink import mavutil

# ArduPilot's own hexa-X table (AP_MotorsMatrix::setup_hexa_matrix), angle in
# degrees CLOCKWISE from the nose. +y is to the RIGHT, +x is FORWARD.
MOTORS = {1: (90, +1), 2: (-90, -1), 3: (-30, +1),
          4: (150, -1), 5: (30, -1), 6: (-150, +1)}
WHERE = {1: "right", 2: "left", 3: "front-left",
         4: "rear-right", 5: "front-right", 6: "rear-left"}
SPINS = {1: "CW", -1: "CCW"}
ARM_RADIUS_M = 0.425          # ZD850: 850 mm across the motors


def thrust_from_pwm(pwm, pwm_min, pwm_max, spin_min, spin_max, expo):
    """Undo ArduPilot's thrust curve, so PWM becomes something additive.

    Motor output is spin_min + (spin_max-spin_min) * actuator(thrust), and
    actuator(thrust) is the MOT_THST_EXPO curve. Averaging raw PWM instead
    would mix a non-linear axis into a linear sum and quietly understate the
    imbalance at the low end.
    """
    out01 = (np.asarray(pwm, dtype=float) - pwm_min) / (pwm_max - pwm_min)
    a = np.clip((out01 - spin_min) / (spin_max - spin_min), 0.0, 1.0)
    if abs(expo) < 1e-6:
        return a
    return ((2 * expo * a + 1 - expo) ** 2 - (1 - expo) ** 2) / (4 * expo)


def load(path):
    m = mavutil.mavlink_connection(path)
    rcou, ctun, att, parm = [], [], [], {}
    while True:
        msg = m.recv_match(type=["RCOU", "CTUN", "ATT", "PARM"])
        if msg is None:
            break
        t = msg.get_type()
        if t == "PARM":
            parm[msg.Name] = msg.Value
        elif t == "RCOU":
            rcou.append((msg.TimeUS / 1e6,
                         [getattr(msg, "C%d" % i) for i in range(1, 7)]))
        elif t == "CTUN":
            ctun.append((msg.TimeUS / 1e6, msg.ThO, msg.CRt))
        elif t == "ATT":
            att.append((msg.TimeUS / 1e6, msg.Roll, msg.Pitch))
    return rcou, ctun, att, parm


def pick_window(ctun, att, want=10.0):
    """The steadiest hovering stretch: throttle up, not climbing, level."""
    if not ctun:
        return None
    # |CRt| < 0.3, not 0.6: a 0.5 m/s descent also passes a loose gate, and
    # on the 20 Oct log that pulled the window onto the let-down instead of
    # the hover. The two disagreed by 6 mm of CoG and 5 points of yaw share.
    ok = [t for t, tho, crt in ctun if tho > 0.2 and abs(crt) < 0.3]
    if not ok:
        return None
    runs, start, prev = [], ok[0], ok[0]
    for t in ok[1:]:
        if t - prev > 0.5:
            runs.append((start, prev))
            start = t
        prev = t
    runs.append((start, prev))
    runs.sort(key=lambda r: r[1] - r[0], reverse=True)
    a, b = runs[0]
    if b - a > want:                    # trim the climb-out off the front
        a = b - want
    return a, b


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="a dataflash .BIN")
    ap.add_argument("--from", dest="t0", type=float, default=None)
    ap.add_argument("--to", dest="t1", type=float, default=None)
    ap.add_argument("--radius", type=float, default=ARM_RADIUS_M,
                    help="arm radius in metres (ZD850: 0.425)")
    a = ap.parse_args()

    rcou, ctun, att, parm = load(a.log)
    if not rcou:
        sys.exit("no RCOU in that log - nothing to read")

    if a.t0 is not None and a.t1 is not None:
        t0, t1 = a.t0, a.t1
    else:
        w = pick_window(ctun, att)
        if w is None:
            sys.exit("no hovering stretch in that log (throttle never came up)")
        t0, t1 = w
        print(f"  hover window chosen automatically: {t0:.1f}-{t1:.1f} s")

    P = np.array([v for t, v in rcou if t0 <= t <= t1], dtype=float)
    if len(P) < 10:
        sys.exit(f"only {len(P)} RCOU samples in {t0:.1f}-{t1:.1f} s")

    expo = parm.get("MOT_THST_EXPO", 0.65)
    T = thrust_from_pwm(P, parm.get("MOT_PWM_MIN", 1000.0),
                        parm.get("MOT_PWM_MAX", 2000.0),
                        parm.get("MOT_SPIN_MIN", 0.15),
                        parm.get("MOT_SPIN_MAX", 0.95), expo)
    pm, tm = P.mean(0), T.mean(0)

    frame = (int(parm.get("FRAME_CLASS", 0)), int(parm.get("FRAME_TYPE", 0)))
    if frame != (2, 1):
        print(f"  *** FRAME_CLASS/TYPE is {frame}, not (2,1) hexa-X - the "
              f"motor table below is for hexa-X and will be wrong ***")
    print(f"  {len(P)} samples over {t1 - t0:.1f} s, "
          f"MOT_THST_EXPO={expo:g}\n")

    print("%-4s %-12s %-4s %7s %8s %9s"
          % ("mot", "position", "spin", "pwm", "thrust", "vs mean"))
    for i in range(6):
        ang, yawf = MOTORS[i + 1]
        print("%-4d %-12s %-4s %7.0f %8.3f %+8.1f%%"
              % (i + 1, WHERE[i + 1], SPINS[yawf], pm[i], tm[i],
                 100 * (tm[i] - tm.mean()) / tm.mean()))
    print("\n  pwm spread %.0f us, thrust spread %.0f%% of the mean"
          % (pm.ptp(), 100 * tm.ptp() / tm.mean()))

    # Six thrusts onto four demands. The mixer's own factors, so the answer is
    # in the same axes the controller works in.
    A = np.array([[1.0,
                   -np.sin(np.radians(MOTORS[i][0])),
                   np.cos(np.radians(MOTORS[i][0])),
                   MOTORS[i][1]] for i in range(1, 7)])
    sol, _, _, _ = np.linalg.lstsq(A, tm, rcond=None)
    resid = float(np.sqrt(((tm - A @ sol) ** 2).mean()))
    thr, roll_in, pitch_in, yaw_in = sol

    share = {"roll": abs(roll_in), "pitch": abs(pitch_in), "yaw": abs(yaw_in)}
    total = sum(share.values()) or 1.0
    print("\n  what the imbalance is made of:")
    for k, v in sorted(share.items(), key=lambda kv: -kv[1]):
        print("    %-6s %5.1f%%   (demand %+.5f)" % (k, 100 * v / total,
                                                     {"roll": roll_in, "pitch": pitch_in, "yaw": yaw_in}[k]))
    print("    mean thrust  %.4f" % thr)
    print("    residual     %.6f rms = %.1f%% of the spread"
          % (resid, 100 * resid / max(tm.ptp(), 1e-9)))
    if resid / max(tm.ptp(), 1e-9) > 0.10:
        print("    *** over 10%: the six motors are NOT explained by three "
              "axes. Check the frame type, or look for one motor behaving "
              "differently from the other five. ***")

    # In a steady hover the moments balance about the CoG, so the thrust
    # centroid is the CoG.
    y = np.array([a.radius * np.sin(np.radians(MOTORS[i][0])) for i in range(1, 7)])
    x = np.array([a.radius * np.cos(np.radians(MOTORS[i][0])) for i in range(1, 7)])
    y_cg = float((tm * y).sum() / tm.sum())
    x_cg = float((tm * x).sum() / tm.sum())
    print("\n  centre of gravity, against the middle of the frame:")
    print("    %5.1f mm to the %s" % (abs(y_cg) * 1000,
                                      "RIGHT" if y_cg > 0 else "LEFT"))
    print("    %5.1f mm %s" % (abs(x_cg) * 1000,
                               "FORWARD" if x_cg > 0 else "AFT"))
    print("    (move the battery the SAME way to correct it: the CoG is "
          "where the thrust is concentrated)")

    if abs(yaw_in) > max(abs(roll_in), abs(pitch_in)):
        print("\n  yaw is the largest share, so the biggest piece of this is "
              "NOT the CoG.\n  Check, in this order: each motor mount level "
              "with the arm, no arm\n  twisted in its clamp, and the CW props "
              "matched to the CCW props\n  (same model and pitch). A standing "
              "yaw demand also spends the\n  authority that holding heading "
              "needs - on 20 Oct the yaw rate\n  controller was asked for "
              "10.6 deg/s and delivered 3.5.")

    per = T.std(0)
    print("\n  per-motor variation during the window (noise, not trim): %s"
          % "  ".join("M%d %.3f" % (i + 1, per[i]) for i in range(6)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
