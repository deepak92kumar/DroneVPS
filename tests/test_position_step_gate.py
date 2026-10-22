"""The position is ACCUMULATED, so one bad frame is a permanent offset.

fuser.predict() adds each frame's displacement to a running total. A single bad
flow frame therefore does not produce one bad reading - it produces an error
that stays in the position for the rest of the flight. vps_00016 has a
single-sample 20.55 m step at 41.3 m altitude, which at 30 Hz is 616 m/s, and
it is still in that flight's position at the end.

The other two channels were already defended and this one was not:

    map fix     a fix beyond 3 sigma is DROPPED, and an accepted one may move
                the estimate by at most max_correction_m = 0.5 m
    velocity    a physical gate and a 5 m/s^2 output slew since 20 Oct
    position    nothing

Worse, the order made the two disagree. predict() ran unconditionally and
push_flow() could then refuse the SAME displacement on its own gate, so the
position swallowed steps the velocity had rejected - while the comment beside
it claimed they "cannot disagree about what the camera saw". One decision now
feeds both.

THE CEILING IS ONE NUMBER FOR EVERY MODE, at the pilot's instruction: "sabhi
mode main 3-4 ms se jada ka speed". 4 m/s sits above LOIT_SPEED_MS, which he
set to 3 m/s the same evening, so nothing the position controller can command
is refused. The cost is his own stick: in AltHold or Stabilize he can fly
faster than 4 m/s, and while he does every step is refused - which is what the
escape hatch is for.

A REFUSAL IS NOT "THE AIRCRAFT DID NOT MOVE". It is "this measurement cannot be
right", so the frame is treated as a GAP: the accelerometer carries the
position and the variance grows DR_TRUST times faster. Freezing instead would
tell the flight controller the aircraft is stationary, which is the exact fault
the gap handling was written to avoid (log 214, "TRACK LOST 17").
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "vps_vio"))

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


from live_node import VpsNode  # noqa: E402

src = open(os.path.join(HERE, "vps_vio", "live_node.py")).read()


class Bare(VpsNode):
    """gate_step needs no camera, so it is exercised on its own.

    focal is the real one off this aircraft's camera.txt - the ceiling is in
    pixels and a pixel is altitude / focal metres, so the number matters.
    """

    def __init__(self):
        self.n_step_rejected = 0
        self._step_refusals = 0
        self.focal = 791.723


g = Bare()
DT = 1.0 / 30.0
FX = 791.723


def px_to_m(px, alt):
    """One pixel is altitude / focal metres on the ground."""
    return px * alt / FX


# ------------------------------------------------- the unit, not just the value

print("the ceiling is in PIXELS, because that is the unit the error is constant in")
check("50 px a frame", VpsNode.STEP_MAX_PX == 50.0)
check("there is no speed ceiling left", not hasattr(VpsNode, "STEP_MAX_SPEED_MPS"),
      "4 m/s is 3.17 px at 40 m but 1.41 px at 90 m - it shrinks as you climb")
check("there is no mode-dependent branch", "custom_mode" not in src,
      '"sabhi mode main" - one limit, not a mode-aware one')

print("\nwhat the camera really produces passes, at every altitude")
# p50 3.18, p90 13.54, p95 22.39 px over 22738 recorded steps.
for alt in (40.0, 70.0, 90.0, 110.0):
    for px in (3.18, 13.54, 22.39, 40.0):
        d = px_to_m(px, alt)
        _, _, act = Bare().gate_step(d, 0.0, DT, alt)
        if act != "ok":
            check(f"{px:.1f} px at {alt:.0f} m is accepted", False, act)
check("every normal step from p50 to 40 px passes at 40-110 m", True,
      "3.18 / 13.54 / 22.39 / 40 px")

print("\nand the glitch dies at the altitude it happened at")
g2 = Bare()
# vps_00016: 20.55 m in one frame at 41.3 m altitude = 394 px.
_, _, act = g2.gate_step(20.55, 0.0, DT, 41.3)
check("the 20.55 m step from vps_00016 is refused", act == "refused",
      f"{20.55 / (41.3 / FX):.0f} px")
check("it is counted", g2.n_step_rejected == 1)
check("60 px is refused too",
      Bare().gate_step(px_to_m(60, 70.0), 0.0, DT, 70.0)[2] == "refused")
check("and the diagonal is measured, not each axis",
      Bare().gate_step(px_to_m(40, 70.0), px_to_m(40, 70.0), DT, 70.0)[2]
      == "refused", "40 px north and 40 east is 56.6 px")

# ------------------------------------------------- the escape hatch CLIPS

print("\nthe escape hatch concedes the direction but keeps the bound")
g3 = Bare()
alt = 70.0
big = px_to_m(400, alt)                  # a sustained 400 px/frame
acts = [g3.gate_step(big, 0.0, DT, alt) for _ in range(8)]
kinds = [a[2] for a in acts]
check("the first STEP_ESCAPE_N are refused",
      kinds[:VpsNode.STEP_ESCAPE_N] == ["refused"] * VpsNode.STEP_ESCAPE_N,
      str(kinds[:6]))
check("then it concedes", "clipped" in kinds, str(kinds))
i = kinds.index("clipped")
check("and what it concedes is AT the ceiling, not the raw step",
      abs(acts[i][0] - px_to_m(50, alt)) < 1e-6,
      f"{acts[i][0] / (alt / FX):.1f} px, not 400")
check("direction is preserved", acts[i][0] > 0 and acts[i][1] == 0.0)
check("a refused step contributes nothing",
      acts[0][0] == 0.0 and acts[0][1] == 0.0)
check("which is a sixth of a second of lag at 30 Hz",
      abs(VpsNode.STEP_ESCAPE_N * DT - 0.167) < 0.01,
      f"{VpsNode.STEP_ESCAPE_N * DT:.3f} s")

print("\nand it re-arms, so the next isolated glitch still dies")
check("the next impossible step is refused again",
      g3.gate_step(big, 0.0, DT, alt)[2] == "refused",
      "standing open for the rest of the flight would be a hole")

print("\none good frame resets the run")
g4 = Bare()
for _ in range(3):
    g4.gate_step(big, 0.0, DT, alt)
check("three refusals so far", g4.n_step_rejected == 3)
g4.gate_step(px_to_m(3.0, alt), 0.0, DT, alt)
check("a normal frame clears the consecutive count", g4._step_refusals == 0,
      "otherwise scattered glitches would add up to an escape")

# ------------------------------------------------- no evidence, no refusal

print("\nwith nothing to judge the step against, it is used as measured")
for dt, alt_, why in ((None, 70.0, "the first flowed frame of a run"),
                      (0.0, 70.0, "zero interval"),
                      (-0.1, 70.0, "time went backwards"),
                      (0.9, 70.0, "a gap longer than the sane frame window"),
                      (DT, 0.0, "no altitude, so no metres-per-pixel"),
                      (DT, -0.5, "the barometer reads below zero on the ground")):
    _, _, act = Bare().gate_step(999.0, 0.0, dt, alt_)
    check(f"dt={dt}, alt={alt_} -> accepted ({why})", act == "ok", act)

print("\nthe wiring: one decision, both consumers, and a gap on refusal")
# Matched loosely on purpose. An earlier version of this pinned the exact call
# text "self.fuser.predict(dn, de)", and when dt was added for the per-second
# noise rate the string stopped matching and the test raised rather than
# failing a named check - which is how it went unnoticed for a commit.
check("the gate runs BEFORE predict",
      src.index("self.gate_step(dn, de, dt_step, f[\"alt\"])")
      < src.index("self.fuser.predict(dn, de"))
# Matched on ORDER and on which variables are used, not on call text - the
# comment above says exactly that, and the line this replaced pinned
# "push_flow(f[\"t\"], dn / dt_step, de / dt_step)" and broke the moment the
# velocity moved onto a multi-frame baseline on 22 Oct. The property is
# unchanged: the velocity is built from the SAME gated dn/de the fuser got.
check("the velocity is built from the GATED step, after the gate",
      src.index("dn, de, action = self.gate_step(")
      < src.index("self._vel_cum_n += dn")
      < src.index("self.vel.push_flow("),
      "the position must not swallow what the velocity refuses, and a clipped "
      "step must reach both")
check("and it accumulates both axes from those same values",
      "self._vel_cum_n += dn" in src and "self._vel_cum_e += de" in src)
check("the baseline is dropped when a frame does not flow",
      src.index("if not flowed:") < src.index("self._vel_hist.clear()"),
      "a span across a gap carries the gap's time but not its displacement")
check("a refused frame leaves flowed False, so the gap handling runs",
      "# else: flowed stays False" in src,
      "the accelerometer carries it and the variance grows DR_TRUST faster")
check("the count reaches the CSV", '"n_step_rej"' in src)

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
