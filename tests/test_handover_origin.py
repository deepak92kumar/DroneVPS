"""Where the origin comes from, in each of the three cases the pilot named.

The pilot's instruction, 21 Oct, and the argument is his own. When the switch
goes from the VPS back to GPS, the flight controller does not drag the aircraft
across whatever error had built up:

    "man lo error 50 mtr hai tho gps drone ko 50 mtr lekar correct nahi karta,
     wo direct apne current position inject kar deta hai aur wahi stable ho
     jata hai"

Handing over TO the VPS has to behave the same way. It did not.

  1. ARM ON THE VPS, then loiter          origin = SCR_USER1/2
  2. ARM ON GPS                           origin = where the FC says it is
  3. SWITCH TO THE VPS IN FLIGHT          origin = where the FC says it is NOW,
                                          and the VIO displacement is dropped

Case 3 did not exist. The origin was only ever re-taken at the ARM edge, so a
flight that armed on GPS, flew, and switched to the VPS at altitude began
sending `takeoff origin + every metre of VIO drift from the whole GPS phase`.
Measured on the three recordings with real travel below 60 m, against the FC's
own GPS-driven position:

    vps_00013    drift max  5.73 m over 53.8 m travelled
    vps_00019    drift max  5.24 m over 43.6 m travelled
    vps_00016    a single-sample 20.55 m jump at 41.3 m altitude

The EKF holds the GPS position at the instant of the switch, so it saw a step of
exactly that size. Re-anchoring makes the first VPS position equal what the EKF
already holds: no step, and the drift is discarded rather than flown off.

This file pins the three cases and the two that must NOT fire.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "vps_vio"))

from map_fuse import MapFuser  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


src = open(os.path.join(HERE, "vps_vio", "live_node.py")).read()

# ------------------------------------------------- the state machine itself

from live_node import should_handover  # noqa: E402

print("the node watches both edges, not just arming")
check("the armed flag is tracked frame to frame", "_armed_was = [None]" in src)
check("and GPS_PRIMARY is too", "_src_was = [None]" in src)

print("\nthe handover predicate, driven rather than read")
# (src_val, src_was, armed_was, armed_now, have_origin) -> expected, why
CASES = [
    ((1, 0, True, True, True), True,
     "case 3: armed, flying on GPS, switch moves to the VPS"),
    ((1, None, True, True, True), False,
     "first frame - previous source unknown, so a boot on the VPS is a "
     "take-off, not a handover"),
    ((1, 1, True, True, True), False,
     "already on the VPS - not an edge, so nothing to hand over"),
    ((0, 1, True, True, True), False,
     "the other direction: VPS -> GPS needs no re-anchor, we just stop sending"),
    ((1, 0, False, True, True), False,
     "armed only as of THIS frame - that is case 1, and SCR_USER must stand"),
    ((1, 0, None, True, True), False,
     "armed state was unknown last frame"),
    ((1, 0, True, False, True), False,
     "disarmed now - a switch on the ground is case 1"),
    ((1, 0, True, True, False), False,
     "no origin yet - nothing to replace"),
    ((None, 0, True, True, True), False,
     "GPS_PRIMARY unreadable - do not re-anchor on a guess"),
    ((1, None, None, None, True), False,
     "cold start, nothing known"),
]
for args, expect, why in CASES:
    got = should_handover(*args)
    check(f"{'fires' if expect else 'does not fire'}: {why}", got is expect,
          f"should_handover{args} = {got}")

# ------------------------------------------------- case 3 does the right thing

print("\ncase 3 - switching to the VPS in flight")
h = src[src.index("# THE HANDOVER, IN FLIGHT."):]
h = h[:h.index("_src_was[0] = src_val")]
check("the origin is re-taken from the FC's own position",
      "node.set_origin(*f[\"gps\"], sigma_m=f.get(\"gps1_hacc\"))" in h)
check("and the VIO displacement is dropped, not carried",
      "node.fuser.reset_displacement()" in h,
      "carrying it is the whole fault - it is the step the EKF sees")
check("how much drift was discarded is reported",
      "displacement discarded" in h and "carried = " in h,
      "a handover that silently dropped 20 m should say so")
check("it is announced on the radio", 'notifier.event("handover"' in h)

print("\nand when there is no fix to hand over from - the jammed case")
check("the origin and displacement are KEPT",
      "NO FIX, drift carried" in h)
check("SCR_USER1/2 is NOT used as the fallback",
      "scr_origin" not in h,
      "it sat 147-157 m from the aircraft on three flights - that is a worse "
      "jump than the drift")

# ------------------------------------------------- what must NOT fire

print("\nthe two cases this must not touch")
# The conditions themselves live in should_handover and are driven above. What
# is checked here is that the loop actually ASKS it, rather than carrying its
# own copy of the rule that could drift from the tested one.
check("the loop defers to the tested predicate",
      "should_handover(src_val, _src_was[0], _armed_was[0]," in h,
      "an inline copy of the condition is a second rule to keep in step")
check("and it passes the previous frame's armed state, not just this one",
      "_armed_was[0]" in h and "armed_now" in h)

# The arm edge must still be the one that reads SCR_USER.
arm = src[src.index("# THE ORIGIN IS RE-TAKEN AT ARM."):]
arm = arm[:arm.index("# THE HANDOVER, IN FLIGHT.")]
check("case 1 is still the arm edge's job",
      'scr_origin, "SCR_USER1/2 at arm"' in arm)
check("case 2 is still the arm edge's job",
      'f["gps"], "receiver at arm"' in arm)

# ------------------------------------------------- and the ordering

print("\nordering: the handover runs before the armed bookkeeping")
i_h = src.index("# THE HANDOVER, IN FLIGHT.")
i_a = src.index("_armed_was[0] = armed_now", i_h)
i_s = src.index("_src_was[0] = src_val")
check("the handover reads _armed_was before it is overwritten", i_h < i_a)
check("and both trackers are updated after it", i_s < i_a or i_s > i_h)

# ------------------------------------------------- the arithmetic it relies on

print("\nreset_displacement is what makes the step zero")
f = MapFuser(flow_drift_frac=0.07, fix_sigma_m=20.0)
for _ in range(400):
    f.predict(1.0, 0.0)
drifted = float((f.pos[0] ** 2 + f.pos[1] ** 2) ** 0.5)
check("400 m of flow leaves a displacement to discard", drifted > 300.0,
      f"{drifted:.1f} m")
f.reset_displacement()
after = float((f.pos[0] ** 2 + f.pos[1] ** 2) ** 0.5)
check("and it goes to exactly zero", after == 0.0, f"{after:.3f} m")
check("with the uncertainty back at its initial value",
      abs(f.sigma_m - 2.0) < 1e-9, f"sigma {f.sigma_m:.3f} m")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
