"""What the FC is told about velocity, in every case.

This is the flight-sent path. On GPS_INPUT the flight controller applies no
vision health checks - it follows what it is given - so the only safe way to
change this is to pin each case down in a test first. The 28 Sep failure was
exactly this channel: 47.7 m/s reported during a 106 m hover, four EKF
failsafes.

The case that most needs to be here is the deadband one. When the deadband
bites we go back to IGNORE rather than sending zero, because 5 Oct measured
that deadband discarding 49% of genuine motion below 25 m - so a zero would be
a confident false "stationary" to something that follows it.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "vps_vio"))
from live_node import GpsInputSender        # noqa: E402
from velocity_bridge import VelocityState   # noqa: E402

S = GpsInputSender
fails = []


def mk(**kw):
    """A sender without its serial connection: velocity_fields needs no link."""
    s = object.__new__(S)
    s.send_velocity = kw.pop("send_velocity", True)
    return s


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def good(**kw):
    d = dict(vn=6.0, ve=-2.0, valid=True, reason="ok", sent_vn=6.0, sent_ve=-2.0,
             deadbanded=False, deadband_mps=0.9, speed_mps=6.32, age_s=0.04)
    d.update(kw)
    return VelocityState(**d)


print("vertical velocity is never claimed")
for name, s, v in (("off", mk(send_velocity=False), good()),
                   ("none", mk(), None),
                   ("good", mk(), good()),
                   ("stale", mk(), good(valid=False, reason="flow stale")),
                   ("deadbanded", mk(), good(deadbanded=True, speed_mps=0.2))):
    ign = s.velocity_fields(v)[0]
    check(f"IGN_VEL_VERT set ({name})", bool(ign & S.IGN_VEL_VERT))

# ------------------------------------------------------------------ 3 NOV
# EVERY ASSERTION IN THIS BLOCK IS REVERSED FROM WHAT IT WAS, and the old
# ones were wrong about ArduPilot, not about our intent.
#
# This file used to require that each refusal set IGN_VEL_HORIZ, on the
# belief that the flag says "I do not know" to the EKF. It does not.
# AP_GPS_MAV::handle_msg assigns the velocity only inside
#
#     if (have_vel_h) { state.velocity = vel; velocity_to_speed_course(state); }
#
# with no else branch, nothing anywhere clearing state.velocity, and no
# have_horizontal_velocity flag for EK3_SRC1_VELXY=3 to consult. The flag
# therefore FREEZES the previous velocity, and when none was ever sent the
# frozen value is (0,0,0) - which EKF3 fuses as a confident measurement
# that the aircraft is standing still. IGN_SPEED_ACC freezes
# state.speed_accuracy the same way.
#
# Flight 00000118, the 60 s VPS leg at 26.7-36.1 m, above the altitude gate,
# so by the old design nothing was being sent:
#     GPS2 reported speed       0.000 m/s on 200/200 samples
#     the aircraft really did   p50 1.44, p90 3.95, max 6.65 m/s
#     XKF4.SV crossed 1.0 twice (1.43, 2.99); both ERR GPS glitches followed
#     XKF4.SP never passed 0.78 - the position we sent was not the problem
#
# There is no third state on this path. What we send is fused; staying
# silent sends a zero. So a refusal now hands over the last velocity we
# really measured and puts the doubt in the declared accuracy.
print("\na refusal holds the last measurement - it never asserts stillness")
for label, sender, v in (("send_velocity off", mk(send_velocity=False), good()),
                         ("no estimate", mk(), None),
                         ("stale", mk(), good(valid=False, reason="flow 2.0s stale")),
                         ("deadbanded", mk(),
                          good(deadbanded=True, speed_mps=0.2,
                               sent_vn=0.0, sent_ve=0.0))):
    ign, vn, ve, acc, why = sender.velocity_fields(v)
    check(f"{label}: horizontal NOT ignored", not (ign & S.IGN_VEL_HORIZ),
          f"why={why}")
    check(f"{label}: speed accuracy NOT ignored", not (ign & S.IGN_SPEED_ACC))
    check(f"{label}: a zero is declared at the widest uncertainty",
          vn != 0.0 or ve != 0.0 or acc >= S.VEL_HOLD_MAX_ACC - 1e-9,
          f"vn {vn} ve {ve} acc {acc:.2f}")

check("the stale reason still reaches the log",
      "stale" in mk().velocity_fields(
          good(valid=False, reason="flow 2.0s stale"))[4])
check("the deadband reason still reaches the log",
      "deadband" in mk().velocity_fields(
          good(deadbanded=True, speed_mps=0.2))[4])

# And the hold is a real hold: once something has been measured, THAT is
# what a later refusal repeats, not a zero.
held = mk()
held.velocity_fields(good())                      # measures 6.0 / -2.0
ign, vn, ve, acc, why = held.velocity_fields(None)   # then has nothing
check("a later refusal repeats the measured value",
      (vn, ve) == (6.0, -2.0), f"vn {vn} ve {ve} why={why}")
check("and declares more error than when it was fresh",
      acc > held.VEL_ACC_FLOOR, f"{acc:.2f} m/s")
check("the hold is bounded, never beyond the widest honest figure",
      acc <= S.VEL_HOLD_MAX_ACC + 1e-9, f"{acc:.2f} m/s")

print("\na good estimate is sent, with an accuracy")
ign, vn, ve, acc, why = mk().velocity_fields(good())
check("horizontal NOT ignored", not (ign & S.IGN_VEL_HORIZ), f"why={why}")
check("speed accuracy NOT ignored", not (ign & S.IGN_SPEED_ACC))
check("values passed through", vn == 6.0 and ve == -2.0, f"vn {vn} ve {ve}")
check("accuracy has a floor", acc >= 0.4, f"{acc:.2f} m/s")

print("\naccuracy widens as the flow goes stale")
a0 = mk().velocity_fields(good(age_s=0.0))[3]
a1 = mk().velocity_fields(good(age_s=0.2))[3]
a2 = mk().velocity_fields(good(age_s=0.5))[3]
check("monotonic in age", a0 < a1 < a2, f"{a0:.2f} < {a1:.2f} < {a2:.2f}")

print("\nthe bridge's own guards still hold at this boundary")
from velocity_bridge import VelocityBridge  # noqa: E402
b = VelocityBridge()
accepted = b.push_flow(0.0, 47.7, 0.0)   # the exact 28 Sep hover reading
check("47.7 m/s is refused by the bridge outright", accepted is False,
      f"rejected_flow {b.rejected_flow}")
st = b.value(0.0, 106.0)
check("so no estimate exists to send", st.valid is False, st.reason)
ign, vn, ve, acc, why = mk().velocity_fields(st)
# 3 NOV: not "nothing goes on the wire" - nothing CAN be withheld here. The
# bridge refused 47.7 m/s, so there is no measurement, and what goes out is
# a zero declared at the full 6.0 m/s so the EKF gives it no weight.
check("a refused reading is declared worthless, not asserted",
      not (ign & S.IGN_VEL_HORIZ) and vn == 0.0
      and acc >= S.VEL_HOLD_MAX_ACC - 1e-9,
      f"vn {vn:.1f}, acc {acc:.2f}, why={why}")
# A speed under the clamp but still absurd for a hover must reach the FC,
# because the bridge cannot know the aircraft is hovering - that is the EKF's
# job with the speed accuracy we declare. What must NOT happen is it arriving
# with a confident accuracy.
#
# 2 NOV: the altitude is now 10 m, not 106. Velocity is only sent below
# vel_send_max_alt_m, because above that the measured per-axis p90 against
# gps1 passes EKF3's own ~1.0 m/s GPS-quality threshold and an HONEST
# declaration is then the thing that marks us a glitching GPS - which took
# LOITER away five times on log 116. At 106 m the answer now is "not sent at
# all", so testing the declaration there tests nothing. The question this
# case exists to ask - is an absurd speed declared rather than asserted - is
# the same at 10 m.
b3 = VelocityBridge()
b3.push_flow(0.0, 25.0, 0.0)
ign3, vn3, _, acc3, why3 = mk().velocity_fields(b3.value(0.0, 10.0))
check("25 m/s passes but is declared, not asserted",
      not (ign3 & S.IGN_VEL_HORIZ) and acc3 >= 0.4,
      f"vn {vn3:.1f} acc {acc3:.2f} why={why3}")
b2 = VelocityBridge()
check("630 m/s refused by the bridge", b2.push_flow(0.0, 630.0, 0.0) is False)

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
