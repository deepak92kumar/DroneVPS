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

print("\ndefault is off, and off means ignored")
ign, vn, ve, acc, why = mk(send_velocity=False).velocity_fields(good())
check("horizontal ignored", bool(ign & S.IGN_VEL_HORIZ), f"why={why}")
check("speed accuracy ignored", bool(ign & S.IGN_SPEED_ACC))
check("zeroes in the wire fields", vn == 0.0 and ve == 0.0)

print("\nno estimate, and a stale estimate, are both ignored")
ign, vn, ve, acc, why = mk().velocity_fields(None)
check("none -> ignored", bool(ign & S.IGN_VEL_HORIZ), f"why={why}")
ign, vn, ve, acc, why = mk().velocity_fields(good(valid=False, reason="flow 2.0s stale"))
check("invalid -> ignored", bool(ign & S.IGN_VEL_HORIZ), f"why={why}")
check("the reason is passed through", "stale" in why, why)
check("no stale value on the wire", vn == 0.0 and ve == 0.0)

print("\nthe deadband means IGNORE, not zero")
ign, vn, ve, acc, why = mk().velocity_fields(
    good(deadbanded=True, speed_mps=0.2, sent_vn=0.0, sent_ve=0.0))
check("horizontal ignored rather than asserted zero",
      bool(ign & S.IGN_VEL_HORIZ), f"why={why}")
check("why says it was the deadband", "deadband" in why, why)

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
check("and nothing goes on the wire", bool(ign & S.IGN_VEL_HORIZ) and vn == 0.0,
      f"vn {vn:.1f}, why={why}")
# A speed under the clamp but still absurd for a hover must reach the FC,
# because the bridge cannot know the aircraft is hovering - that is the EKF's
# job with the speed accuracy we declare. What must NOT happen is it arriving
# with a confident accuracy.
b3 = VelocityBridge()
b3.push_flow(0.0, 25.0, 0.0)
ign3, vn3, _, acc3, why3 = mk().velocity_fields(b3.value(0.0, 106.0))
check("25 m/s passes but is declared, not asserted",
      not (ign3 & S.IGN_VEL_HORIZ) and acc3 >= 0.4,
      f"vn {vn3:.1f} acc {acc3:.2f} why={why3}")
b2 = VelocityBridge()
check("630 m/s refused by the bridge", b2.push_flow(0.0, 630.0, 0.0) is False)

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
