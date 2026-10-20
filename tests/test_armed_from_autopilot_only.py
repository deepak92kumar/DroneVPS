"""`armed` must come from the autopilot's heartbeat, not from anyone's.

Why this exists. Everything on this aircraft's MAVLink bus heartbeats every
5 seconds, deliberately: the router forgets a UDP client it has not heard from
in 30 s and that client then goes silently deaf, so the RC switch, the
notifier, the watchdog and any tool someone runs all keep talking. Every one
of those heartbeats carries `base_mode = 0`, which reads as DISARMED.

Telemetry took `armed` from the last HEARTBEAT it saw, from anybody. Nothing
depended on that flag until 20 Oct, when the pilot asked for images to be
saved only while armed - and then it mattered, every five seconds. From the
13:49 flight, log 00000217:

  * the frame gate shut once per 5 s: 846 images saved where 980 were due,
    14% of the flight's record gone
  * "VPS FRAMES OFF - DISARMED" went out on the radio twenty times while the
    aircraft was flying
  * "LIDAR ZERO 3.4m - NO AUTO-DISARM" was announced at 1 m altitude, from a
    check written to speak only on the ground

Read off the real bus, 20 s:

    sys/comp   n  autopilot  type  base_mode
    1/1       21          3    13         89   <- the flight controller
    1/221      4          8     6          0   <- ours
    255/0      4          8     6          0   <- a GCS-style client

So the test is component 1 AND an autopilot id that is not
MAV_AUTOPILOT_INVALID. This file pins both directions, because the dangerous
failure is not the old one - it is a filter so strict that the FC's own
heartbeat is thrown away too and no flight ever records an image again.
"""
import os
import sys
import types

from pymavlink import mavutil

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "vps_vio"))

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


ARMED_BIT = mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
INVALID = mavutil.mavlink.MAV_AUTOPILOT_INVALID


class FakeHeartbeat:
    """Only what the branch under test reads."""

    def __init__(self, sysid, compid, autopilot, type_, base_mode, custom_mode=5):
        self._s, self._c = sysid, compid
        self.autopilot, self.type = autopilot, type_
        self.base_mode, self.custom_mode = base_mode, custom_mode

    def get_type(self):
        return "HEARTBEAT"

    def get_srcSystem(self):
        return self._s

    def get_srcComponent(self):
        return self._c


def accepts(msg):
    """The rule as record_flight.py applies it."""
    return not (msg.get_srcComponent() != 1 or msg.autopilot == INVALID)


# ---- the rule itself, on the heartbeats that were really on the bus ----

print("heartbeats seen on the aircraft's own bus")
# 89 is what this aircraft really sends while DISARMED (read off the bus);
# MAV_MODE_FLAG_SAFETY_ARMED is 128, so armed is 89 | 128 = 217.
fc_disarmed = FakeHeartbeat(1, 1, 3, 13, 89)
fc_armed = FakeHeartbeat(1, 1, 3, 13, 89 | ARMED_BIT)
ours = FakeHeartbeat(1, 221, INVALID, 6, 0)        # our own component
gcs = FakeHeartbeat(255, 0, INVALID, 6, 0)         # a GCS-style client

check("the flight controller's heartbeat is used", accepts(fc_armed),
      "component 1, autopilot 3 - without this nothing ever records")
check("our own component's heartbeat is ignored", not accepts(ours),
      "1/221, base_mode 0 - this is what flipped armed false every 5 s")
check("a GCS-style heartbeat is ignored", not accepts(gcs), "255/0")
check("the armed bit is read from the FC's base_mode",
      bool(fc_armed.base_mode & ARMED_BIT) is True
      and bool(fc_disarmed.base_mode & ARMED_BIT) is False,
      f"armed {fc_armed.base_mode}, disarmed {fc_disarmed.base_mode}")

# A component-1 heartbeat from a real autopilot on another system id is still
# an autopilot; the component test is what separates tools from vehicles.
check("component 1 with a real autopilot id is accepted whatever the sysid",
      accepts(FakeHeartbeat(2, 1, 12, 2, 0)))
check("a tool that claims component 1 but no autopilot is still ignored",
      not accepts(FakeHeartbeat(1, 1, INVALID, 6, 0)),
      "autopilot INVALID is the giveaway")

# ---- and that the source actually contains the rule ----

print("\nthe rule is in the code that runs")
src = open(os.path.join(HERE, "vps_vio", "record_flight.py")).read()
hb = src.split('elif t == "HEARTBEAT":')[1].split("elif t ==")[0]
check("the HEARTBEAT branch filters on the component",
      "get_srcComponent() != 1" in hb)
check("the HEARTBEAT branch filters on MAV_AUTOPILOT_INVALID",
      "MAV_AUTOPILOT_INVALID" in hb)
check("and it skips rather than falling through",
      "continue" in hb.split("MAV_AUTOPILOT_INVALID")[1].split("\n", 3)[1]
      or "continue" in hb)
check("armed is still set from base_mode when it does pass",
      "MAV_MODE_FLAG_SAFETY_ARMED" in hb)

# ---- the gate live_node puts in front of the frames ----

print("\nthe frame gate reads it as three states, not two")
ln = open(os.path.join(HERE, "vps_vio", "live_node.py")).read()
check("the gate tests `armed is True`", "if armed is True:" in ln,
      "truthiness would make None mean disarmed-but-also-unknown")
check("an unknown state does not record", "elif not a.frames_any_time:" in ln)
check("Telemetry still starts from a known-safe default",
      "armed=False" in open(os.path.join(HERE, "vps_vio",
                                         "record_flight.py")).read(),
      "False, not None - the bench must not record")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
