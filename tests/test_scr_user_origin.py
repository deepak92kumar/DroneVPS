"""The typed take-off point: read it, and refuse the values that are not one.

13 Oct: the pilot typed coordinates for one site, drove to a second, retyped
nothing, and the aircraft picked up the NEW location by itself - the live
receiver was winning whenever it had a fix, so the typed point only ever
reached the flight controller's home. On the VPS the typed point must win.

This covers the reader and its validation. An unset SCR_USER reads 0, and 0,0
is a real place in the Gulf of Guinea, so falling through matters more than
reading successfully does.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "vps_vio"))
from pymavlink import mavutil                 # noqa: E402
import live_node                              # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


class FakeMsg:
    def __init__(self, pid, val):
        self.param_id = pid
        self.param_value = val

    def get_type(self):
        return "PARAM_VALUE"


class FakeLink:
    """A flight controller that reports whatever values the test chose."""
    def __init__(self, values, heartbeat=True):
        self.target_system = 1
        self.target_component = 1
        self.mav = self
        self._queue = [FakeMsg(k, v) for k, v in values.items()]
        self._hb = heartbeat

    def wait_heartbeat(self, timeout=None):
        return object() if self._hb else None

    def param_request_read_send(self, *a, **kw):
        pass

    def recv_match(self, type=None, blocking=False, timeout=None):
        return self._queue.pop(0) if self._queue else None

    def close(self):
        pass


def with_values(values, heartbeat=True):
    saved = mavutil.mavlink_connection
    mavutil.mavlink_connection = lambda *a, **kw: FakeLink(values, heartbeat)
    try:
        return live_node.read_scr_user("fake:endpoint", timeout_s=0.5)
    finally:
        mavutil.mavlink_connection = saved


print("a real typed point is returned as (lon, lat)")
r = with_values({"SCR_USER1": 30.784391, "SCR_USER2": 74.607185})
check("returns a pair", r is not None and len(r) == 2, repr(r))
check("lon first, lat second", r is not None and abs(r[0] - 74.607185) < 1e-6
      and abs(r[1] - 30.784391) < 1e-6, repr(r))

print("\nvalues that are not a position fall through")
cases = [
    ("both unset (0,0)", {"SCR_USER1": 0.0, "SCR_USER2": 0.0}),
    ("latitude unset", {"SCR_USER1": 0.0, "SCR_USER2": 74.607185}),
    ("longitude unset", {"SCR_USER1": 30.784391, "SCR_USER2": 0.0}),
    ("latitude out of range", {"SCR_USER1": 123.0, "SCR_USER2": 74.6}),
    ("longitude out of range", {"SCR_USER1": 30.78, "SCR_USER2": 274.6}),
    ("not a number", {"SCR_USER1": float("nan"), "SCR_USER2": 74.6}),
    ("infinite", {"SCR_USER1": 30.78, "SCR_USER2": float("inf")}),
    ("only one replied", {"SCR_USER1": 30.784391}),
    ("neither replied", {}),
]
for name, vals in cases:
    check(f"{name} -> None", with_values(vals) is None, repr(with_values(vals)))

print("\nno link is not an origin either")
check("no heartbeat -> None",
      with_values({"SCR_USER1": 30.78, "SCR_USER2": 74.6}, heartbeat=False) is None)

saved = mavutil.mavlink_connection


def boom(*a, **kw):
    raise OSError("port busy")


mavutil.mavlink_connection = boom
try:
    check("a failed connection -> None, not an exception",
          live_node.read_scr_user("fake:endpoint", timeout_s=0.5) is None)
finally:
    mavutil.mavlink_connection = saved

print("\n0,0 is a real place, which is why it must be refused")
check("Gulf of Guinea not accepted",
      with_values({"SCR_USER1": 0.0, "SCR_USER2": 0.0}) is None)

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
