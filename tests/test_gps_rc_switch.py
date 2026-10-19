"""The three-position GPS/VPS switch, and why its middle is blank.

This switch decides which position source the aircraft flies on, so each
position is pinned here rather than trusted.

RC6 travels 1045 / 1495 / 1945, read off the flight controller as RC6_MIN and
RC6_MAX. Low is the real GPS, high is the VPS, and the centre detent commands
NOTHING - it holds whatever is selected. A two-way threshold would have made
the centre mean "real GPS", so passing through it on the way to the VPS would
have commanded a source change nobody asked for.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "vps_vio"))
from gps_rc_switch import RcGpsSwitch  # noqa: E402

GPS, VPS, HOLD = 0, 1, None
fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def sw(low=1270, high=1720):
    s = RcGpsSwitch.__new__(RcGpsSwitch)
    s.low, s.high = low, high
    return s


s = sw()
label = {GPS: "real GPS", VPS: "VPS", HOLD: "hold"}

print("the three detents")
for v, want in ((1045, GPS), (1495, HOLD), (1945, VPS)):
    got = s.decide(v)
    check(f"{v} us -> {label[want]}", got == want, f"got {label[got]}")

print("\nthe guard bands, exactly")
for v, want in ((1269, GPS), (1270, GPS), (1271, HOLD),
                (1719, HOLD), (1720, VPS), (1721, VPS)):
    got = s.decide(v)
    check(f"{v} us -> {label[want]}", got == want, f"got {label[got]}")

print("\nthe whole dead zone is dead")
dead = [v for v in range(1271, 1720) if s.decide(v) is not HOLD]
check("nothing between the bands decides anything", not dead,
      f"{len(dead)} values decided something" if dead else "450 us wide")

print("\nthe centre cannot be reached by accident from either end")
# Sweeping low -> high must not pass through a VPS command before 1720,
# and sweeping high -> low must not pass through a GPS command before 1270.
up = [s.decide(v) for v in range(1045, 1721)]
check("sweeping up, no VPS before 1720", VPS not in up[:-1],
      f"first VPS at {1045 + up.index(VPS)}" if VPS in up else "")
down = [s.decide(v) for v in range(1945, 1269, -1)]
check("sweeping down, no GPS before 1270", GPS not in down[:-1],
      f"first GPS at {1945 - down.index(GPS)}" if GPS in down else "")

print("\nfull travel is covered - no pulse width is undefined")
undef = [v for v in range(900, 2101) if s.decide(v) not in (GPS, VPS, HOLD)]
check("every width 900-2100 decides something or holds", not undef)

print("\na narrower switch still works")
s2 = sw(low=1200, high=1800)
check("1045 still real GPS", s2.decide(1045) == GPS)
check("1500 still holds", s2.decide(1500) is HOLD)
check("1945 still VPS", s2.decide(1945) == VPS)

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
