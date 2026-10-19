"""Where the origin may come from, and when the receiver is forbidden.

The pilot's rule: while the aircraft is on the VPS the origin comes from the
typed point and nothing else. A receiver that happens to have a fix sitting on
the bench before arming must not quietly become the origin, because every
position afterwards is measured from it.

13 Oct caught this the other way round - the receiver was winning whenever it
had a fix, so the aircraft picked up a stale site by itself after the pilot
drove to a new one. This is the same failure with the sign reversed, and it is
worse: a wrong origin does not look wrong, it looks like it is working.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "vps_vio"))
from live_node import choose_origin  # noqa: E402

CLI = (74.500000, 30.500000)
SCR = (74.613182, 30.896742)
GPS = (74.600000, 30.900000)
ON_GPS, ON_VPS, UNKNOWN = 0, 1, None

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def got(cli, scr, primary, fix):
    o, why = choose_origin(cli, scr, primary, fix)
    return o, why


print("the typed point wins, on either source")
for primary, label in ((ON_VPS, "on the VPS"), (ON_GPS, "on the receiver")):
    o, why = got(None, SCR, primary, GPS)
    check(f"SCR_USER used {label}", o == SCR, f"{o} / {why}")

print("\nan explicit --origin beats everything")
for primary in (ON_VPS, ON_GPS, UNKNOWN):
    o, _ = got(CLI, SCR, primary, GPS)
    check(f"--origin wins with GPS_PRIMARY={primary}", o == CLI, str(o))

print("\nON THE VPS the receiver is forbidden, fix or no fix")
o, why = got(None, None, ON_VPS, GPS)
check("refused even with a good fix available", o is None, f"{o} / {why}")
check("and says why", "VPS" in why, why)
o, _ = got(None, None, ON_VPS, None)
check("refused with no fix either", o is None)

print("\nNOT KNOWING is treated as being on the VPS")
o, why = got(None, None, UNKNOWN, GPS)
check("GPS_PRIMARY unreadable -> refused", o is None, f"{o} / {why}")
check("and says why", "could not be read" in why, why)

print("\non the receiver, the receiver is allowed - that is what was asked for")
o, why = got(None, None, ON_GPS, GPS)
check("GPS used when GPS_PRIMARY=0", o == GPS, f"{o} / {why}")
o, why = got(None, None, ON_GPS, None)
check("but not before it has a fix", o is None, f"{o} / {why}")

print("\nthe refusal is never silent")
for args in ((None, None, ON_VPS, GPS), (None, None, UNKNOWN, GPS),
             (None, None, ON_GPS, None)):
    o, why = got(*args)
    check(f"refusal carries a reason ({args[2]})", o is None and bool(why), why)

print("\nno path returns a receiver position while on the VPS")
bad = [(s, f) for s in (None, SCR) for f in (None, GPS)
       if (lambda r: r is not None and r == f)(got(None, s, ON_VPS, f)[0])]
check("exhaustive: never the receiver on the VPS", not bad, str(bad))

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
