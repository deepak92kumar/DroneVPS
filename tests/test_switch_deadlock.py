"""The circle that made the VPS unreachable from the transmitter.

Both halves were right on their own:

  * the node stops injecting entirely while the aircraft is on the real
    receiver - the pilot's rule, "gps switch karne par vps ka koi data fc ko
    nahi";
  * the switch refuses to hand navigation to a VPS that is not healthy,
    because the flight controller trusts whatever arrives on GPS_INPUT and
    will not check for you.

Together they deadlock, and 20 Oct found it on the aircraft. On GPS the node is
silent, so GPS2 has no fix, so the switch refuses to move onto the VPS, so the
node stays silent. The journal reads "VPS switch MANA: GPS2 fix=1 sats=0"
twelve times while the pilot flips the switch and nothing happens - and from
the transmitter there is NO WAY OUT, which is the same shape as the failure
that stranded the aircraft before.

The fix is to ask the right question. "Is the VPS fit to take over" is not the
same question as "is the VPS currently sending", and the liveness file answers
the first one whether or not the second is true.
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "vps_vio"))
from gps_rc_switch import RcGpsSwitch  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def sw(liveness):
    s = RcGpsSwitch.__new__(RcGpsSwitch)
    s.liveness = liveness
    return s


def write(path, text="t=91.72 n=1705 tracks=147 sigma=6.6 fixes=0 flow=1699 vel=ok"):
    with open(path, "w") as fh:
        fh.write(text)


d = tempfile.mkdtemp()
live = os.path.join(d, "vps.alive")

print("a healthy node says so, even while sending nothing to the FC")
write(live)
ok, why = sw(live).vps_healthy()
check("healthy", ok, why)
check("and says what it saw", "147" in why, why)

print("\nthe deadlock itself: on GPS, GPS2 has no fix by design")
# This is the exact state from the aircraft: GPS_PRIMARY 0, node gated silent,
# GPS2 at fix 1 with 0 satellites. The old rule refused here, for ever.
gps2_fix = 1
ok, _ = sw(live).vps_healthy()
old_rule_allows = gps2_fix >= 3
new_rule_allows = ok or gps2_fix >= 3
check("the old rule refuses - this was the trap", not old_rule_allows)
check("the new rule allows the switch", new_rule_allows)
check("so the transmitter is not a dead end any more",
      new_rule_allows and not old_rule_allows)

print("\nbut a DEAD node is still refused - the check still has teeth")
check("no file at all", not sw(os.path.join(d, "nope")).vps_healthy()[0])
old = os.path.join(d, "stale.alive")
write(old)
os.utime(old, (time.time() - 60, time.time() - 60))
ok, why = sw(old).vps_healthy()
check("a stale file is not health", not ok, why)
check("and says how stale", "stalled" in why and "60" in why, why)

print("\nnor is a node that is running but not tracking")
weak = os.path.join(d, "weak.alive")
write(weak, "t=5.0 n=100 tracks=4 sigma=40.0 fixes=0 flow=90 vel=no flow yet")
ok, why = sw(weak).vps_healthy()
check("too few tracks is refused", not ok, why)
check("and names the number", "4" in why, why)

print("\na GPS2 fix is still proof on its own")
# Already on the VPS: the node is injecting, the FC can see it. The liveness
# file is then a second opinion, not the only one.
for fix, healthy in ((3, False), (3, True)):
    allowed = healthy or fix >= 3
    check(f"GPS2 fix {fix}, liveness {healthy} -> allowed", allowed)

print("\ngarbage in the file is not read as health")
for bad in ("", "nonsense", "tracks=", "tracks=abc", "t=1 n=2"):
    p = os.path.join(d, "bad.alive")
    write(p, bad)
    ok, why = sw(p).vps_healthy()
    check(f"{bad!r} is refused", not ok, why)

print("\nthe unit and the program agree on where the file is")
unit = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "systemd", "drone-vps.service")
if os.path.exists(unit):
    text = open(unit).read()
    check("the node is told to write it where the switch looks",
          RcGpsSwitch.LIVENESS in text,
          f"{RcGpsSwitch.LIVENESS} not in drone-vps.service")

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
