"""The altitude in GPS_INPUT, and the loop the first attempt created.

Log 214: the real receiver read 300.73 m and our GPS2 read 0.0-1.8 m, because
LiveSource never put alt_amsl in the frame dict and the sender fell back to the
RELATIVE altitude - which on the ground is zero.

The obvious fix - read GLOBAL_POSITION_INT.alt and send it - is worse, and the
bench caught it the same evening: GLOBAL_POSITION_INT is the EKF's altitude,
the EKF's origin altitude came from our own injection, so the altitude we sent
was the altitude we had just been handed. It read 2.43 m and was climbing from
0.01 m. A number that is wrong is a bug; a number that is wrong because it is
its own input is a loop.

So the reference has to come from outside the EKF: the pilot's --site-alt, or
the real receiver's own AMSL the first time it has a 3D fix. The relative
altitude added to it is barometer-driven (EK3_SRC1_POSZ is the baro), so it is
not something we fed in either.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "vps_vio"))
from live_node import AmslRef  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


print("with nothing to go on it refuses to invent an altitude")
r = AmslRef()
check("no reference yet", r.amsl(0.0) is None)
check("and no source to report", r.source is None)
check("a relative altitude alone is not an AMSL", r.amsl(50.0) is None)

print("\nthe pilot's own figure wins and is never overwritten")
r = AmslRef(fixed_m=216.0)
check("site elevation used", r.amsl(0.0) == 216.0, str(r.amsl(0.0)))
check("plus height above ground", r.amsl(100.0) == 316.0, str(r.amsl(100.0)))
check("source says where it came from", r.source == "--site-alt", str(r.source))
r.observe(3, 999.0, 0.0)
check("a receiver fix does not override it", r.amsl(0.0) == 216.0, str(r.amsl(0.0)))

print("\notherwise the REAL receiver sets it, once, on a 3D fix")
r = AmslRef()
r.observe(1, 300.73, 0.0)
check("a no-fix receiver is ignored", r.amsl(0.0) is None)
r.observe(2, 300.73, 0.0)
check("a 2D fix is ignored too", r.amsl(0.0) is None)
r.observe(3, 300.73, 0.0)
check("a 3D fix on the ground sets the ground", r.amsl(0.0) == 300.73,
      str(r.amsl(0.0)))
check("source names the receiver", "GPS1" in (r.source or ""), str(r.source))

print("\nthe reference is the GROUND, not wherever the aircraft was")
r = AmslRef()
# First 3D fix arrives at 40 m above home: ground is 340.7 - 40 = 300.7.
r.observe(3, 340.73, 40.0)
check("height at the moment of the fix is removed",
      abs(r.ref - 300.73) < 1e-9, f"{r.ref}")
check("and on the ground it reads the ground", abs(r.amsl(0.0) - 300.73) < 1e-9)
check("and at 120 m it reads 420.73", abs(r.amsl(120.0) - 420.73) < 1e-9,
      str(r.amsl(120.0)))

print("\nonce set it does not drift with the receiver")
r = AmslRef()
r.observe(3, 300.73, 0.0)
for wobble in (305.0, 295.0, 311.2, 288.4):
    r.observe(3, wobble, 0.0)
check("later fixes do not move it", abs(r.ref - 300.73) < 1e-9, str(r.ref))

print("\nmissing pieces are refused, not guessed around")
r = AmslRef()
r.observe(3, None, 0.0)
check("a fix with no altitude sets nothing", r.ref is None)
r.observe(3, 300.0, None)
check("no relative altitude sets nothing", r.ref is None)
r.observe(None, 300.0, 0.0)
check("no fix type sets nothing", r.ref is None)
r = AmslRef()
r.observe(3, 300.73, 0.0)
check("and a known reference still needs a height to add to",
      r.amsl(None) is None)

print("\nnothing here reads the EKF, which is the whole point")
# The loop was: EKF altitude in, the same altitude straight back out. Rather
# than grep the source for the name, pin the behaviour that makes a loop
# impossible: the only things that can move the reference are the receiver's
# own AMSL and a barometer-driven relative height, and the sent altitude is a
# pure function of the reference and the height - so feeding our own output
# back in cannot change it.
r = AmslRef()
r.observe(3, 300.73, 0.0)
sent = r.amsl(10.0)
for _ in range(50):
    # Pretend the EKF now reports OUR number back at us, at every level.
    r.observe(3, sent, 10.0)
    sent2 = r.amsl(10.0)
check("the altitude is a fixed point, not a ramp", abs(sent2 - sent) < 1e-9,
      f"{sent} -> {sent2}")
check("the reference never moved", abs(r.ref - 300.73) < 1e-9, str(r.ref))

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
