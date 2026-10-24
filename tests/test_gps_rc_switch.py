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



# --------------------------------------------------------- the 23 Oct flight

print("\nthe write that was skipped, and the read that caused it")

# WHAT HAPPENED. gps_rc_switch set GPS_PRIMARY=1, read back 0 from a
# PARAM_VALUE that predated its own write, and recorded applied=0. The pilot
# then asked for the receiver. want 0 matched the false applied 0, so the
# command was never sent, GPS_PRIMARY stayed 1, and the VPS node - which read
# it correctly - injected for another 238 s. The aircraft landed on the VPS
# with the transmitter asking for GPS. Both halves are pinned here.


class FakeParamLink:
    """A flight controller that answers PARAM_VALUE, plus a stale queue.

    `queued` are messages already in flight when a read starts, which is the
    normal state through the router: the VPS node re-requests GPS_PRIMARY
    every 0.5 s and every client gets a copy.
    """

    class _Msg:
        def __init__(self, pid, val):
            self.param_id, self.param_value = pid, val

    class _Mav:
        def __init__(self, outer):
            self.outer = outer

        def param_request_read_send(self, *a, **k):
            self.outer.reads += 1
            self.outer.pending.append(
                FakeParamLink._Msg("GPS_PRIMARY", float(self.outer.value)))

        def param_set_send(self, sysid, comp, name, value, ptype):
            self.outer.sets.append(int(value))
            if not self.outer.refuse_writes:
                self.outer.value = int(value)

    def __init__(self, value=0, queued=(), refuse_writes=False):
        self.value = int(value)
        self.pending = [FakeParamLink._Msg("GPS_PRIMARY", float(v)) for v in queued]
        self.sets = []
        self.reads = 0
        self.refuse_writes = refuse_writes
        self.target_system = self.target_component = 1
        self.mav = FakeParamLink._Mav(self)

    def recv_match(self, type=None, blocking=False, timeout=None):
        return self.pending.pop(0) if self.pending else None


def switcher(link, applied):
    s = RcGpsSwitch.__new__(RcGpsSwitch)
    s.low, s.high = 1270, 1720
    s.m = link
    s.applied = applied
    s.said = []
    s.say = lambda text: s.said.append(text)
    return s


# 1. A read must not return a value that was queued before the request.
link = FakeParamLink(value=1, queued=[0, 0])
s = switcher(link, applied=1)
got = s.get_param("GPS_PRIMARY")
check("a read drains the stale queue first", got == 1.0,
      f"queue held two 0s while the FC held 1; read returned {got}")

# 2. A write is verified, and a refused write is reported rather than assumed.
link = FakeParamLink(value=1, refuse_writes=True)
s = switcher(link, applied=1)
import time as _t
_sleep, _t.sleep = _t.sleep, lambda *_: None
got = s.set_and_verify(0, tries=3)
_t.sleep = _sleep
check("a write that does not take is retried", len(link.sets) == 3,
      f"{len(link.sets)} attempts")
check("and it reports what the FC actually holds, not what we asked",
      got == 1, f"asked for 0, reporting {got}")

# 3. THE SKIP IS GONE. A wrong cached belief must not swallow the command.
_src_lines = [ln for ln in open(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vps_vio", "gps_rc_switch.py")).read().splitlines()
    if not ln.lstrip().startswith("#")]
check("the applied cache no longer gates the write",
      not any("want == self.applied" in ln and "continue" in ln
              for ln in _src_lines)
      and not any(ln.strip() == "if want == self.applied:" for ln in _src_lines),
      "one wrong read used to become a wrong aircraft for the rest of the "
      "flight; the comment above it may still quote the old line")

# 4. And the command really is sent when belief and request already agree.
link = FakeParamLink(value=1)
s = switcher(link, applied=0)
_sleep, _t.sleep = _t.sleep, lambda *_: None
s.set_and_verify(0)
_t.sleep = _sleep
check("asking for the source we think is already set still writes",
      link.sets == [0], f"sets sent: {link.sets}")


# ------------------------------------------ the node's half of the same gap

print("\nthe node says how old its own reading is")

from live_node import SourceWatcher  # noqa: E402


class QuietLink:
    """A link that answers once and then goes silent."""

    class _Msg:
        def __init__(self, v):
            self.param_id, self.param_value = "GPS_PRIMARY", float(v)

    class _Mav:
        def param_request_read_send(self, *a, **k):
            pass

    def __init__(self, first=1):
        self.pending = [QuietLink._Msg(first)]
        self.target_system = self.target_component = 1
        self.mav = QuietLink._Mav()

    def recv_match(self, type=None, blocking=False, timeout=None):
        return self.pending.pop(0) if self.pending else None


w = SourceWatcher(QuietLink(first=1))
w.poll(100.0)
check("a fresh reading is not stale", w.value == 1 and not w.stale(100.0),
      f"value {w.value}, age {w.age_s(100.0)}")
check("still fine inside the window", not w.stale(102.9),
      f"age {w.age_s(102.9):.1f}s against {SourceWatcher.STALE_AFTER_S}s")
w.poll(104.0)
check("silence past the window is reported", w.stale(104.0),
      f"age {w.age_s(104.0):.1f}s - six missed answers")
check("and the value is KEPT, not cleared", w.value == 1,
      "going silent toward the FC while the aircraft is on the VPS is what "
      "left it with no position at all on 20 Oct")
check("a stale reading still permits injection", w.on_vps() is True,
      "this reports; it does not decide. The pilot's switch does not move "
      "on its own, so the last reading is still the best there is")

w2 = SourceWatcher(None)
check("never read at all has no age", w2.age_s(5.0) is None and not w2.stale(5.0),
      "unknown is a different state from stale, and must not read as stale")



# ------------------------------------------------- switch means switch

print("\nthe pilot's switch is never refused")

# THE PILOT'S RULE, 23 October: "agar aisa ho tho alt hold, gps kabhi nahi -
# samjo gps physically nikal gaya hai, jab tak switch na ho tab tak no gps."
#
# The script used to answer a VPS request with "VPS REFUSED: VPS node is not
# running" and leave the aircraft on the receiver. On flight 00000234 that
# fired over and over while the node crash-looped, and the aircraft stayed on
# GPS1 through all of it.
#
# If the VPS is asked for and the VPS is dead, the aircraft gets no position
# and the EKF drops it into ALT_HOLD. That is the right outcome: quietly
# flying a receiver nobody asked for is the system choosing GNSS on an
# aircraft built to fly without it, and under spoofing the receiver is the
# attack rather than the refuge.

_sw_src = open(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "vps_vio", "gps_rc_switch.py")).read()
_code = [ln for ln in _sw_src.splitlines() if not ln.lstrip().startswith("#")]

check("no path refuses the switch and skips the write",
      not any("REFUSED" in ln and "say" in ln for ln in _code),
      "the refusal is gone; the comment above it may still quote the old text")
check("and the health check no longer ends in 'continue'",
      not any("self.refusals += 1" in _code[i] and
              any("continue" in _code[j] for j in range(i + 1, min(i + 4, len(_code))))
              for i in range(len(_code))),
      "vps_healthy() is kept and still reported - it no longer decides")
check("a dead node is SAID, loudly, and the switch happens anyway",
      any("VPS DEAD" in ln and "switching anyway" in ln for ln in _code),
      "the pilot has to know the VPS is dead; they do not have to be overruled")
check("vps_healthy itself is still there to report with",
      "def vps_healthy" in _sw_src)

# And nothing anywhere may put the aircraft on GPS by itself.
check("the script never commands GPS1 on its own",
      not any(("set_primary(0)" in ln or "want = 0" in ln) for ln in _code),
      "GPS_PRIMARY only ever follows the RC channel")

_ln_src = open(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "vps_vio", "live_node.py")).read()
check("and the node only stops injecting when the pilot asked for GPS",
      "return self.value != 0" in _ln_src,
      "GPS_PRIMARY = 0 is the pilot's switch and the only thing that silences "
      "the injection")

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
