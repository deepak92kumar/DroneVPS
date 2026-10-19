"""What reaches the flight controller when the pilot is on the real receiver.

The pilot's rule, in their words: "gps switch karne par vps ka koi data fc ko
nahi srif asli gps m9n wala jana chiye aur change ka massage bhi aana chiye."

Until 20 Oct neither half held. read_gps_primary ran ONCE, before the loop, and
the node then injected GPS_INPUT for the rest of the flight whatever the
transmitter said - and no source change was ever announced. Log 214 has 35 VPS
heartbeats in it and not one line naming a source.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "vps_vio"))
from live_node import SourceWatcher  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


class Param:
    def __init__(self, pid, val):
        self.param_id, self.param_value = pid, val


class FakeMav:
    def __init__(self):
        self.requests = 0

    def param_request_read_send(self, *a, **k):
        self.requests += 1


class FakeLink:
    """Hands out queued PARAM_VALUE messages, as the FC broadcasts them."""

    def __init__(self, queue=()):
        self.queue = list(queue)
        self.mav = FakeMav()
        self.target_system = self.target_component = 1

    def recv_match(self, type=None, blocking=False):
        return self.queue.pop(0) if self.queue else None


print("the parameter is read while the node runs, not once at startup")
lk = FakeLink()
w = SourceWatcher(lk, period_s=2.0)
check("nothing known before the first answer", w.poll(0.0) is None)
check("unknown still allows injection", w.on_vps() is True)
check("it asked the FC", lk.mav.requests == 1, str(lk.mav.requests))
w.poll(0.5)
check("and does not ask again inside the period", lk.mav.requests == 1,
      str(lk.mav.requests))
w.poll(2.5)
check("but does after it", lk.mav.requests == 2, str(lk.mav.requests))

print("\na switch to the real receiver stops injection")
lk = FakeLink([Param("GPS_PRIMARY", 1.0)])
w = SourceWatcher(lk, period_s=2.0)
check("GPS_PRIMARY=1 is the VPS", w.poll(0.0) == 1 and w.on_vps() is True)
lk.queue.append(Param("GPS_PRIMARY", 0.0))
check("GPS_PRIMARY=0 is the receiver", w.poll(3.0) == 0)
check("and the VPS must not inject", w.on_vps() is False)
lk.queue.append(Param("GPS_PRIMARY", 1.0))
w.poll(6.0)
check("switching back resumes it", w.on_vps() is True)

print("\nother parameters do not move the source")
lk = FakeLink([Param("GPS_TYPE", 0.0), Param("EK3_SRC1_POSXY", 3.0)])
w = SourceWatcher(lk, period_s=2.0)
w.value = 1
w.poll(0.0)
check("unrelated PARAM_VALUE is ignored", w.value == 1, str(w.value))

print("\nseveral queued messages: the LAST one wins")
lk = FakeLink([Param("GPS_PRIMARY", 1.0), Param("GPS_PRIMARY", 0.0)])
w = SourceWatcher(lk, period_s=2.0)
check("drains the whole queue in one poll", w.poll(0.0) == 0, str(w.value))
check("so a stale value cannot linger", w.on_vps() is False)

print("\nthe switch is counted, so a log can show it happened")
lk = FakeLink([Param("GPS_PRIMARY", 0.0)])
w = SourceWatcher(lk, period_s=2.0)
w.poll(0.0)
check("reads are counted", w.reads == 1, str(w.reads))

print("\nunknown fails TOWARDS sending, which is the deliberate choice")
# Injecting into GPS2 while the FC navigates on GPS1 is ignored - the primary
# is the primary and GPS_AUTO_SWITCH is 0. Going silent while actually on the
# VPS is what left the aircraft with no position at all on 20 Oct.
w = SourceWatcher(None, period_s=2.0)
check("no link at all is not a crash", w.poll(0.0) is None)
check("and it still allows injection", w.on_vps() is True)


class Boom(FakeLink):
    def recv_match(self, **k):
        raise OSError("link gone")


w = SourceWatcher(Boom(), period_s=2.0)
try:
    w.poll(0.0)
    check("a broken link is swallowed, not flown into", True)
except Exception as e:
    check("a broken link is swallowed, not flown into", False, repr(e))
check("and a broken link does not silence the VPS", w.on_vps() is True)

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
