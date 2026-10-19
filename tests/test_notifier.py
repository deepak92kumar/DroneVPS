"""The VPS notifications, because the pilot reads a transmitter, not a terminal.

The marker daemon has thirty-one STATUSTEXT lines. The VPS path had none, so
from the air there was no way to tell a healthy VPS from a dead one. These
pin the two rules that keep it useful instead of noisy: changes are announced
once, and the heartbeat is paced.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "vps_vio"))
from live_node import Notifier  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


class FakeMav:
    def __init__(self):
        self.lines = []

    def statustext_send(self, sev, text):
        self.lines.append((sev, text.decode()))


class FakeLink:
    def __init__(self):
        self.mav = FakeMav()


class Vel:
    def __init__(self, valid=True, reason="ok", speed=1.5):
        self.valid, self.reason, self.speed_mps = valid, reason, speed


class Node:
    def __init__(self, n_fix=0):
        self.n_fix = n_fix


def est(tracks=250, sigma=6.0, alt=110.0, gyro_n=500, vel=None):
    return dict(tracks=tracks, sigma=sigma, alt=alt, gyro_n=gyro_n,
                vel=vel or Vel())


print("a state change is announced once, not every frame")
lk = FakeLink()
n = Notifier(lk, period_s=1e9)
for _ in range(5):
    n.tick(0.0, est(tracks=250), Node())
good = [t for _, t in lk.mav.lines if "TRACK OK" in t]
check("tracking OK announced once over 5 frames", len(good) == 1,
      f"{len(good)} times")

before = len(lk.mav.lines)
for _ in range(5):
    n.tick(0.0, est(tracks=4), Node())
lost = [t for _, t in lk.mav.lines[before:] if "TRACK LOST" in t]
check("losing tracking announced once", len(lost) == 1, f"{len(lost)} times")
check("and it is critical, not chatter",
      any(s == Notifier.SEV_CRIT for s, t in lk.mav.lines if "TRACK LOST" in t))

before = len(lk.mav.lines)
for _ in range(5):
    n.tick(0.0, est(tracks=250), Node())
check("recovering is announced too",
      any("TRACK OK" in t for _, t in lk.mav.lines[before:]))

print("\nthe heartbeat is paced, not per frame")
lk = FakeLink()
n = Notifier(lk, period_s=10.0)
for i in range(300):
    n.tick(i * 0.1, est(), Node(n_fix=3))      # 30 s of frames at 10 Hz
beats = [t for _, t in lk.mav.lines if t.startswith("VPS t")]
check("about one heartbeat per 10 s over 30 s", 3 <= len(beats) <= 4,
      f"{len(beats)} beats")

print("\nevery line fits what the protocol will carry")
lk = FakeLink()
n = Notifier(lk, period_s=0.0)
n.tick(0.0, est(tracks=1, vel=Vel(False, "flow 2.00s stale (> 0.6)")), Node())
n.tick(1.0, est(gyro_n=0), Node())
n.event("origin", "x", "VPS ORIGIN SCR_USER1/2 (typed on the RC)", 6)
longest = max((len(t) for _, t in lk.mav.lines), default=0)
check("no line over 50 characters", longest <= 50, f"longest {longest}")
check("something was actually sent", len(lk.mav.lines) > 3,
      f"{len(lk.mav.lines)} lines")

print("\nthe things a pilot has to know are covered")
lk = FakeLink()
n = Notifier(lk, period_s=1e9)
n.tick(0.0, est(tracks=2), Node())
n.tick(0.1, est(vel=Vel(False, "no flow yet")), Node())
n.tick(0.2, est(gyro_n=0), Node())
text = " | ".join(t for _, t in lk.mav.lines)
for want in ("TRACK LOST", "VEL", "GYRO SILENT"):
    check(f"{want} reported", want in text)

print("\na dead camera IMU is not read as calm")
check("gyro_n 0 says SILENT, not OK", "GYRO SILENT" in text)

print("\ndisabled means silent")
lk = FakeLink()
n = Notifier(lk, period_s=0.0, enabled=False)
for i in range(20):
    n.tick(i, est(), Node())
check("nothing sent when disabled", not lk.mav.lines, f"{len(lk.mav.lines)}")
n2 = Notifier(None, period_s=0.0)
n2.tick(0.0, est(), Node())
check("no link is not a crash", n2.sent == 0)

print("\na link that throws does not take the flight down")
class Boom:
    class mav:
        @staticmethod
        def statustext_send(*a, **k):
            raise OSError("link gone")
n3 = Notifier(Boom(), period_s=0.0)
try:
    n3.tick(0.0, est(), Node())
    check("a broken link is swallowed", True)
except Exception as e:
    check("a broken link is swallowed", False, repr(e))

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
