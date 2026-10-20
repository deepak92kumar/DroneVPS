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


class Worker:
    def __init__(self, min_alt_m=60.0):
        self.min_alt_m = min_alt_m


class Node:
    def __init__(self, n_fix=0, min_alt_m=60.0):
        self.n_fix = n_fix
        self.worker = Worker(min_alt_m) if min_alt_m is not None else None


def est(tracks=250, sigma=6.0, alt=110.0, gyro_n=500, vel=None, accel_n=500,
        src=None):
    return dict(tracks=tracks, sigma=sigma, alt=alt, gyro_n=gyro_n,
                accel_n=accel_n, src=src, vel=vel or Vel())


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

print("\nthe accelerometer has its own line - a dead gyro does not cover it")
lk = FakeLink()
n = Notifier(lk, period_s=1e9)
n.tick(0.0, est(accel_n=0), Node())
t2 = " | ".join(t for _, t in lk.mav.lines)
check("accel_n 0 says SILENT", "ACCEL SILENT" in t2, t2)
check("and it is critical", any(s == Notifier.SEV_CRIT for s, t in lk.mav.lines
                               if "ACCEL" in t))
lk = FakeLink()
n = Notifier(lk, period_s=1e9)
n.tick(0.0, est(accel_n=120), Node())
check("a live accelerometer says OK",
      any("ACCEL OK" in t for _, t in lk.mav.lines))

print("\n'no map fix' says WHY, because on the ground it is not a fault")
lk = FakeLink()
n = Notifier(lk, period_s=1e9)
n.tick(0.0, est(alt=0.0), Node(n_fix=0, min_alt_m=60.0))
low = [t for _, t in lk.mav.lines if "MAP" in t]
check("below the minimum it gives the number", low and "60" in low[0], str(low))
check("and it is not screamed as a warning",
      all(s == Notifier.SEV_INFO for s, t in lk.mav.lines if "MAP" in t), str(low))
lk = FakeLink()
n = Notifier(lk, period_s=1e9)
n.tick(0.0, est(alt=110.0), Node(n_fix=0, min_alt_m=60.0))
high = [(s, t) for s, t in lk.mav.lines if "MAP" in t]
check("high enough and still nothing IS a warning",
      high and high[0][0] == Notifier.SEV_WARN and "NO MAP FIX" in high[0][1],
      str(high))
check("a node with no worker does not crash the notifier",
      Notifier(FakeLink(), period_s=1e9).tick(0.0, est(), Node(min_alt_m=None)) is None)

print("\nthe pilot reads English, in 50 characters")
# The pilot's instruction, 20 Oct: "massages jo hai english mein likho ye mana
# kay hai thora professional ke tarah kaam karo mai hindi mein chat kar raha hu
# iska matlab ye nahi ki mere ko english samajh nahi aayega". Chat language and
# cockpit language are not the same thing, and a line on a transmitter screen
# is a cockpit line. STATUSTEXT also truncates at 50, so a line that runs over
# does not just read badly, it arrives cut in half.
HINGLISH = ("nahi", "hai", "raha", "karo", "kar ", "jayega", "gaya", "mana",
            "asli", "ab ", "chal", "band", "kuch", "anjaan", "padha",
            "bhejenge", "zameen", "hawa")
lk = FakeLink()
n = Notifier(lk, period_s=0.0)
for case in (est(), est(tracks=3), est(vel=Vel(False, "flow 2.00s stale (> 0.6)")),
             est(gyro_n=0), est(accel_n=0), est(alt=0.0), est(src="GPS1")):
    n.tick(0.0, case, Node())
for text in ("VPS ORIGIN SCR_USER1/2 (typed on the RC)",
             "VPS SHEET MISS - NO MAP FIX",
             "VPS NO ORIGIN - SET SCR_USER1/2",
             "VPS ON - injecting as GPS2",
             "GPS1 ON - VPS data blocked",
             "SOURCE UNKNOWN - cannot read GPS_PRIMARY",
             "VPS AMSL UNKNOWN - pass --site-alt"):
    n.event(text[:6], text, text, Notifier.SEV_INFO)

lines = [t for _, t in lk.mav.lines]
over = [t for t in lines if len(t) > 50]
check("no line is truncated by the protocol", not over,
      str([(len(t), t) for t in over]))
hing = [t for t in lines
        if any(w in t.lower() for w in HINGLISH)]
check("every line is in English", not hing, str(hing))
check("and there are plenty of them to check", len(lines) > 15, str(len(lines)))
check("all printable ASCII", all(all(32 <= ord(c) < 127 for c in t) for t in lines))

print("\nthe source lines name the source AND what the other one is doing")
for t in ("VPS ON - injecting as GPS2", "GPS1 ON - VPS data blocked"):
    check(f"{t!r} says both halves",
          ("VPS" in t and "GPS" in t), t)

print("\nthe heartbeat carries which source is live")
lk = FakeLink()
n = Notifier(lk, period_s=0.0)
n.tick(0.0, est(src="VPS"), Node(n_fix=2))
beat = [t for _, t in lk.mav.lines if t.startswith("VPS t")]
check("the heartbeat names the source", beat and "VPS" in beat[-1], str(beat[-1:]))

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
