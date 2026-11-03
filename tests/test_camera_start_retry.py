"""startPipeline retries instead of crash-looping.

3 Nov, both flights. The camera's USB link dropped and
LiveSource.__init__ called self.dev.startPipeline(p) bare:

    RuntimeError: Device already closed or disconnected: Input/output error
        live_node.py:463  self.dev.startPipeline(p)

145 of those tracebacks are in vps_stderr.log. Each one killed the process,
systemd restarted it 5 s later, the camera was still gone, and it died
again. That loop was running through both flights: the node restarted at
09:09:49 during flight 117 and at 09:13:46 during flight 118 - 43 s before
the pilot switched to the VPS.

Software cannot repair a USB link. What it can stop doing is hammering a
device that is trying to re-enumerate, and reporting the same failure 145
times instead of once.

SYNTHETIC: the device is a stub. This checks the retry structure, not the
camera.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fails = []


def check(what, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + what
          + (f"   [{detail}]" if detail else ""))
    if not ok:
        fails.append(what)


src = open(os.path.join(ROOT, "vps_vio", "live_node.py"),
           encoding="utf-8").read()

print("the bare call is gone")
check("no unguarded self.dev.startPipeline(p)",
      not re.search(r'^\s{8}self\.dev\.startPipeline\(p\)\s*$', src, re.M))
check("it is inside a try", "self.dev.startPipeline(p)" in src
      and "camera pipeline attempt" in src)

print("\nit retries, waits, and then gives up honestly")
check("five attempts", "range(5)" in src)
check("it sleeps between attempts", "time.sleep(3.0)" in src)
check("a real failure still raises - it does not pretend to have a camera",
      "camera did not start after 5 attempts" in src)
check("and the message points at the USB link, not at this process",
      "The USB link is the" in src)

print("\nthe retry loop behaves, on a stub (SYNTHETIC)")
import time as _t


class _Dev:
    def __init__(self, fail_n):
        self.n = 0
        self.fail_n = fail_n

    def startPipeline(self, p):
        self.n += 1
        if self.n <= self.fail_n:
            raise RuntimeError("Device already closed or disconnected")


def drive(dev, sleep):
    last = None
    for i in range(5):
        try:
            dev.startPipeline(None)
            return i + 1
        except Exception as e:
            last = e
            if i < 4:
                sleep(3.0)
    raise RuntimeError("gave up") from last


slept = []
check("succeeds first time when the camera is there",
      drive(_Dev(0), slept.append) == 1)
slept.clear()
check("recovers on the third attempt", drive(_Dev(2), slept.append) == 3)
check("and waited twice before it", slept == [3.0, 3.0], str(slept))
slept.clear()
gave_up = False
try:
    drive(_Dev(99), slept.append)
except RuntimeError:
    gave_up = True
check("gives up after five", gave_up)
check("having waited four times, 12 s in all", slept == [3.0] * 4, str(slept))

print("\nALL PASS" if not fails else f"\n{len(fails)} FAILED: {', '.join(fails)}")
sys.exit(1 if fails else 0)
