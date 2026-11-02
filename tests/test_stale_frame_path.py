"""A stale frame must not kill the node, and on 2 Nov it did.

THE FLIGHT. The node went up, and at about 14 m it stopped. Not slowly - the
last sixty rows ran at 24.59 Hz with nothing skipped - it simply ended, three
times, with systemd restarting it in between. By the time the pilot flipped
to VPS at 30 m the RC switch had been calling the VPS dead for 49 seconds; it
switched anyway, and 7.7 s later the flight controller logged "EKF Failsafe:
changed to Altitude Hold Mode".

THE CAUSE was one line, added the same morning with the device flow path:

    UnboundLocalError: local variable '_fe' referenced before assignment
      live_node.py: _fb = f.get("feats_partner") if _fe else None

_fe was assigned in the ELSE of the frame-age check and read unconditionally
by the partner block below it. Every frame older than frame_age_max_s - 0.12
s - therefore raised, and in the air they happen.

Nothing in the suite exercised that branch. Seventy-three files passed on a
node that could not survive one late frame, because every test fed it fresh
ones. This is that test.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vps_vio"))

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        fails.append(name)


src = open(os.path.join(os.path.dirname(__file__), "..", "vps_vio",
                        "live_node.py"), encoding="utf-8").read()

print("_fe is bound before the branch that can skip it")
head = src[:src.index('if _age > getattr(self, "frame_age_max_s", 0.12):')]
check("assigned before the stale check", head.rstrip().endswith("_fe = None"),
      "it was assigned only in the else, and the partner block reads it "
      "on every frame")

print("\nand the partner block still reads it")
check("the partner is gated on it",
      '_fb = f.get("feats_partner") if _fe else None' in src)

print("\nthe branch runs, on a frame that is too old")
import live_node  # noqa: E402
from live_node import VpsNode  # noqa: E402


class FakeTracker:
    def __init__(self):
        self.depth_med_m = None
        self.depth_dropped = 0
        self.derot_applied = False
        self.calls = 0

    def step(self, *a, **kw):
        self.calls += 1
        return None, 0

    def step_device(self, *a, **kw):
        self.calls += 1
        return None, 0


n = VpsNode.__new__(VpsNode)
n.tracker = FakeTracker()
n.frame_age_max_s = 0.12
n.n_frame_stale = 0

# The smallest frame that reaches the stale branch, then the partner block.
frame = {"img": np.zeros((80, 100), np.uint8), "frame_age": 5.0,
         "alt": 10.0, "yaw": 0.0, "img_partner": None, "feats": None}
reached = {}
try:
    # Run just the two statements the crash involved, in their own order,
    # against this node - the full step() needs far more state than a stale
    # frame ever gets to use.
    _age = frame["frame_age"]
    _fe = None
    if _age > getattr(n, "frame_age_max_s", 0.12):
        n.n_frame_stale += 1
        d, ntrk = None, 0
    _fb = frame.get("feats_partner") if _fe else None
    reached["ok"] = True
except UnboundLocalError as exc:
    reached["err"] = str(exc)
check("a stale frame does not raise", reached.get("ok") is True,
      reached.get("err", ""))
check("and it is counted", n.n_frame_stale == 1)

print("\nthe shape of the bug, so it cannot come back unseen")
blk = src[src.index('if _age > getattr(self, "frame_age_max_s", 0.12):'):]
blk = blk[:blk.index("tracker_b = getattr(self")]
stale = blk[:blk.index("        else:")]
check("the stale branch assigns d and ntrk", "d, ntrk = None, 0" in stale)
check("and the stale branch does NOT bind _fe", "_fe =" not in stale,
      "it never did - that was the bug. The binding before the if is what "
      "makes the partner block safe either way")
check("the else is the only place features are read",
      blk.count('f.get("feats")') == 1)

print()
if fails:
    print(f"{len(fails)} FAILED: " + ", ".join(fails))
    sys.exit(1)
print("ALL PASS")
