"""The gap between two frames, and what fills it.

Flow speaks once a frame. At the 17 Hz log 214 actually ran, that leaves 60 ms
per gap in which nothing is measured, and log 214 shows what the gap costs:
"TRACK LOST 17" inside the first second, and while flow was down the estimate
did not move at all. A frozen position is not a neutral answer - the flight
controller reads it as an aircraft standing still, which is a confident claim
about something nobody saw.

So: gyro fills the ANGLE (derotation, already covered by the flow tests) and
acceleration fills the TRANSLATION. These pin the translation half.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "vps_vio"))
from live_node import VpsNode, accel_to_ne  # noqa: E402
from map_fuse import MapFuser  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def near(a, b, tol=1e-6):
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- #
print("the rotation puts acceleration in the same frame as the flow")
# Nose north. Body forward is -cam_y, so -1 on cam_y is +1 forward = north.
an, ae = accel_to_ne(0.0, -1.0, 0.0)
check("nose north, -cam_y -> +north", near(an, 1.0) and near(ae, 0.0), f"{an:.3f},{ae:.3f}")
# +cam_x is body right, which with the nose north is east.
an, ae = accel_to_ne(1.0, 0.0, 0.0)
check("nose north, +cam_x -> +east", near(an, 0.0) and near(ae, 1.0), f"{an:.3f},{ae:.3f}")
# Nose east: forward becomes east, right becomes south.
an, ae = accel_to_ne(0.0, -1.0, 90.0)
check("nose east, forward -> +east", near(an, 0.0) and near(ae, 1.0), f"{an:.3f},{ae:.3f}")
an, ae = accel_to_ne(1.0, 0.0, 90.0)
check("nose east, right -> -north", near(an, -1.0) and near(ae, 0.0), f"{an:.3f},{ae:.3f}")
# Rotation preserves length at every heading, which a sign error would not.
bad = [y for y in range(0, 360, 7)
       if not near(np.hypot(*accel_to_ne(0.3, -0.7, y)), np.hypot(0.3, 0.7), 1e-9)]
check("magnitude preserved at every heading", not bad, str(bad[:5]))

print("\nvariance grows faster for a step nobody saw")
a, b = MapFuser(flow_drift_frac=0.07, fix_sigma_m=12.0), MapFuser(flow_drift_frac=0.07, fix_sigma_m=12.0)
a.predict(1.0, 0.0)
b.predict(1.0, 0.0, trust=3.0)
check("dead-reckoned step is doubted more", b.var > a.var, f"{a.var:.4f} vs {b.var:.4f}")
check("and the position moved the same either way",
      near(float(a.pos[0]), float(b.pos[0])), f"{a.pos} {b.pos}")


# --------------------------------------------------------------------------- #
class FakeTracker:
    """Returns the queued pixel displacement, or None for a dropout."""

    def __init__(self, seq):
        self.seq = list(seq)
        self.n = 0
        self.depth_med_m = None
        self.depth_dropped = 0

    # **kw since 25 Oct: the node now hands the gyro rotation in as
    # derot_deg so each track can be derotated before the fit. A fake
    # that refuses unknown keywords would make every such change look
    # like a gap-fill regression.
    def step(self, img, depth=None, **kw):
        d = self.seq[min(self.n, len(self.seq) - 1)]
        self.n += 1
        return (None, 0) if d is None else (np.array(d, float), 200)


class FakeWorker:
    def submit(self, *a, **k):
        return False

    def poll(self):
        return None

    def stop(self):
        pass


def node():
    n = VpsNode.__new__(VpsNode)
    from velocity_bridge import VelocityBridge
    n.cache = None
    n.focal = 791.723
    n.focal_y = 791.583
    n.fuser = MapFuser(flow_drift_frac=0.07, fix_sigma_m=12.0)
    n.worker = FakeWorker()
    n.fix_period = 1e9
    n._next_fix = 1e9
    n.prev_att = None
    n.prev_t = None
    n.n_fix = n.n_flow = n.n_dr = 0
    n.n_step_rejected = 0
    n._step_refusals = 0
    # The flow velocity's baseline, also skipped with __init__. See
    # VpsConfig.flow_vel_frames: the velocity is measured over the span of the
    # last N accepted displacements, not over one frame.
    import collections as _c
    from config import VpsConfig as _Cfg
    _cfg = _Cfg()
    n._vel_frames = int(_cfg.flow_vel_frames)
    # The baseline lengthens with height; see VpsConfig.flow_vel_frames_high.
    n._vel_frames_high = int(_cfg.flow_vel_frames_high)
    n._vel_alt_m = float(_cfg.flow_vel_alt_m)
    n._vel_alt_span_m = 15.0
    # The step gate's recorded input, also skipped with __init__.
    n.step_px = None
    n.step_action = "ok"
    n._vel_cum_n = n._vel_cum_e = 0.0
    n._vel_hist = _c.deque(maxlen=max(n._vel_frames, n._vel_frames_high) + 1)
    # The bare node skips __init__, so the derotation source has to be set
    # here. "gyro" is the flying default; this fixture's frames carry no gyro
    # delta, so it falls back to the FC attitude on its own.
    n.derotate = "gyro"
    n.n_derot_gyro = n.n_derot_quat = n.n_derot_fc = n.n_derot_both = 0
    n.derot_dis_roll = n.derot_dis_pitch = None
    from flow_odom import TiltValidator
    n.val_gyro = TiltValidator("gyro")
    n.val_quat = TiltValidator("quat")
    n.vel = VelocityBridge()
    n.set_origin(74.613182, 30.896742)
    return n


def frame(t, accel=(), alt=100.0, yaw=0.0):
    return dict(t=t, img=np.zeros((80, 128), np.uint8), alt=alt, yaw=yaw,
                roll=0.0, pitch=0.0, accel=list(accel), accel_n=len(accel))


print("\nflow drops out and the position carries on")
n = node()
n.tracker = FakeTracker([(0.0, 0.0), (12.0, 0.0), (12.0, 0.0), None, None, None])
# Two real flow frames first, so the bridge has a velocity to carry.
hz, dt = 17.0, 1.0 / 17.0
t = 0.0
for i in range(3):
    # ~6 accelerometer samples per gap, which is 100 Hz against 17 Hz frames.
    acc = [(t + j * 0.01, 0.0, 0.0, 0.0) for j in range(6)]
    n.step(frame(t, acc))
    t += dt
moved_before = float(np.linalg.norm(n.fuser.pos))
check("flow moved the estimate", moved_before > 0.01,
      f"{n.fuser.pos[0]:.3f} N / {n.fuser.pos[1]:.3f} E")
check("n_flow counted the flow frames", n.n_flow == 2, str(n.n_flow))
check("nothing dead reckoned yet", n.n_dr == 0, str(n.n_dr))

pos_at_dropout = n.fuser.pos.copy()
var_at_dropout = n.fuser.var
for i in range(3):
    acc = [(t + j * 0.01, 0.0, 0.0, 0.0) for j in range(6)]
    n.step(frame(t, acc))
    t += dt
check("the dropout was dead reckoned", n.n_dr == 3, str(n.n_dr))
check("position kept moving through the dropout",
      float(np.linalg.norm(n.fuser.pos - pos_at_dropout)) > 0.01,
      f"moved {float(np.linalg.norm(n.fuser.pos - pos_at_dropout)):.3f} m "
      f"with no flow at all")
check("and it says it is less sure", n.fuser.var > var_at_dropout,
      f"{var_at_dropout:.4f} -> {n.fuser.var:.4f}")

print("\na dropout longer than the bridge allows stops dead reckoning")
n2 = node()
n2.tracker = FakeTracker([(0.0, 0.0), (12.0, 0.0), None])
t = 0.0
for i in range(2):
    n2.step(frame(t))
    t += dt
n2.step(frame(t))
dr_short = n2.n_dr
# Now jump well past flow_timeout_s. The velocity is stale, so it must refuse.
t += n2.vel.flow_timeout_s + 1.0
before = n2.fuser.pos.copy()
n2.step(frame(t))
check("stale flow is not dead reckoned", n2.n_dr == dr_short, f"{dr_short} -> {n2.n_dr}")
check("and the position did not jump",
      float(np.linalg.norm(n2.fuser.pos - before)) < 1e-9,
      f"moved {float(np.linalg.norm(n2.fuser.pos - before)):.4f} m")

print("\nacceleration reaches the bridge, 100 Hz against 17 Hz frames")
n3 = node()
n3.tracker = FakeTracker([(0.0, 0.0), (12.0, 0.0), (12.0, 0.0)])
n3.tracker = FakeTracker([(0.0, 0.0)] + [(12.0, 0.0)] * 9)
t = 0.0
for i in range(10):
    acc = [(t + j * 0.01, 0.2, -0.5, 9.8) for j in range(6)]
    n3.step(frame(t, acc))
    t += dt
# The bridge integrates nothing before the first flow sample, so the first two
# frames are warm-up by design. After that every sample has to count.
check("every sample integrated once flow has started",
      n3.vel.accel_steps >= 8 * 6, f"{n3.vel.accel_steps} steps, 48 after warm-up")
check("more accel steps than flow updates",
      n3.vel.accel_steps > n3.vel.flow_updates,
      f"{n3.vel.accel_steps} accel vs {n3.vel.flow_updates} flow")

print("\na silent accelerometer is not read as a standstill")
n4 = node()
n4.tracker = FakeTracker([(0.0, 0.0), (12.0, 0.0)])
t = 0.0
for i in range(2):
    n4.step(frame(t, ()))          # no samples at all
    t += dt
check("no samples -> no accel steps", n4.vel.accel_steps == 0, str(n4.vel.accel_steps))
check("and accel_n reports zero, not a fake OK",
      n4.step(frame(t, ()))["accel_n"] == 0)

print("\nyaw missing means the rotation is unknown, so nothing is pushed")
n5 = node()
n5.tracker = FakeTracker([(0.0, 0.0)])
f = frame(0.0, [(0.0, 1.0, 1.0, 9.8)])
f["yaw"] = None
check("a frame with no yaw is refused outright", n5.step(f) is None)

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
